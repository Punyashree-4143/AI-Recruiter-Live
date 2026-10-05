import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def _score(value: Any, default: int = 50) -> int:
    try:
        return round(min(max(float(value), 0), 100))
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def _fallback_evaluation(candidate: dict[str, Any]) -> dict[str, Any]:
    breakdown = candidate.get("score_breakdown", {})
    required = _score(
        breakdown.get("required_skill_coverage", 0) * 100
    )
    preferred = _score(
        breakdown.get("preferred_skill_coverage", 0) * 100
    )
    title = _score(
        breakdown.get("title_match", 0) * 100
    )
    experience = _score(
        breakdown.get("experience_score", 0) * 100
    )
    fit_score = _score(
        (
            required * 0.50
            + preferred * 0.15
            + title * 0.20
            + experience * 0.15
        )
    )

    if fit_score >= 80:
        recommendation = "Strong Fit"
    elif fit_score >= 60:
        recommendation = "Moderate Fit"
    else:
        recommendation = "Weak Fit"

    matched = [
        *breakdown.get("matched_required_skills", []),
        *breakdown.get("matched_preferred_skills", []),
    ]
    missing = breakdown.get("missing_required_skills", [])

    return {
        "fit_score": fit_score,
        "technical_fit": required,
        "role_alignment": title,
        "experience_alignment": experience,
        "evidence": matched,
        "gaps": missing,
        "recommendation": recommendation,
    }


def _normalize_evaluation(
    result: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    fallback = _fallback_evaluation(candidate)

    return {
        "fit_score": _score(
            result.get("fit_score"),
            fallback["fit_score"],
        ),
        "technical_fit": _score(
            result.get("technical_fit"),
            fallback["technical_fit"],
        ),
        "role_alignment": _score(
            result.get("role_alignment"),
            fallback["role_alignment"],
        ),
        "experience_alignment": _score(
            result.get("experience_alignment"),
            fallback["experience_alignment"],
        ),
        "evidence": (
            _string_list(result.get("evidence"))
            or fallback["evidence"]
        ),
        "gaps": (
            _string_list(result.get("gaps"))
            or fallback["gaps"]
        ),
        "recommendation": str(
            result.get("recommendation")
            or fallback["recommendation"]
        ).strip(),
    }


def evaluate_candidate(
    job_description,
    candidate,
    role_profile=None,
):
    candidate_document = candidate.get("document", "")[:6000]
    candidate_metadata = candidate.get("metadata", {})
    score_breakdown = candidate.get("score_breakdown", {})
    role_context = role_profile or {}

    prompt = f"""
You are a senior recruiter evaluating a candidate for the supplied job.

JOB DESCRIPTION:
{job_description}

STRUCTURED ROLE PROFILE:
{json.dumps(role_context, ensure_ascii=False)}

CANDIDATE PROFILE:
{candidate_document}

CANDIDATE METADATA:
Current title: {candidate_metadata.get('current_title', '')}
Years of experience: {candidate_metadata.get('years_of_experience', 0)}
Industry: {candidate_metadata.get('industry', '')}

DETERMINISTIC MATCH EVIDENCE:
{json.dumps(score_breakdown, ensure_ascii=False)}

Return ONLY valid JSON:

{{
    "fit_score": 0,
    "technical_fit": 0,
    "role_alignment": 0,
    "experience_alignment": 0,
    "evidence": [],
    "gaps": [],
    "recommendation": "Strong Fit"
}}

Rules:

- Evaluate this specific role without assuming any profession or domain.
- Prioritize required skill coverage, then preferred skills, title or career
  relevance, and required experience.
- Every item in evidence must cite a concrete fact present in the candidate
  profile or deterministic match evidence.
- Every gap must map to a stated job requirement that lacks profile evidence.
- Do not infer skills from unrelated titles, employers, or industries.
- Do not use GitHub activity, recruiter response rate, interview completion,
  open-to-work status, or similar engagement signals as evidence of job fit.
- All scores must be integers from 0 to 100.
- Return JSON only, without markdown or explanation.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if match:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return _normalize_evaluation(result, candidate)

    except Exception as error:
        print(f"Evaluation Agent Error: {error}")

    return _fallback_evaluation(candidate)
