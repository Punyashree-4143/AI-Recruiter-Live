import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from src.jd_parser import parse_job_description


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    result = []
    seen = set()

    for item in value:
        if not isinstance(item, str):
            continue

        cleaned = " ".join(item.split())
        normalized = cleaned.casefold()

        if cleaned and normalized not in seen:
            seen.add(normalized)
            result.append(cleaned)

    return result


def _number(value: Any) -> float:
    try:
        return max(float(value), 0)
    except (TypeError, ValueError):
        return 0


def _build_search_query(profile: dict[str, Any], fallback: str) -> str:
    terms = [
        profile.get("role", ""),
        profile.get("domain", ""),
        *profile.get("equivalent_titles", []),
        *profile.get("related_titles", []),
        *profile.get("required_skills", []),
        *profile.get("preferred_skills", []),
    ]

    query = " ".join(
        str(term).strip()
        for term in terms
        if str(term).strip()
    )
    return query or fallback


def _normalize_profile(
    result: dict[str, Any],
    job_description: str,
) -> dict[str, Any]:
    fallback = parse_job_description(job_description)

    role = str(result.get("role") or fallback["role"]).strip()
    required_skills = (
        _string_list(result.get("required_skills"))
        or fallback["required_skills"]
    )
    preferred_skills = (
        _string_list(result.get("preferred_skills"))
        or fallback["preferred_skills"]
    )
    equivalent_titles = _string_list(
        result.get("equivalent_titles")
        or result.get("target_titles")
    )
    related_titles = _string_list(
        result.get("related_titles")
        or result.get("related_roles")
    )
    experience_required = (
        _number(result.get("experience_required"))
        or _number(result.get("minimum_experience"))
        or fallback["experience_required"]
    )

    profile = {
        "role": role,
        "domain": str(result.get("domain", "")).strip(),
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "equivalent_titles": equivalent_titles,
        "related_titles": related_titles,
        "experience_required": experience_required,
    }
    profile["target_titles"] = [
        title
        for title in [role, *equivalent_titles]
        if title
    ]
    profile["search_query"] = (
        str(result.get("search_query", "")).strip()
        or _build_search_query(profile, job_description)
    )

    return profile


def understand_job_description(
    job_description: str,
) -> dict[str, Any]:
    prompt = f"""
You are a senior recruiter analyzing a job description for candidate search.

JOB DESCRIPTION:

{job_description}

Return ONLY valid JSON using this exact structure:

{{
    "role": "",
    "domain": "",
    "required_skills": [],
    "preferred_skills": [],
    "equivalent_titles": [],
    "related_titles": [],
    "experience_required": 0,
    "search_query": ""
}}

Rules:

- Use only the job description as the source of role requirements.
- Work for any profession or function; do not assume a technical or AI role.
- Put explicit must-have qualifications in required_skills.
- Put explicit nice-to-have qualifications in preferred_skills.
- Infer only directly implied skills needed to perform stated responsibilities.
- equivalent_titles must be genuinely interchangeable with the target role.
- related_titles may represent adjacent backgrounds, but must not be treated as equivalent.
- experience_required must be the minimum required years as a number.
- Build a natural-language semantic search query from the role, titles, domain,
  required skills, and preferred skills.
- Do not use Boolean operators.
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
                return _normalize_profile(result, job_description)

    except Exception as error:
        print(f"JD Agent Error: {error}")

    return _normalize_profile({}, job_description)
