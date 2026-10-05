from groq import Groq
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def hiring_decision(
    job_description,
    comparison_result,
    candidates
):

    candidate_summary = []

    for candidate in candidates[:5]:

        candidate_summary.append(
            f"""
Candidate ID:
{candidate['candidate_id']}

Title:
{candidate['metadata'].get('current_title', '')}

Experience:
{candidate['metadata'].get('years_of_experience', 0)}

Score:
{candidate['score']}

Recommendation:
{candidate.get('recommendation', '')}
"""
        )

    prompt = f"""
You are a senior hiring manager.

JOB DESCRIPTION:

{job_description}

COMPARISON RESULT:

{comparison_result}

TOP CANDIDATES:

{''.join(candidate_summary)}

Select the best candidate.

IMPORTANT:
- Return ONLY valid JSON.
- "confidence" MUST be a number from 0 to 100.
- Do NOT write words such as "nine", "ninety", or "high" for confidence.
- Do NOT use markdown.
- Do NOT add any text before or after the JSON.

Return exactly this structure:

{{
    "recommended_candidate": "CAND_XXXXXXX",
    "confidence": 90,
    "strengths": [],
    "risks": [],
    "final_decision": ""
}}
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        print("\nGROQ HIRING RESPONSE:")
        print(content)

        # -------------------------------------------------
        # Try normal JSON parsing first
        # -------------------------------------------------

        try:

            result = json.loads(content)

        except json.JSONDecodeError:

            # -------------------------------------------------
            # Extract JSON if model added surrounding text
            # -------------------------------------------------

            match = re.search(
                r"\{.*\}",
                content,
                re.DOTALL
            )

            if not match:
                raise ValueError(
                    "Could not extract JSON from hiring response."
                )

            json_text = match.group()

            # -------------------------------------------------
            # Repair common invalid confidence values
            # -------------------------------------------------

            json_text = re.sub(
                r'"confidence"\s*:\s*0\.\s*nine\b',
                '"confidence": 90',
                json_text,
                flags=re.IGNORECASE
            )

            json_text = re.sub(
                r'"confidence"\s*:\s*0\.\s*ninety\b',
                '"confidence": 90',
                json_text,
                flags=re.IGNORECASE
            )

            json_text = re.sub(
                r'"confidence"\s*:\s*nine\b',
                '"confidence": 90',
                json_text,
                flags=re.IGNORECASE
            )

            result = json.loads(json_text)

        # -------------------------------------------------
        # Validate confidence
        # -------------------------------------------------

        confidence = result.get("confidence", 0)

        if isinstance(confidence, str):

            confidence_match = re.search(
                r"\d+(?:\.\d+)?",
                confidence
            )

            if confidence_match:
                confidence = float(
                    confidence_match.group()
                )
            else:
                confidence = 0

        result["confidence"] = confidence

        return result

    except Exception as e:

        print(
            f"Hiring Decision Agent Error: {e}"
        )

    return {
        "recommended_candidate": "",
        "confidence": 0,
        "strengths": [],
        "risks": [],
        "final_decision": ""
    }