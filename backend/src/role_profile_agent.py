from groq import Groq
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def build_role_profile(
    job_description
):

    prompt = f"""
You are a senior recruiter.

Analyze this job description.

JOB DESCRIPTION:

{job_description}

Return ONLY valid JSON.

Format:

{{
    "role": "",
    "domain": "",
    "seniority": "",
    "experience_required": 0,

    "core_skills": [],

    "responsibilities": [],

    "equivalent_titles": [],

    "related_roles": [],

    "search_query": ""
}}

Rules:

- Do NOT assume any profession.
- Understand the role naturally.
- Infer missing but obvious skills.
- Infer responsibilities.
- Infer equivalent job titles.
- Infer related career paths.
- Create a semantic search query.

Return ONLY JSON.
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

        match = re.search(
            r"\{.*\}",
            content,
            re.DOTALL
        )

        if match:

            result = json.loads(
                match.group()
            )

            return result

    except Exception as e:

        print(
            f"Role Profile Agent Error: {e}"
        )

    return {
        "role": "",
        "domain": "",
        "seniority": "",
        "experience_required": 0,
        "core_skills": [],
        "responsibilities": [],
        "equivalent_titles": [],
        "related_roles": [],
        "search_query": job_description
    }