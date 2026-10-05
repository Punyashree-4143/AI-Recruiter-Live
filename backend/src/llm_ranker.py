from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def generate_llm_reasoning(
    job_description,
    candidate
):
    prompt = f"""
You are an expert technical recruiter.

Job Description:
{job_description}

Candidate Information:

Current Title:
{candidate['metadata'].get('current_title', '')}

Years of Experience:
{candidate['metadata'].get('years_of_experience', 0)}

Skill Match:
{candidate['score_breakdown'].get('skill_match', 0)}

Write exactly 2 professional sentences explaining why this candidate is suitable for the role.

Do not mention numerical scores.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=120
    )

    return response.choices[0].message.content.strip()