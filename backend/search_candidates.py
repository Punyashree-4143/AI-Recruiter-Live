from functools import lru_cache
from pathlib import Path

from src.comparison_agent import compare_candidates
from src.evaluation_agent import evaluate_candidate
from src.explainability_agent import generate_explanation
from src.hiring_decision_agent import hiring_decision
from src.hybrid_search import hybrid_search
from src.ingestion import load_and_prepare_data
from src.jd_intelligence_agent import understand_job_description
from src.jd_parser import parse_job_description
from src.recruiter_ranker import rank_candidates
from src.skill_gap_analyzer import analyze_skill_gap
from src.submission_generator import generate_submission


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "candidates_demo.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "ranked_candidates.csv"


@lru_cache(maxsize=1)
def _load_candidate_data():
    return load_and_prepare_data(str(DATA_PATH))


def run_recruitment_pipeline(job_description):
    documents, metadata = _load_candidate_data()

    # -----------------------------
    # JD Intelligence Agent
    # -----------------------------

    jd_analysis = understand_job_description(
        job_description
    )

    parsed_jd = parse_job_description(
        job_description
    )

    for key in (
        "role",
        "required_skills",
        "preferred_skills",
        "equivalent_titles",
        "related_titles",
        "experience_required",
    ):
        if not jd_analysis.get(key):
            jd_analysis[key] = parsed_jd.get(key)

    query = jd_analysis["search_query"]

    print("\nJD Intelligence Agent Output:\n")
    print(jd_analysis)

    # -----------------------------
    # Retrieval Agent
    # -----------------------------

    hybrid_results = hybrid_search(
        query=query,
        documents=documents,
        metadata=metadata,
        top_k=50
    )

    # -----------------------------
    # Recruiter Ranking Agent
    # -----------------------------

    ranked_candidates = rank_candidates(
        hybrid_results=hybrid_results,
        metadata=metadata,
        documents=documents,
        query=query,
        role_profile=jd_analysis
    )

    # -----------------------------
    # Evaluation Agent
    # -----------------------------

    print(
        "\nRunning Recruiter Evaluation Agent...\n"
    )

    top_candidates = ranked_candidates[:10]

    for candidate in top_candidates:

        # -----------------------------
        # Skill Gap Analysis
        # -----------------------------

        skill_gap = analyze_skill_gap(
            document=candidate["document"],
            required_skills=jd_analysis.get(
                "required_skills",
                []
            ),
            preferred_skills=jd_analysis.get(
                "preferred_skills",
                []
            )
        )

        candidate["skill_gap"] = skill_gap

        candidate["skill_coverage"] = (
            skill_gap["coverage"]
        )

        # -----------------------------
        # Explainability Agent
        # -----------------------------

        candidate["explanation"] = (
            generate_explanation(
                candidate,
                jd_analysis
            )
        )

        # -----------------------------
        # LLM Evaluation
        # -----------------------------

        evaluation = evaluate_candidate(
            job_description,
            candidate,
            jd_analysis
        )

        candidate["evaluation"] = evaluation

        candidate["llm_fit_score"] = (
            evaluation.get(
                "fit_score",
                50
            )
        )

        candidate["recommendation"] = (
            evaluation.get(
                "recommendation",
                "Moderate Fit"
            )
        )

        llm_score = (
            evaluation.get(
                "fit_score",
                50
            ) * 0.5
            +
            evaluation.get(
                "technical_fit",
                50
            ) * 0.2
            +
            evaluation.get(
                "role_alignment",
                50
            ) * 0.2
            +
            evaluation.get(
                "experience_alignment",
                50
            ) * 0.1
        )

        candidate["score"] = round(
            (
                candidate["score"] * 100 * 0.60
            )
            +
            (
                llm_score * 0.40
            ),
            2
        )

    top_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # -----------------------------
    # Skill Gap Summary
    # -----------------------------

    print("\nSkill Gap Summary:\n")

    for candidate in top_candidates:

        print(
            f"{candidate['candidate_id']} | "
            f"Coverage: "
            f"{candidate['skill_gap']['coverage']}%"
        )

        print(
            "Matched Required Skills:",
            candidate["skill_gap"][
                "matched_required"
            ]
        )

        print(
            "Missing Required Skills:",
            candidate["skill_gap"][
                "missing_required"
            ]
        )

        print("-" * 60)

    # -----------------------------
    # Candidate Explainability
    # -----------------------------

    print("\nCandidate Explainability:\n")

    for candidate in top_candidates:

        exp = candidate["explanation"]

        print(
            f"\n{candidate['candidate_id']}"
        )

        print(
            f"Title: {exp['title']}"
        )

        print(
            f"Experience: "
            f"{exp['experience']} years"
        )

        print(
            f"Skill Coverage: "
            f"{exp['coverage']}%"
        )

        print(
            "Matched Skills:",
            exp["matched_skills"]
        )

        print(
            "Missing Skills:",
            exp["missing_skills"]
        )

    # -----------------------------
    # Comparison Agent
    # -----------------------------

    print(
        "\nRunning Candidate Comparison Agent...\n"
    )

    comparison_result = compare_candidates(
        job_description,
        top_candidates
    )

    print(
        "\nComparison Agent Output:\n"
    )

    print(
        comparison_result
    )

    # -----------------------------
    # Hiring Decision Agent
    # -----------------------------

    print(
        "\nRunning Hiring Decision Agent...\n"
    )

    decision = hiring_decision(
        job_description,
        comparison_result,
        top_candidates
    )

    print(
        "\nHiring Decision:\n"
    )

    print(
        decision
    )

    # -----------------------------
    # Save Results
    # -----------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    generate_submission(
        ranked_candidates=top_candidates,
        output_path=str(OUTPUT_PATH),
        job_description=job_description
    )

    print(
        "\nSearch completed successfully."
    )

    return {
        "jd_analysis": jd_analysis,
        "candidates": top_candidates,
        "comparison": comparison_result,
        "decision": decision
    }


def main():
    job_description = """
We are hiring a Senior Backend Engineer.

Requirements:
Java
Spring Boot
Microservices
Kafka
REST APIs
AWS

Preferred:
Docker
Kubernetes

5+ years experience.
"""

    run_recruitment_pipeline(
        job_description
    )


if __name__ == "__main__":
    main()
