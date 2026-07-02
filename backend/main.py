from src.ingestion import load_and_prepare_data
from src.embeddings import create_vector_store
from src.hybrid_search import hybrid_search
from src.recruiter_ranker import rank_candidates
from src.submission_generator import generate_submission
from src.jd_parser import parse_job_description
from src.jd_intelligence_agent import understand_job_description


def main():

    print("\nLoading candidate data...\n")

    documents, metadata = load_and_prepare_data(
        "../data/sample_candidates.json"
    )

    print(
        f"Successfully loaded "
        f"{len(documents)} candidates"
    )

    print("\nCreating vector database...\n")

    create_vector_store(
        documents,
        metadata
    )

    # ==================================
    # JOB DESCRIPTION INPUT
    # ==================================

    job_description = """
    We are hiring an AI Engineer.

    Requirements:

    - Python
    - NLP
    - LLM Fine-Tuning
    - AWS

    Preferred:

    - MLOps
    - LangChain
    - RAG

    3+ years experience.
    """

    print("\nParsing Job Description...\n")

    parsed_jd = parse_job_description(
        job_description
    )

    jd_analysis = understand_job_description(
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

    print("PARSED JD:")
    print(jd_analysis)

    # Convert parsed JD into search query

    query = jd_analysis["search_query"]

    print("\nGenerated Search Query:")
    print(query)

    # ==================================
    # HYBRID SEARCH
    # ==================================

    print("\nRunning Hybrid Search...\n")

    hybrid_results = hybrid_search(
        query=query,
        documents=documents,
        metadata=metadata,
        top_k=20
    )

    print(
        f"Retrieved "
        f"{len(hybrid_results)} candidates"
    )

    # ==================================
    # RECRUITER RANKING
    # ==================================

    print("\nRunning Recruiter Ranking...\n")

    ranked_candidates = rank_candidates(
        hybrid_results=hybrid_results,
        metadata=metadata,
        documents=documents,
        query=query,
        role_profile=jd_analysis
    )

    print("\nFINAL RECRUITER RANKING\n")

    for rank, candidate in enumerate(
        ranked_candidates[:5],
        start=1
    ):

        candidate_id = candidate["candidate_id"]

        score = candidate["score"]

        metadata_info = candidate["metadata"]

        breakdown = candidate[
            "score_breakdown"
        ]

        title = metadata_info.get(
            "current_title",
            "Unknown"
        )

        experience = metadata_info.get(
            "years_of_experience",
            0
        )

        print(f"\n{rank}. {candidate_id}")

        print(f"Title: {title}")

        print(
            f"Experience: "
            f"{experience} years"
        )

        print(
            f"Final Score: {score}"
        )

        print(
            f"Skill Match: "
            f"{breakdown['skill_match']}"
        )

        print(
            f"Title Match: "
            f"{breakdown['title_match']}"
        )

        print(
            f"Experience Score: "
            f"{breakdown['experience_score']}"
        )

        print(
            f"GitHub Score: "
            f"{breakdown['github_score']}"
        )

        print(
            f"Response Score: "
            f"{breakdown['response_score']}"
        )

        print(
            f"Interview Score: "
            f"{breakdown['interview_score']}"
        )

        print(
            f"Open To Work Score: "
            f"{breakdown['open_to_work_score']}"
        )

        print("-" * 80)

    # ==================================
    # CSV OUTPUT
    # ==================================

    print("\nGenerating submission file...\n")

    generate_submission(
        ranked_candidates,
        "../outputs/ranked_candidates.csv"
    )

    print(
        "\nSubmission file generated successfully!"
    )

    print(
        "\nRecruiter Ranking Completed Successfully!"
    )


if __name__ == "__main__":
    main()
