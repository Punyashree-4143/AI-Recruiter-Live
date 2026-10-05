import json
import random
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "candidates_demo.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "candidates_demo_500.jsonl"

random.seed(42)

groups = defaultdict(list)

print(f"Reading dataset from:\n{INPUT_FILE}\n")

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:
        line = line.strip()

        if not line:
            continue

        candidate = json.loads(line)

        title = (
            candidate.get("profile", {})
            .get("current_title", "Unknown")
            .strip()
        )

        groups[title].append(candidate)

print(f"Found {len(groups)} unique job titles.")

# Number to select from each role
role_limits = {
    "Backend Engineer": 35,
    "Data Engineer": 35,
    "QA Engineer": 35,
    "Frontend Engineer": 35,
    "Software Engineer": 35,
    "DevOps Engineer": 35,
    ".NET Developer": 35,
    "Full Stack Developer": 35,
    "Java Developer": 35,

    "Recommendation Systems Engineer": 26,
    "Cloud Engineer": 24,

    "Operations Manager": 14,
    "Customer Support": 14,
    "Marketing Manager": 14,
    "Business Analyst": 14,
    "Project Manager": 14,
    "Accountant": 13,
    "Civil Engineer": 13,
    "Mechanical Engineer": 13,
    "HR Manager": 13,
    "Graphic Designer": 13,
}

filtered_candidates = []

for title, limit in role_limits.items():

    candidates = groups.get(title, [])

    random.shuffle(candidates)

    selected = candidates[:limit]

    filtered_candidates.extend(selected)

    print(
        f"{title:<35} "
        f"Available: {len(candidates):<4} "
        f"Selected: {len(selected)}"
    )

# Remove duplicate candidate IDs
seen_ids = set()
unique_candidates = []

for candidate in filtered_candidates:

    candidate_id = candidate["candidate_id"]

    if candidate_id not in seen_ids:
        seen_ids.add(candidate_id)
        unique_candidates.append(candidate)

filtered_candidates = unique_candidates

# Shuffle final dataset
random.shuffle(filtered_candidates)

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for candidate in filtered_candidates:
        f.write(json.dumps(candidate, ensure_ascii=False))
        f.write("\n")

print("\n===================================")
print("Filtering completed!")
print("===================================")
print(f"Total Candidates : {len(filtered_candidates)}")
print(f"Saved to         : {OUTPUT_FILE}")
print("===================================")