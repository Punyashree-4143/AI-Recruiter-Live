import json
import random
from collections import defaultdict
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "candidates.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "candidates_demo.jsonl"

random.seed(42)

groups = defaultdict(list)

print(f"Reading dataset from:\n{INPUT_FILE}\n")

# -----------------------------
# Read candidates
# -----------------------------

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

# -----------------------------
# Create balanced dataset
# -----------------------------

demo_candidates = []

for title, candidates in groups.items():

    random.shuffle(candidates)

    demo_candidates.extend(
        candidates[:50]
    )

# -----------------------------
# Fill remaining candidates
# -----------------------------

if len(demo_candidates) < 1000:

    selected_ids = {
        candidate["candidate_id"]
        for candidate in demo_candidates
    }

    remaining = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            candidate = json.loads(line)

            if candidate["candidate_id"] not in selected_ids:
                remaining.append(candidate)

    random.shuffle(remaining)

    demo_candidates.extend(
        remaining[:1000 - len(demo_candidates)]
    )

# -----------------------------
# Trim if more than 1000
# -----------------------------

demo_candidates = demo_candidates[:1000]

# -----------------------------
# Save demo dataset
# -----------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for candidate in demo_candidates:
        f.write(json.dumps(candidate))
        f.write("\n")

print("\n===================================")
print(f"Created demo dataset successfully!")
print(f"Total Candidates : {len(demo_candidates)}")
print(f"Saved to         : {OUTPUT_FILE}")
print("===================================")