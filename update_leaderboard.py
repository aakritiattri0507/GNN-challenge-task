import os
import pandas as pd

# Folder where submissions are stored
SUBMISSIONS_DIR = "submissions"

# Path to leaderboard
LEADERBOARD_FILE = "leaderboard.md"

# Placeholder scoring function (replace with real scoring script call if needed)
def get_score(submission_file):
    # This should call scoring_script.py or compute F1
    # For now, we'll just return a dummy value
    return 0.0

# Build leaderboard
rows = []
for file in os.listdir(SUBMISSIONS_DIR):
    if file.endswith(".csv"):
        score = get_score(os.path.join(SUBMISSIONS_DIR, file))
        rows.append((file, score))

# Sort by score descending
rows.sort(key=lambda x: x[1], reverse=True)

# Write markdown table
with open(LEADERBOARD_FILE, "w") as f:
    f.write("| Submission | Macro F1 |\n")
    f.write("|------------|----------|\n")
    for sub, score in rows:
        f.write(f"| {sub} | {score:.4f} |\n")

print(f"Leaderboard updated: {LEADERBOARD_FILE}")

