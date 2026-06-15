# Builds sample_candidates.jsonl - the 58-candidate set app.py loads by
# default. Picks the 8 "ideal" candidates, a few honeypots that look good on
# the surface, and a mix of relevant/irrelevant titles, from the full pool.
#
#   python notebooks/02_build_sandbox_sample.py /path/to/candidates.jsonl

import json
import os

IDEAL = {"CAND_0005538","CAND_0006567","CAND_0030468","CAND_0037980",
         "CAND_0061257","CAND_0068351","CAND_0080766","CAND_0093193"}

# A couple of honeypots that look great on the surface (per honeypot indicator
# B/C: timeline doesn't reconcile with claimed years_of_experience).
# We'll find some dynamically below by re-running the detector.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rank import detect_honeypot

honeypots_with_good_titles = []
ml_titled_good = []
off_target = []
general_tech = []

GOOD_TITLES = {"ML Engineer","AI Research Engineer","Data Scientist",
               "Senior Software Engineer (ML)","Computer Vision Engineer",
               "Junior ML Engineer","AI Specialist","Recommendation Systems Engineer",
               "Machine Learning Engineer","Applied ML Engineer","Search Engineer",
               "AI Engineer","Senior Data Scientist","NLP Engineer","Senior NLP Engineer",
               "Senior Machine Learning Engineer","Staff Machine Learning Engineer",
               "Senior AI Engineer","Senior Applied Scientist","Lead AI Engineer"}
OFF_TARGET = {"Business Analyst","HR Manager","Mechanical Engineer","Accountant",
               "Project Manager","Customer Support","Operations Manager",
               "Content Writer","Sales Executive","Civil Engineer",
               "Graphic Designer","Marketing Manager"}
GENERAL_TECH = {"Software Engineer","Full Stack Developer","Cloud Engineer",
                 "Java Developer",".NET Developer","DevOps Engineer",
                 "Mobile Developer","Frontend Engineer","QA Engineer"}

ideal_records = {}
sample = []

with open(sys.argv[1] if len(sys.argv) > 1 else "candidates.jsonl") as f:
    for line in f:
        c = json.loads(line)
        cid = c["candidate_id"]
        title = c["profile"]["current_title"]

        if cid in IDEAL:
            ideal_records[cid] = c
            continue

        if detect_honeypot(c) and title in GOOD_TITLES and len(honeypots_with_good_titles) < 3:
            honeypots_with_good_titles.append(c)
            continue

        if title in GOOD_TITLES and len(ml_titled_good) < 12:
            ml_titled_good.append(c)
            continue

        if title in OFF_TARGET and len(off_target) < 18:
            off_target.append(c)
            continue

        if title in GENERAL_TECH and len(general_tech) < 17:
            general_tech.append(c)
            continue

# Assemble: all 8 ideals + 3 honeypots + 12 good ML + 18 off-target + 17 general-tech = 58
sample = [ideal_records[cid] for cid in IDEAL] + honeypots_with_good_titles \
         + ml_titled_good + off_target + general_tech

print(f"Sample size: {len(sample)}")
print(f"  Ideal pool: {len(ideal_records)}")
print(f"  Honeypots (good titles): {len(honeypots_with_good_titles)}")
for h in honeypots_with_good_titles:
    p = h["profile"]
    print(f"    {h['candidate_id']}: {p['current_title']} @ {p['current_company']}, yoe={p['years_of_experience']}")
print(f"  ML-titled good: {len(ml_titled_good)}")
print(f"  Off-target: {len(off_target)}")
print(f"  General tech: {len(general_tech)}")

with open(os.path.join(os.path.dirname(__file__), "..", "sample_candidates.jsonl"), "w") as f:
    for c in sample:
        f.write(json.dumps(c) + "\n")
