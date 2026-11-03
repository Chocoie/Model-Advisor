"""Build data/scenarios.json from the scenario-ranking study.

    python tools/build_scenario_data.py <path to scenario_rank folder>

Each scenario was ranked by 2-3 human annotators. Their picks are mapped onto our
taxonomy (data/scenario_skill_map.json) and merged into one consensus weight list.
"""

import csv
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TAXONOMY = json.load(open(os.path.join(HERE, "..", "taxonomy.json"), encoding="utf-8"))
OWN = {s for c in TAXONOMY["capabilities"] for s in c["skills"]}
SKILL_MAP = json.load(open(os.path.join(DATA, "scenario_skill_map.json"), encoding="utf-8"))
KEEP = 50      # a skill needs a consensus weight of at least this
MAX_KEEP = 8


def to_own(skill):
    return skill if skill in OWN else SKILL_MAP.get(skill)


def main(root):
    per = {}       # scenario -> {user: [skills in rank order]}
    group = {}
    for f in sorted(glob.glob(os.path.join(root, "annotations", "user*_annotation.csv"))):
        uid = int(re.search(r"user(\d+)_", f).group(1))
        for r in csv.DictReader(open(f, encoding="utf-8", errors="replace")):
            text = r["scenario"].strip()
            picks = [s.strip() for s in r["ranked_list"].split(",") if s.strip()]
            per.setdefault(text, {})[uid] = picks
            group[text] = uid // 100 if uid < 1000 else 10

    out, demand, lost = [], Counter(), Counter()
    for text, users in per.items():
        score = Counter()
        for picks in users.values():
            seen = set()
            for rank, raw in enumerate(picks):
                own = to_own(raw)
                if not own:
                    lost[raw] += 1
                    continue
                if own in seen:
                    continue
                seen.add(own)
                score[own] += 1 - 0.5 * rank / max(len(picks), 1)
        weights = [{"skill": s, "weight": round(100 * v / len(users))} for s, v in score.items()]
        weights = sorted((w for w in weights if w["weight"] >= KEEP), key=lambda w: -w["weight"])[:MAX_KEEP]
        for w in weights:
            demand[w["skill"]] += 1
        out.append({"text": text, "group": group[text], "annotators": len(users), "skills": weights})

    json.dump({"scenarios": out}, open(os.path.join(DATA, "scenarios.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    prior = {s: round(demand[s] / len(out), 3) for s in sorted(OWN, key=lambda s: -demand[s])}
    avg = sum(len(s["skills"]) for s in out) / len(out)
    json.dump({"scenarios": len(out), "avg_skills": round(avg, 2), "skill_share": prior,
               "unmapped_picks": dict(lost.most_common())},
              open(os.path.join(DATA, "skill_prior.json"), "w", encoding="utf-8"), indent=1)
    print(len(out), "scenarios; avg", round(avg, 2), "skills; unmapped picks:", sum(lost.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "scenario_rank", "scenario_rank"))
