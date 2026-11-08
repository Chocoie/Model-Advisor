"""
Human-ranked scenarios: finds the scenarios closest to a task (tf-idf cosine) so the
classifier prompt can show how people weighted skills for similar tasks.
Data (data/scenarios.json, data/skill_prior.json) comes from tools/build_scenario_data.py.
"""

import math
import re
from collections import Counter
from services.data_loader import load_json


def _load_optional(name, default):
    try:
        return load_json(name)
    except FileNotFoundError:
        return default


SCENARIOS = _load_optional("scenarios.json", {"scenarios": []})["scenarios"]
PRIOR = _load_optional("skill_prior.json", {})

_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to "
            "was were will with an ai agent required should can may also often their they them "
            "which who into than then these those such more most".split())


def _tokens(text):
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP]


_DOC_TOKENS = [Counter(_tokens(sc["text"])) for sc in SCENARIOS]
_DF = Counter(w for d in _DOC_TOKENS for w in d)
_IDF = {w: math.log((1 + len(SCENARIOS)) / (1 + n)) + 1 for w, n in _DF.items()}


def _vec(counts):
    v = {w: c * _IDF.get(w, 0) for w, c in counts.items() if w in _IDF}
    mag = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / mag for w, x in v.items()}


_DOC_VECS = [_vec(d) for d in _DOC_TOKENS]


def similar_scenarios(text, k=4, exclude=None):
    """The k scenarios closest to `text`. `exclude` skips a scenario with that exact text (for eval)."""
    q = _vec(Counter(_tokens(text)))
    if not q:
        return []
    scored = []
    for i, dv in enumerate(_DOC_VECS):
        if exclude is not None and SCENARIOS[i]["text"] == exclude:
            continue
        sim = sum(x * dv.get(w, 0) for w, x in q.items())
        if sim > 0 and SCENARIOS[i]["skills"]:
            scored.append((sim, i))
    scored.sort(reverse=True)
    return [SCENARIOS[i] for _, i in scored[:k]]


def examples_block(text, k=4, exclude=None):
    """Prompt lines showing how people weighted skills for similar tasks."""
    hits = similar_scenarios(text, k, exclude)
    if not hits:
        return []
    lines = ["", "HOW PEOPLE PICKED SKILLS FOR SIMILAR TASKS (human consensus; weight 100 = everyone",
             "chose it first). Use as calibration, not as the answer; the user's task comes first:"]
    for sc in hits:
        lines.append("- Task: " + sc["text"][:300])
        lines.append("  Skills: " + ", ".join("%s %d" % (x["skill"], x["weight"]) for x in sc["skills"]))
    if PRIOR.get("avg_skills"):
        lines.append("People typically pick about %.0f skills per task, not every skill that is loosely "
                     "related." % PRIOR["avg_skills"])
    return lines
