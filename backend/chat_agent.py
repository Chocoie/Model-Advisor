"""
PLACEHOLDER CHAT AGENT (keyword-based)

This file will be replaced by the LLM team's real agent.
Keep the function names and return shapes the same so main.py doesn't change:

    next_step(history) -> {"type": "question", "question": str}
                       or {"type": "ready", "skills": [skill names from VALID_SKILLS]}

    explain(skills, results) -> str

history format: [{"role": "user" | "assistant", "content": str}, ...]
"""

import json
from ranking_data import DATA_DIR, VALID_SKILLS

# keywords.json comes from the chatbot branch (LLM team's prototype)
with open(DATA_DIR / "keywords.json", encoding="utf-8") as f:
    KEYWORDS = json.load(f)


# TODO (LLM team): could be kept as a fallback if the LLM is down
def match_skills(text, max_skills=2):
    """ skills whose keywords appear most often in the text, best first"""

    text = text.lower()
    hits = {}
    for skill, words in KEYWORDS.items():
        count = sum(1 for w in words if w in text)
        if count and skill in VALID_SKILLS:
            hits[skill] = count
    best = sorted(hits, key=hits.get, reverse=True)
    return best[:max_skills]


# TODO (LLM team): replace with an LLM call that reads the history and
# either asks a follow-up question or returns the skills
def next_step(history):
    """ placeholder agent """

    user_text = " ".join(m["content"] for m in history if m["role"] == "user")
    skills = match_skills(user_text)
    if not skills:
        return {"type": "question",
                "question": "Can you tell me more about the task? What should the model do?"}
    return {"type": "ready", "skills": skills}


# TODO (LLM team): replace with an LLM-written explanation of the recommendation
def explain(skills, results):
    """ placeholder"""

    lines = [f"Skills detected: {', '.join(skills)}", "Top models:"]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['model']} (avg rank {r['avg_rank']})")
    return "\n".join(lines)