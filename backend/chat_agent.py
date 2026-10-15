"""
  placeholder agent, the contract the LLM team must keep:
  next_action(history) -> {"type": "question", "question": text}
                       or {"type": "ready", "skills": list of skill names}
  explain_results(skills, results) -> text
"""

import json
from ranking_data import DATA_DIR, VALID_SKILLS

with open(DATA_DIR / "keywords.json", encoding="utf-8") as f:
    KEYWORDS = json.load(f)

def match_skills(text, max_skills = 2): 
    """ Keyword look-up fallback with max skills default set to 2 """

    text = text.lower()
    hits = {}
    for skill, words in KEYWORDS.items():
        count = sum(1 for w in words if w in text)
        if count > 0 and skill in VALID_SKILLS:
            hits[skill] = count
    best = sorted(hits, key=hits.get, reverse=True)
    return best[:max_skills]


# TODO replace with actual chat agent
def next_action(history):
    """ 
        Based on conversation, evaluates if follow-up question is necessary or 
        if enough information was gathered.
    """
    user_text = " ".join(m["content"] for m in history if m["role"] == "user")
    
    # TODO 
    # Chat Agent here
    # skills = ...
    # if Chat Agent down:
    #     skills = match skills
    
    skills = match_skills(user_text)

    if not skills:
        return{"type" : "question", 
               "question" : "Could you give me more details on what are your goals? Do you have any constraints or limitations?"}
    return {"type": "ready", "skills": skills}



# TODO replace with chat agent explaination
def explain_results(skills, results):
    """ Returns the detected skills and the ranked models as text."""
    lines = [f"Skills detected: "]
    for s in skills:
        lines.append(f"- {s}")
    lines.append("Top Models: ")
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['model']} (avg rank {r['avg_rank']})")
    return "\n".join(lines)