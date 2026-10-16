"""
chat_agent.py: connects the conversation to the LLM team's code.

Contract main.py relies on:
  next_action(history) -> {"type": "question", "question": text}
                       or {"type": "ready", "skills": list of skill names}
  explain_results(skills, results) -> text

next_action uses llm_classifier.classify (keyword or LLM mode, set in data/config.json).
explain_results adds an LLM-written explanation when the LLM is on,
and falls back to a plain list otherwise.
"""

import json
from llm_classifier import classify, call_llm, extract_json, use_local_classifier

def next_action(history):
    """
    Decide whether to ask the user another question or start ranking.
    Calls llm_classifier.classify on the conversation history.
    """
    result, _, error = classify(history)
    if error:
        return{"type" : "question", 
                       "question" : "Sorry, I did not quite get that. Could you describe what you need the model to do?"}
    if result["type"] == "question":
        return {"type": "question", "question": result["text"]}
    if result["type"] == "answer":
        return {"type": "ready", "skills": result["skills"]}


def explain_results(skills, results):
    """
    Build the reply shown with the rankings: the plain list, plus a short
    LLM-written explanation when the LLM is on. Falls back to the plain list
    if the LLM is off, unreachable, or returns something unusable.
    """
    plain = plain_list(skills, results)
    if use_local_classifier():
        return plain                           # LLM is off

    prompt = (
        "Explain in 2-3 sentences why these models are recommended for these skills. "
        "Use only the data given. Respond as JSON: {\"explanation\": \"...\"}\n"
        f"Skills: {skills}\nResults: {json.dumps(results)}"
    )
    try:
        raw = call_llm([{"role": "user", "content": prompt}])
        parsed, _ = extract_json(raw)
        if parsed:
            return plain + "\n\n" + parsed["explanation"]
        return plain
    except Exception:
        return plain                           # API down or no key
    

def plain_list(skills, results):
    """Format the detected skills and top-ranked models as a plain text list."""
    lines = ["Skills detected:"]
    for s in skills:
        lines.append(f"- {s}")
    lines.append("Top Models:")
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['model']} (avg rank {r['avg_rank']})")
    return "\n".join(lines)