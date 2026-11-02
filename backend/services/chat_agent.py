"""
Chat agent logic.
Decides the next step in a conversation (ask a follow-up question or detect skills)
and turns ranking results into a readable reply.
"""

import json
from services.llm_classifier import classify, call_llm, extract_json, use_local_classifier, classify_local

def next_action(history):
    """
    Decide whether to ask the user another question or start ranking.
    Calls llm_classifier.classify on the conversation history.
    """
    try:
        result, _, error = classify(history)
    except Exception as e:
        print("LLM failed:", repr(e))
        result, _, error = classify_local(history) 
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