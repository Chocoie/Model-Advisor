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
        return {"type": "ready", "skills": result["skills"], "weights": result["weights"]}
    
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

def plain_list(weights, results):
    """Format the weighted skills and top models as a plain text list."""
    label = lambda w: "essential" if w >= 80 else "important" if w >= 55 else "minor"
    lines = ["Skills detected:"]
    for x in weights:
        lines.append(f"- {x['skill']} ({label(x['weight'])})")
    lines.append("Top Models:")
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['model']} (match {r['fit'] / 10:.1f}/10)")
    return "\n".join(lines)

def generate_title(history):
    """Ask the LLM for a short conversation title. Returns None if it fails."""
    title_prompt = (
        "Give a short title (3 to 6 words) naming the type of AI model this user is looking for, "
        "based on the conversation. Example: 'Architectural floor plan generator'. "
        'Respond with ONLY this JSON: {"title": "..."}'
    )
    try:
        full = [{"role": "system", "content": title_prompt}] + history
        raw = call_llm(full)
        parsed, err = extract_json(raw)
        if not parsed:
            print("generate_title: bad reply:", err, raw[:200])
            return None
        title = str(parsed.get("title", "")).strip().strip('"').strip("'")
        return title[:60] or None
    except Exception as e:
        print("generate_title failed:", repr(e))
        return None
