"""
Chat agent logic.
Decides the next step in a conversation (answer a follow-up about the current
recommendation, ask a question, or detect skills) and turns ranking results into a reply.

Called by routers/conversations.py: next_action(history), then explain_results(weights, results).
"""

from services.llm_classifier import (classify, call_llm, extract_json, classify_local,
                                     validate_and_normalize_answer)
from services.followup import followup, recommend, context_from, need_label, display_name

import re

# Memory of the last recommendation, for follow-up questions. The conversation history (saved
# per user by the backend) is the source of truth: _CONTEXT is an exact in-memory copy, and after
# a server restart the context is rebuilt from the recommendation reply text itself.
_CONTEXT = {}   # recommendation reply text -> context for follow-up questions
_PENDING = {}   # weights key -> full answer from next_action, picked up by explain_results

SKILLS_HEADER = "Skills detected:"
COST_NOTE = "Cost and speed matter to you"
LABEL_WEIGHT = {"essential": 90, "important": 65, "minor": 40}   # inverse of need_label


def _key(weights):
    return tuple((w["skill"], w["weight"]) for w in weights)


def _context_from_reply(text):
    """Rebuild a follow-up context from a recommendation reply, or None if it isn't one.
    Recovers skills, need levels and the cost/speed priority (exact weights and the goal
    profile's text fields are not in the reply)."""
    if not text.startswith(SKILLS_HEADER):
        return None
    block = text.split("Top Models:", 1)[0]
    weights = [{"skill": s, "weight": LABEL_WEIGHT[label]}
               for s, label in re.findall(r"^- (.+) \((essential|important|minor)\)$", block, re.M)]
    if not weights:
        return None
    profile = {"cost": "high"} if COST_NOTE in text else {}
    return {"weights": weights, "subtasks": [], "profile": profile, "summary": ""}


def last_context(history):
    """Context of the latest recommendation in the conversation (follow-up answers in
    between don't reset it), or None."""
    for m in reversed(history):
        if m.get("role") == "assistant":
            ctx = _CONTEXT.get(m["content"]) or _context_from_reply(m["content"])
            if ctx:
                return ctx
    return None


def next_action(history):
    """
    Decide whether to answer a follow-up, ask the user a question, or start ranking.
    Returns {"type": "question", "question"} (any text reply) or
    {"type": "ready", "skills", "weights"}.
    """
    users = [m for m in history if m.get("role") == "user"]
    ctx = last_context(history)
    if ctx and users:
        out = followup(users[-1]["content"], ctx)
        if out and not out["answer"]:
            return {"type": "question", "question": out["reply"]}
        if out:
            return _ready(out["answer"])

    try:
        result, _, error = classify(history)
    except Exception as e:
        print("LLM failed:", repr(e))
        result, _, error = classify_local(history)
    if error or not result:
        return {"type": "question",
                "question": "Sorry, I did not quite get that. Could you describe what you need the model to do?"}
    if result["type"] == "question":
        return {"type": "question", "question": result["text"]}
    return _ready(result)


def _ready(ans):
    _PENDING[_key(ans["weights"])] = ans
    return {"type": "ready", "skills": ans["skills"], "weights": ans["weights"]}


def explain_results(weights, results):
    """
    Build the reply shown with the rankings: skills, top models (with the user's cost/speed
    priority and split plan applied) and an explanation built from the benchmark data.
    `results` from ranking.rank_skills is not used: the agent re-ranks with the goal profile.
    """
    ans = _PENDING.pop(_key(weights), None)
    if ans is None:
        ans = {"type": "answer", "skills": weights}
        validate_and_normalize_answer(ans)
    recommend(ans)
    reply = plain_list(ans) + "\n\n" + ans["explanation"]
    _CONTEXT[reply] = context_from(ans)
    return reply


def plain_list(ans):
    """Format the weighted skills and top models as a plain text list."""
    lines = [SKILLS_HEADER]
    for x in ans["weights"]:
        lines.append(f"- {x['skill']} ({need_label(x['weight'])})")
    lines.append("Top Models:")
    for i, r in enumerate(ans["recommendations"], start=1):
        lines.append(f"{i}. {display_name(r['model'])} (match {r['score'] / 10:.1f}/10)")
    if ans.get("classifier_note"):
        lines += ["", ans["classifier_note"]]
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
