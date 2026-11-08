"""
Follow-up questions about an existing recommendation ("why is X ranked below Y?",
"what if cost matters more?", "give me a prompt"). Answered from the benchmark data only;
nothing here asks an LLM for its opinion.

The "context" is the last answer saved for the conversation:
{"weights", "subtasks", "profile", "summary", "unsupported_note"}.
"""

import math
import re
from services.llm_classifier import (CONFIG, KEYWORDS, TAXONOMY, norm, validate_and_normalize_answer)
from services.ranking import SCORES, skill_score

CHANGE_RX = r"more important|prioriti[sz]|what if|instead|matters? more|care more|focus on|rather"
GENERIC_TOKENS = {"instruct", "preview", "it", "v1"}
WEAK_THRESHOLD = CONFIG.get("weak_threshold", 0.7)
PRIORITY_SHIFT = {"unspecified": 0.0, "low": 0.0, "medium": 0.1, "high": 0.25}

# ---- ranking extras on top of services/ranking.py (cost/speed boost, relative scores, split plan)

KNOWN_SIZES = {"microsoft_Phi-3.5-mini-instruct": 3.8, "mistralai_Mistral-Nemo-Instruct-2407": 12.0}

def display_name(model):
    """'meta-llama_Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct'."""
    return model.split("_", 1)[-1]

def model_size_b(model):
    """Parameter count in billions from the model name, or None."""
    if model in KNOWN_SIZES:
        return KNOWN_SIZES[model]
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-z])", display_name(model))
    return float(m.group(1)) if m else None

_SIZES = [s for s in map(model_size_b, SCORES) if s]

def efficiency(model):
    """0-1, higher = smaller (cheaper/faster). Estimated from size; no measured cost or latency."""
    size = model_size_b(model)
    if not size or max(_SIZES) == min(_SIZES):
        return 0.5
    return 1 - math.log(size / min(_SIZES)) / math.log(max(_SIZES) / min(_SIZES))

def efficiency_weight(profile):
    p = profile or {}
    return min(0.5, PRIORITY_SHIFT.get(p.get("cost"), 0) + PRIORITY_SHIFT.get(p.get("latency"), 0))

def _best(skill):
    return max(skill_score(m, skill) for m in SCORES)

def rank_skills(weights, top_n=3, lam=0.0):
    """Like ranking.rank_skills, plus "score" (fit scaled up for small models by lam),
    "size_b" and "relative" (each skill as % of the best model). Sorted by score."""
    best = {w["skill"]: _best(w["skill"]) for w in weights}
    # skills no model has a score for can't separate models: leave them out (see no_data_skills)
    measured = [w for w in weights if best[w["skill"]]]
    total = sum(w["weight"] for w in measured)
    out = []
    for m in SCORES:
        raw = {w["skill"]: skill_score(m, w["skill"]) for w in weights}
        fit = sum(w["weight"] * raw[w["skill"]] / best[w["skill"]] for w in measured) / total if total else 0.0
        # scaling (not adding) means a weak small model never wins just because it is cheap
        score = fit * (1 - lam + lam * efficiency(m))
        out.append({"model": m, "fit": round(100 * fit, 1), "score": round(100 * score, 1),
                    "size_b": model_size_b(m), "scores": {s: round(v, 3) for s, v in raw.items()},
                    "relative": {w["skill"]: round(100 * raw[w["skill"]] / best[w["skill"]], 1)
                                 for w in measured}})
    out.sort(key=lambda r: -r["score"])
    return out[:top_n]

def build_plan(subtasks, weights, lam=0.0):
    """Best single model vs a per-part split. A part gets its own model only when the single
    model reaches under WEAK_THRESHOLD of that part's best. None for fewer than two parts."""
    if len(subtasks) < 2:
        return None
    single = rank_skills(weights, 1, lam)[0]["model"]
    parts, split = [], False
    for st in subtasks:
        ranked = rank_skills(st["skills"], len(SCORES), lam)
        best = ranked[0]
        single_score = next(r["score"] for r in ranked if r["model"] == single)
        weak = best["model"] != single and single_score < WEAK_THRESHOLD * best["score"]
        split = split or weak
        parts.append({"name": st["name"], "skills": st["skills"],
                      "model": best["model"] if weak else single,
                      "fit": best["score"] if weak else single_score,
                      "single_model_fit": single_score, "weak": weak})
    return {"single_model": single, "split": split, "subtasks": parts}

# ---- building and presenting a recommendation ------------------------------------

def need_label(w):
    """Plain-words version of a 1-100 need weight."""
    return "essential" if w >= 80 else "important" if w >= 55 else "minor"

def recommend(ans):
    """Attach ranking, top 3, split plan and explanation to a validated answer. Returns ans."""
    lam = efficiency_weight(ans.get("profile"))
    ans["efficiency_weight"] = lam
    ans["ranking"] = rank_skills(ans["weights"], len(SCORES), lam)
    ans["recommendations"] = ans["ranking"][:3]
    ans["plan"] = build_plan(ans.get("subtasks", []), ans["weights"], lam)
    ans["explanation"] = build_explanation(ans)
    return ans

def build_explanation(ans):
    sw, top, plan = ans["weights"], ans["recommendations"][0], ans.get("plan")
    need = ", ".join("%s (%s)" % (x["skill"], need_label(x["weight"])) for x in sw)
    parts = []
    if ans.get("change_note"):
        parts.append(ans["change_note"])
    parts.append("Based on your needs, your use case calls for " + need + ".")
    parts.append("For a single model, we recommend %s, which reaches %.0f%% of the best "
                 "available score across these weighted skills." % (display_name(top["model"]), top["fit"]))
    if ans.get("efficiency_weight"):
        parts.append("Cost and speed matter to you, so smaller models get a boost "
                     "(estimated from model size, not measured).")
    weak = [s for s, p in top["relative"].items() if p < 100 * WEAK_THRESHOLD]
    if weak:
        parts.append("Its weaker spot%s: %s." % ("s" if len(weak) > 1 else "", ", ".join(weak)))
    if plan and plan["split"]:
        assign = "; ".join("%s for %s" % (display_name(p["model"]), p["name"]) for p in plan["subtasks"])
        parts.append("Because the parts of this project need different strengths, we "
                     "recommend splitting it: " + assign + ".")
    elif plan:
        parts.append("The parts of this project are close enough that one model covers them all.")
    no_data = [x["skill"] for x in sw if not _best(x["skill"])]
    if no_data:
        parts.append("No model has benchmark results for %s yet, so %s not counted in the match."
                     % (", ".join(no_data), "it is" if len(no_data) == 1 else "they are"))
    if ans.get("unsupported_note"):
        parts.append(ans["unsupported_note"])
    parts.append("Models are scored from published benchmark results for these skills, "
                 "not by asking an AI which one is better.")
    return " ".join(parts)

def context_from(ans):
    """The part of an answer worth saving so later messages can be follow-ups."""
    return {k: ans.get(k) for k in ("weights", "subtasks", "profile", "summary", "unsupported_note")}

def _answer_from(ctx):
    if not isinstance(ctx, dict):
        return None
    ans = {"type": "answer", "skills": ctx.get("weights"), "subtasks": ctx.get("subtasks"),
           "profile": ctx.get("profile"), "summary": ctx.get("summary", "")}
    if ctx.get("unsupported_note"):
        ans["unsupported_note"] = str(ctx["unsupported_note"])[:500]
    return ans if validate_and_normalize_answer(ans)[0] else None

# ---- finding skills and models in a phrase ---------------------------------------

def find_skill(t):
    """The taxonomy skill best matching a phrase (name or keywords), or None."""
    best, best_score = None, 0
    for cap in TAXONOMY["capabilities"]:
        for skill in cap["skills"]:
            hits = [kw for kw in KEYWORDS.get(skill, [])
                    if re.search(r"(?<![a-z0-9])" + re.escape(kw), t)]
            score = len(hits) + (3 if norm(skill) in t else 0)
            if score > best_score:
                best, best_score = skill, score
    return best

def find_models(t):
    """Models whose name tokens appear in the phrase; several means ambiguous."""
    words = set(re.findall(r"[a-z0-9.]+", t))
    scored = []
    for m in SCORES:
        toks = [x for x in re.split(r"[-\s_]+", norm(display_name(m)))
                if x not in GENERIC_TOKENS and len(x) > 1]
        scored.append((sum(1 for x in toks if x in words), m))
    top = max((s for s, _ in scored), default=0)
    return [m for s, m in scored if s == top] if top else []

def _pick_model(t):
    """(model, None) for one clear match, (None, question) if ambiguous, (None, None) if none."""
    found = find_models(t)
    if len(found) == 1:
        return found[0], None
    if 1 < len(found) <= 8:
        return None, "Which model do you mean: " + ", ".join(display_name(m) for m in found) + "?"
    return None, None

# ---- the answers -----------------------------------------------------------------

def _top_model(ans):
    return rank_skills(ans["weights"], 1, efficiency_weight(ans["profile"]))[0]["model"]

def _why(ans, model):
    ranking = recommend(dict(ans))["ranking"]
    names = [r["model"] for r in ranking]
    rank = names.index(model) + 1
    top, mine = ranking[0], ranking[rank - 1]
    name = display_name(model)
    if rank == 1:
        return "%s is ranked #1 for your needs (fit %.0f%%)." % (name, mine["fit"])
    gaps = sorted(((mine["relative"][s] - top["relative"][s], s) for s in mine["relative"]))
    lines = ["%s is #%d (fit %.0f%%) behind %s (fit %.0f%%)." % (
        name, rank, mine["fit"], display_name(top["model"]), top["fit"])]
    behind = [(d, s) for d, s in gaps if d < 0][:3]
    if behind:
        lines.append("Biggest gaps vs the leader: " + "; ".join(
            "%s (%.0f%% of best vs %.0f%%)" % (s, mine["relative"][s], top["relative"][s])
            for _, s in behind) + ".")
    ahead = [s for d, s in reversed(gaps) if d > 0][:2]
    if ahead:
        lines.append("It does better than the leader on: " + ", ".join(ahead) + ".")
    if efficiency_weight(ans["profile"]):
        lines.append("Your cost/speed priority is also counted (size-based estimate).")
    return " ".join(lines)

def _weak_spots(ans, model):
    mine = next(r for r in rank_skills(ans["weights"], len(SCORES)) if r["model"] == model)
    low = sorted(mine["relative"].items(), key=lambda kv: kv[1])[:3]
    return ("The benchmark has scores per skill, not individual test answers, so I can't show "
            "specific failed examples. Where %s scores lowest relative to the best model: %s."
            % (display_name(model), "; ".join("%s (%.0f%% of best)" % kv for kv in low)))

def _prompt_template(ans):
    p = ans["profile"]
    skills = ", ".join(x["skill"] for x in ans["weights"])
    task = p["task"] or ans.get("summary") or "the task described below"
    return (
        "Here is a starting prompt for %s (built from your goal profile; ModelAdvisor has not "
        "tested prompt variants yet):\n\n"
        "You are an assistant for: %s.\n"
        "Domain: %s\n"
        "Focus on these abilities: %s.\n"
        "Constraints: %s\n"
        "Output format: %s\n"
        "Work step by step, then give the final answer in the format above.\n"
        "Example input/output: %s"
        % (display_name(_top_model(ans)), task, p["domain"] or "(add your domain)", skills,
           p["constraints"] or "(add any limits)", p["output_format"] or "(state the format you want)",
           p["example_inputs"] or "(add two examples)"))

def followup(text, ctx):
    """Handle a question about the current recommendation.

    Returns {"reply": str, "answer": ans_or_None}: "answer" is a new recommendation when the
    user changed their priorities. Returns None if the text is a new task, not a follow-up.
    """
    t = norm(text)
    ans = _answer_from(ctx)
    if not ans:
        return None

    if re.search(CHANGE_RX, t):
        profile, changed = dict(ans["profile"]), []
        if re.search(r"\b(cost|cheap|cheaper|budget|price|afford\w*)\b", t):
            profile["cost"] = "high"
            changed.append("cost")
        if re.search(r"\b(fast|faster|speed|latency|quick\w*|real[- ]time)\b", t):
            profile["latency"] = "high"
            changed.append("speed")
        weights = ans["weights"]
        if not changed:
            skill = find_skill(t)
            if skill:
                weights = [{"skill": skill, "weight": 100}] + [
                    {"skill": x["skill"], "weight": round(x["weight"] * 0.7)}
                    for x in ans["weights"] if x["skill"] != skill]
                changed.append(skill)
        if changed:
            before = _top_model(ans)
            ans["profile"], ans["skills"] = profile, weights
            validate_and_normalize_answer(ans)
            after = _top_model(ans)
            ans["change_note"] = "With %s prioritized, the top model %s." % (
                " and ".join(changed),
                "stays " + display_name(after) if after == before
                else "changes from %s to %s" % (display_name(before), display_name(after)))
            recommend(ans)
            return {"reply": ans["explanation"], "answer": ans}

    if re.search(r"\bwhy\b", t):
        model, ask = _pick_model(t)
        if model or ask:
            return {"reply": _why(ans, model) if model else ask, "answer": None}
    if re.search(r"fail|mistake|wrong|weak|bad at|struggl|worst", t):
        model, ask = _pick_model(t)
        if ask:
            return {"reply": ask, "answer": None}
        return {"reply": _weak_spots(ans, model or _top_model(ans)), "answer": None}
    if re.search(r"\b(which|what)\b.*\b(best|better|strongest)\b", t):
        skill = find_skill(t)
        if skill:
            recs = rank_skills([{"skill": skill, "weight": 100}], 3)
            return {"reply": "Best at %s: %s." % (skill, "; ".join(
                "%s (score %s)" % (display_name(r["model"]), r["scores"][skill]) for r in recs)),
                "answer": None}
    if re.search(r"\bprompt\b", t):
        return {"reply": _prompt_template(ans), "answer": None}
    return None
