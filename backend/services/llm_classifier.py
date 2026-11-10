"""
LLM-based skill classifier.
Builds the prompt from the skill taxonomy, calls the LLM API, and validates its JSON answer.
The answer has weighted skills (1-100), optional subtasks for multi-part projects,
and a goal profile (domain, constraints, cost/latency priorities...).
Includes a harmful-request check, an out-of-scope check, and a local keyword
fallback when no API key is set.

Ranking models from the weights is NOT done here (see services/ranking.py).
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from services.data_loader import load_json, MAPPING, SKILLS
from services import scenarios

# ---- 1. settings, skill list and keywords --------------------------------------

# .env (API key) is read from backend/data
BASE_DIR = Path(__file__).parent.parent / "data"
CONFIG = load_json("config.json")

def build_taxonomy():
    """Group the skills by capability from MAPPING, in the shape:
    {"capabilities": [{"name": capability, "skills": [skill, ...]}, ...]}"""
    groups = {}
    for row in MAPPING:
        skills = groups.setdefault(row["Capability"], [])
        if row["Skill"] not in skills:
            skills.append(row["Skill"])
    return {"capabilities": [{"name": cap, "skills": skills}
                             for cap, skills in groups.items()]}

TAXONOMY = build_taxonomy()
KEYWORDS = load_json("keywords.json")

# skill -> capability, e.g. "Code Generation" -> "Coding"
SKILL_TO_CAP = {}
for _cap in TAXONOMY["capabilities"]:
    for _skill in _cap["skills"]:
        SKILL_TO_CAP[_skill] = _cap["name"]

# skill -> one-line definition, from data/skill_definitions.csv (used when
# "use_skill_definitions" is true in config.json)
# data/skill_definitions.json (from the skill-definitions branch) fills skills the CSV lacks.
SKILL_DEFINITIONS = {k: v["definition"] for k, v in load_json("skill_definitions.json").items()}
SKILL_DEFINITIONS.update({row["Skill / Certificate"].strip(): row["Definition"].strip()
                          for row in SKILLS if row.get("Definition")})

# Limits for the classifier answer (can be set in config.json)
MIN_WEIGHT = CONFIG.get("min_weight", 30)      # drop skills weighted under this
MAX_SKILLS = CONFIG.get("max_skills", 8)       # at most this many skills per answer
MAX_SUBTASKS = CONFIG.get("max_subtasks", 4)   # at most this many project parts
# The LLM classifier is held to fewer, stronger skills (people typically pick 3-5)
LLM_MIN_WEIGHT = CONFIG.get("llm_min_weight", 50)
LLM_MAX_SKILLS = CONFIG.get("llm_max_skills", 5)

# ---- 2. skill name lookups (checks the LLM's skill names are real) ------------

def norm(s):
    """Normalize a string for case/whitespace-insensitive matching."""
    return re.sub(r"\s+", " ", str(s).strip().lower())

def build_skill_lookup():
    """
    Build skill lookup tables for validating the model's output against the
    real taxonomy.
    """
    skill_lookup = {}
    for cap in TAXONOMY["capabilities"]:
        for skill in cap["skills"]:
            skill_lookup[norm(skill)] = (skill, cap["name"])
    return skill_lookup

SKILL_LOOKUP = build_skill_lookup()

def canonical_skill(s):
    """Looks up normalized skill in SKILL_LOOKUP. Returns (skill, capability) or None."""
    return SKILL_LOOKUP.get(norm(s))

def normalize_weights(raw):
    """Accept skills as names, {skill, weight} dicts or a {skill: weight} map.

    Returns [{"skill", "weight"}] with canonical names and weights 1-100, strongest
    first. Skills under MIN_WEIGHT are dropped (the top skill is always kept).
    Plain names with no weight get 100, 80, 60... by order.
    """
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, dict):
        raw = [{"skill": k, "weight": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        return []
    found = {}
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            name, w = item.get("skill") or item.get("name"), item.get("weight")
        else:
            name, w = item, None
        got = canonical_skill(name)
        if not got:
            continue
        try:
            w = int(round(float(w)))
        except (TypeError, ValueError):
            w = max(MIN_WEIGHT, 100 - 20 * i)
        found[got[0]] = max(found.get(got[0], 0), max(1, min(100, w)))
    ordered = sorted(found.items(), key=lambda kv: -kv[1])
    keep = [(s, w) for s, w in ordered if w >= MIN_WEIGHT] or ordered[:1]
    return [{"skill": s, "weight": w} for s, w in keep[:MAX_SKILLS]]

# ---- 3. goal profile -----------------------------------------------------------

LEVELS = ("low", "medium", "high")
TEXT_FIELDS = ("task", "domain", "example_inputs", "constraints", "output_format")
LEVEL_FIELDS = ("risk_tolerance", "accuracy", "creativity", "safety", "latency", "cost")

def normalize_profile(raw):
    """Goal profile: free-text fields plus low/medium/high levels ('unspecified' if absent)."""
    raw = raw if isinstance(raw, dict) else {}
    out = {f: str(raw.get(f) or "").strip()[:500] for f in TEXT_FIELDS}
    for f in LEVEL_FIELDS:
        v = norm(raw.get(f) or "")
        out[f] = v if v in LEVELS else "unspecified"
    return out

# ---- 4. API key (from data/.env or the environment) --------------------------
# Could replace with:
# load_dotenv(BASE_DIR / ".env") -> from dotenv import load_dotenv

def _load_env_file():
    """
    Load KEY=value lines from data/.env into environment variables.
    Does nothing if there is no .env file.
    """
    path = os.path.join(BASE_DIR, ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass

_load_env_file()

# API key: one shared Ollama key on the server is used for every user
# _request / get_api_key are set so each user could add their own key. (not used rn)
# TODO:
#   - if we get GPU access on the lab server: no key is needed.
#   - if not: keep the shared key, and maybe let users add their own key
#     once the shared quota runs out

_request = threading.local()

def get_api_key():
    """
    Return the key for the current request: the user's own key if one was set
    on _request, otherwise the shared server key.
    """
    user_key = getattr(_request, "key", "")
    if user_key:
        return user_key
    return get_server_key()

def get_server_key():
    """
    Return the shared server key. "api_key" in config.json.
    """
    key = CONFIG["backend"].get("api_key", "")
    if isinstance(key, str) and key.startswith("env:"):
        return os.environ.get(key[4:], "") or ""
    return key or os.environ.get("OLLAMA_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

def _headers():
    """
    HTTP headers for the LLM request: JSON content type, plus
    "Authorization: Bearer <key>" when a key is available.
    """
    h = {"Content-Type": "application/json"}
    key = get_api_key()
    if key:
        h["Authorization"] = "Bearer " + key
    return h

# ---- 5. the LLM: instructions, call, and checking its reply -------------------

def build_system_prompt(query="", exclude=None):
    """
    Build the instructions sent to the LLM before the conversation:
    its role, the skills grouped by capability (from TAXONOMY), the rules for
    follow-up questions, weights, subtasks, the goal profile, out-of-scope tasks,
    and the exact JSON reply format.
    """
    lines = [
        "You are a strict categorization assistant. Your ONLY job is:",
        "1) Talk briefly with the user to understand what they are evaluating, and",
        "2) Classify it into one or MORE skills from the fixed taxonomy below.",
        "",
        "TAXONOMY (capability -> skills):",
    ]
    for cap in TAXONOMY["capabilities"]:
        lines.append("- " + cap["name"] + ": " + ", ".join(cap["skills"]))
    if CONFIG.get("use_skill_definitions", False):
        lines += ["", "SKILL DEFINITIONS (what each skill means):"]
        lines += ["- %s: %s" % (s, SKILL_DEFINITIONS[s]) for s in SKILL_TO_CAP if s in SKILL_DEFINITIONS]
    lines += [
        "",
        "STRICT RULES:",
        "- Do NOT answer unrelated questions or discuss anything outside this task.",
        "- If the user asks which model to use (e.g. \"what model?\"), do NOT explain or refuse:",
        "  respond with the answer JSON for the task discussed so far; the app attaches the",
        "  model recommendations to every answer.",
        "  If the user goes off-topic, politely steer back to the task.",
        "- You MUST respond with ONLY a single JSON object. No other text, no markdown.",
        "",
        "FOLLOW-UP QUESTIONS (be as specific as possible about the skills needed):",
        "- PREFER ANSWERING. If the user's first message is detailed (about 40+ words) or already",
        "  makes the skills clear, answer immediately with no question. Ask only when the message",
        "  is short or vague AND the answer would change which skills you pick.",
        "- Each question must separate two concrete candidates, e.g. \"Does the model need to",
        "  WRITE new code or EXPLAIN/DEBUG existing code?\" Never ask vague questions like",
        "  \"can you tell me more?\".",
        "- For a large or multi-part project, first ask the user to list its main parts, then",
        "  ask about the skills each part needs. Ask one question at a time, at most %d total."
        % CONFIG.get("max_questions", 6),
        "- Stop asking as soon as every part of the task maps clearly to skills.",
        "",
        "WEIGHTS:",
        "- Give every skill a need weight from 1 to 100 (100 = central to the task, 30 = minor).",
        "- Include only skills with weight >= %d, at most %d skills. People typically need only 3-5;" % (LLM_MIN_WEIGHT, LLM_MAX_SKILLS),
        "  leave out skills that are merely related.",
        "",
        "LARGE PROJECTS:",
        "- If the task has 2 or more distinct parts that need different kinds of skills",
        "  (e.g. coding AND emotional tone AND planning), also return 'subtasks': a list of at",
        "  most %d parts, each with a short name and its own weighted skills." % MAX_SUBTASKS,
        "- Omit 'subtasks' for a single focused task.",
        "",
        "GOAL PROFILE:",
        "- Also fill 'profile' from what the user said. Leave a field empty (or omit the level)",
        "  when they did not say; never guess. Text fields: task, domain, example_inputs,",
        "  constraints, output_format. Levels (low/medium/high): risk_tolerance, accuracy,",
        "  creativity, safety, latency, cost. Set cost or latency to high only if the user",
        "  stresses budget or speed.",
        "",
        "OUT OF SCOPE:",
        "- The taxonomy covers text-based skills only. If the task needs something it does not",
        "  cover (image/video/audio generation, speech, robotics, ...), do not force a match.",
        '  Respond {"type":"unsupported","text":"what is not covered and what you CAN help with"}.',
        "  If only part of the task is out of scope, answer for the covered part and mention the",
        "  rest in 'summary'.",
        "",
        "RESPONSE FORMAT:",
        'If you need more context: {"type":"question","text":"your question"}',
        'If you have enough to classify:',
        '{"type":"answer","skills":[{"skill":"<skill1>","weight":90},{"skill":"<skill2>","weight":40}],'
        '"subtasks":[{"name":"<part>","skills":[{"skill":"<skill>","weight":80}]}],'
        '"profile":{"task":"","domain":"","example_inputs":"","constraints":"","output_format":"",'
        '"risk_tolerance":"","accuracy":"","creativity":"","safety":"","latency":"","cost":""},'
        '"confidence":0.9,"summary":"one short sentence"}',
        "- Skill names must be EXACT names from the taxonomy, most important first.",
        "- 'confidence' is a number between 0.0 and 1.0.",
    ]
    lines += scenarios.examples_block(query, exclude=exclude)
    return "\n".join(lines)

def is_blocked_backend(name):
    """True if a model belongs to a family barred from running the chatbot itself
    ("blocked_backend_families" in config.json). Recommendations are not affected."""
    n = norm(name)
    return any(re.search(r"(?<![a-z0-9])" + re.escape(norm(f)), n)
               for f in CONFIG.get("blocked_backend_families", []))

def call_llm(messages):
    """
    Send the conversation to the LLM and return its reply text.
    First asks for a JSON-only reply; if the server rejects that option, retries once without it.
    """
    lm = CONFIG["backend"]
    if is_blocked_backend(lm.get("model", "")):
        raise ValueError("Model '%s' is in blocked_backend_families (config.json). It can still be "
                         "recommended, but not used to run the chatbot." % lm.get("model"))
    url = lm["base_url"].rstrip("/") + "/chat/completions"
    if url.startswith("https") and not get_api_key():
        raise ValueError(
            "No API key set. Click \"API key\" in the header to add your own "
            "Ollama key, or set OLLAMA_API_KEY in .env on the server.")

    payload = {
        "model": lm.get("model", "local-model"),
        "messages": messages,
        "temperature": lm.get("temperature", 0.0),
        "stream": False,
    }
    timeout = lm.get("timeout_seconds", 300)

    body = dict(payload)
    body["response_format"] = {"type": "json_object"}

    # prepare request to send llm
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=_headers(),
    )
    # send request
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError:
        # Some engines reject response_format; retry without it.
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=_headers(),
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))

    return resp["choices"][0]["message"]["content"]

def extract_json(text):
    """Find the JSON object in an LLM reply (from the first { to the last })
    and turn it into a dict. Code-block markers and text around it are ignored.
    Returns (dict, None) on success, or (None, error message)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no JSON object found"
    try:
        return json.loads(t[start:end + 1]), None
    except Exception as e:
        return None, str(e)

def validate_and_normalize_answer(ans):
    """
    Check and clean an "answer" dict (from the LLM or the local classifier), in place.

    Adds:
      ans["weights"]    [{"skill", "weight"}], strongest first
      ans["skills"]     plain list of skill names (same order), kept for older code
      ans["skill"]      top skill; ans["capability"] its capability
      ans["subtasks"]   [{"name", "skills": weights}] for multi-part projects, else []
      ans["profile"]    normalized goal profile
    Returns (True, None) if valid, or (False, message to send back to the LLM).
    """
    sw = normalize_weights(ans.get("skills") or ans.get("skill"))
    if not sw:
        msg = ("Your 'skills' are not in the taxonomy. Use EXACT skill names from "
               "the list and respond again with a valid 'answer' JSON object.")
        return False, msg

    ans["weights"] = sw
    ans["skills"] = [x["skill"] for x in sw]
    ans["skill"] = ans["skills"][0]
    ans["capability"] = SKILL_TO_CAP[ans["skill"]]

    subtasks = []
    raw_subtasks = ans.get("subtasks")
    for st in (raw_subtasks if isinstance(raw_subtasks, list) else [])[:MAX_SUBTASKS]:
        ssw = normalize_weights(st.get("skills")) if isinstance(st, dict) else []
        if ssw:
            name = str(st.get("name") or SKILL_TO_CAP[ssw[0]["skill"]]).strip()
            subtasks.append({"name": name, "skills": ssw})
    ans["subtasks"] = subtasks
    ans["profile"] = normalize_profile(ans.get("profile"))

    try:
        conf = float(ans.get("confidence", 0.5))
        ans["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        ans["confidence"] = 0.5
    ans["summary"] = str(ans.get("summary", "")).strip()
    return True, None

# ---- 6. harm and out-of-scope filters ------------------------------------------

DECLINE_TEXT = (
    "I can't help with that. I won't recommend models for causing harm or getting "
    "around their safeguards. If you're evaluating safety, describe it as a "
    "robustness task, e.g. \"which model best resists harmful or jailbreak "
    "requests\" or \"most harmless content generation\".")

UNSUPPORTED_TEXT = (
    "We don't have benchmark data for %s, so I can't recommend a model for it reliably. "
    "The skills I cover are text-based (reasoning, summarizing, writing code, planning, "
    "data interpretation, bias and safety, and so on). If part of your task is text-based, "
    "describe that part.")

def is_harm_seeking(text):
    """True for clearly harm-seeking or safeguard-evading requests.

    Safety-evaluation phrasing (resist, detect, robust...) is exempt so
    legitimate red-team / moderation questions still get recommendations.
    """
    if re.search(KEYWORDS.get("_safety_eval_words", "$^"), text):
        return False
    return any(re.search(p, text) for p in KEYWORDS.get("_harm_patterns", []))

def find_unsupported(text):
    """Phrases in the text that ask for skills outside the taxonomy (images, speech...).
    Uses "_unsupported_patterns" in keywords.json; returns [] if that key is missing."""
    found = []
    for p in KEYWORDS.get("_unsupported_patterns", []):
        m = re.search(p, text)
        if m and m.group(0) not in found:
            found.append(m.group(0))
    return found

NO_QUESTION_NUDGE = "Your message has enough detail. Do not ask a question; respond now with the answer JSON."
BLEND_MIN_WORDS = 15      # shorter inputs are too thin for neighbour matching to be reliable
UNSUPPORTED_NEIGHBOUR = 0.7

def blend_with_neighbours(sw, text, exclude=None):
    """Down-weight skills the most similar human-ranked scenarios never chose.

    Only for longer inputs. A skill with weight < 80 and no support among the 8 nearest
    scenarios is scaled by 0.7; the list is then re-filtered by min_weight. Skills are
    never added, so the model stays in charge of what the task needs.
    """
    if len(text.split()) < BLEND_MIN_WORDS or not sw:
        return sw
    chosen = set()
    for sc in scenarios.similar_scenarios(text, 8, exclude=exclude):
        chosen.update(x["skill"] for x in sc["skills"])
    if not chosen:
        return sw
    out = []
    for x in sw:
        w = x["weight"]
        if x["skill"] not in chosen and w < 80:
            w = int(round(w * UNSUPPORTED_NEIGHBOUR))
        out.append({"skill": x["skill"], "weight": w})
    return normalize_weights(out)

# ---- 7. main entry point -------------------------------------------------------

def classify(messages, exclude=None, examples=True):
    """
    Classify the conversation into weighted skills.

    The LLM does all categorizing; there is no keyword fallback. Returns (result, content, error):
      result   {"type": "question", "text"} or a validated "answer" dict
      content  the result as a JSON string
      error    None, or an error message (result is then None)
    Model ranking is not done here; see services/ranking.py.
    """
    if CONFIG["backend"]["base_url"].startswith("https") and not get_api_key():
        return None, "", "No API key is set (OLLAMA_API_KEY), so the AI model cannot categorize the task."
    latest = [m.get("content", "") for m in messages if m.get("role") == "user"][-1:]
    if latest and is_harm_seeking(norm(latest[0])):
        q = {"type": "question", "text": DECLINE_TEXT}
        return q, json.dumps(q), None

    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    # exclude / examples are for tools/eval_scenarios.py: keep the scenario under test out of
    # its own examples and blend, or drop the examples entirely
    sys_prompt = build_system_prompt(user_text if examples else "", exclude)
    full = [{"role": "system", "content": sys_prompt}] + list(messages)

    # The store saves assistant replies as plain text, so any earlier assistant message
    # counts as a question already asked.
    asked = sum(1 for m in messages if m.get("role") == "assistant")
    raw = call_llm(full)
    parsed, err = extract_json(raw)

    # Answer first: on the first message, give a best-guess recommendation instead of a question
    # (short startup-style prompts got a question ~80% of the time, yet the forced answer had the
    # right main skill 93% of the time). The question is kept as an optional refinement tip.
    # Only very short/vague messages (under answer_first_min_words) still get a question first.
    refine = None
    if (parsed and parsed.get("type") == "question" and asked == 0
            and len(user_text.split()) >= CONFIG.get("answer_first_min_words", 4)):
        refine = str(parsed.get("text", "")).strip() or None
        full.append({"role": "assistant", "content": raw})
        full.append({"role": "user", "content": NO_QUESTION_NUDGE})
        raw = call_llm(full)
        parsed, err = extract_json(raw)

    # one retry if the reply wasn't JSON
    if parsed is None:
        full.append({"role": "assistant", "content": raw})
        full.append({"role": "user", "content":
                     "You must respond with ONLY a JSON object. Follow the format exactly."})
        raw = call_llm(full)
        parsed, err = extract_json(raw)

    if parsed is None:
        return None, raw, "The model did not return valid JSON: " + (err or "")

    ptype = parsed.get("type")

    # questions and out-of-scope notices both go back to the user as a question
    if ptype in ("question", "unsupported"):
        parsed = {"type": "question", "text": str(parsed.get("text", "")).strip()}
        return parsed, json.dumps(parsed), None

    if ptype == "answer":
        ok, msg = validate_and_normalize_answer(parsed)
        if ok:
            # blend with human-ranked neighbours, then hold the LLM to fewer, stronger skills
            blended = blend_with_neighbours(parsed["weights"], user_text, exclude) if examples \
                else parsed["weights"]
            strong = [x for x in blended
                      if x["weight"] >= LLM_MIN_WEIGHT] or parsed["weights"][:1]
            parsed["skills"] = strong[:LLM_MAX_SKILLS]
            ok, msg = validate_and_normalize_answer(parsed)
            missing = find_unsupported(norm(user_text))
            if missing:
                parsed["unsupported_note"] = (
                    "Note: we have no benchmark data for %s, so this recommendation covers only "
                    "the parts of your task we can measure." % ", ".join(missing))
            if refine:
                parsed["refine_question"] = refine
        if not ok:
            # one retry if the skill names weren't in the taxonomy
            full.append({"role": "assistant", "content": raw})
            full.append({"role": "user", "content": msg})
            raw2 = call_llm(full)
            parsed2, _ = extract_json(raw2)
            if parsed2 and parsed2.get("type") == "answer" and validate_and_normalize_answer(parsed2)[0]:
                return parsed2, json.dumps(parsed2), None
            return None, raw, msg or "invalid classification"
        return parsed, json.dumps(parsed), None

    return None, raw, "Unknown response type: " + str(ptype)