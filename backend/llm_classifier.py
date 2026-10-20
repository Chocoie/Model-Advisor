"""
llm_classifier.py: llm's team skill classifier, copied from the chatbot
branch (modelavisorchatbot-main/server.py).

Changes from llm's version:
  - TAXONOMY is built from Faria's dataset-skill mapping (ranking_data.MAPPING)
    instead of taxonomy.json, so the skill list always matches the ranking data
  - load_json comes from ranking_data
  - skill lookup wrapped in build_skill_lookup(); CAP_LOOKUP, canonical_capability
    and SKILL_TO_CAP removed (the capability comes from SKILL_LOOKUP)
  - call_llm's retry reuses payload instead of copying it again
  - extract_json simplified (find/rfind already skip code-block markers)
  - the get_recommendations calls were removed (ranking_data.py ranks models)
  - the web-server parts (Handler, run, index.html) were left out (FastAPI does that)
"""

import json
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from process_ranking_data import load_json, MAPPING

# ---- 1. settings, skill list and keywords (ours) ------------------------------

# .env (API key) is read from backend/data
BASE_DIR = Path(__file__).parent / "data"
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
    """ Looks up normalized skill in SKILL_LOOKUP"""
    return SKILL_LOOKUP.get(norm(s))

# ---- 3. API key (from data/.env or the environment) --------------------------
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

# ---- 4. the LLM: instructions, call, and checking its reply -------------------

def build_system_prompt():
    """
    Build the instructions sent to the LLM before the conversation:
    its role, the 34 skills grouped by capability (from TAXONOMY), the rules,
    and the exact JSON reply format ("question" or "answer").
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
    lines += [
        "",
        "STRICT RULES:",
        "- Do NOT answer unrelated questions or discuss anything outside this task.",
        "- If the user asks which model to use (e.g. \"what model?\"), do NOT explain or refuse:",
        "  respond with the answer JSON for the task discussed so far; the app attaches the",
        "  model recommendations to every answer.",
        "  If the user goes off-topic, politely steer back to the task.",
        "- Ask at most a few short clarifying questions, and only when you lack enough",
        "  context to pick a skill. Otherwise classify immediately.",
        "- You MUST respond with ONLY a single JSON object. No other text, no markdown.",
        "",
        "RESPONSE FORMAT:",
        'If you need more context, respond EXACTLY like: {"type":"question","text":"your question"}',
        'If you have enough to classify, respond EXACTLY like:',
        '{"type":"answer","skills":["<skill1>","<skill2>"],"confidence":0.9,"summary":"one short sentence"}',
        "- 'skills' is a list of 1 to 5 EXACT skill names from the taxonomy, most relevant first.",
        "- Use more than one skill only when the task genuinely spans several skills.",
        "- 'confidence' is a number between 0.0 and 1.0.",
    ]
    return "\n".join(lines)

def call_llm(messages):
    """
    Send the conversation to the LLM and return its reply text.
    First asks for a JSON-only reply; if the server rejects that option, retries once without it.
    """
    lm = CONFIG["backend"]
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
    and turn it into a dict. Text or code-block markers around it are ignored.
    Returns (dict, None) on success, or (None, error message)."""
    t = (text or "").strip()
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
    ans = {"type": "answer", "skills": ["summarization", ...], 
    "confidence": 0.9, "summary": "..."} from extract_json()
    """

    # extract skills into raw_skills, if str -> convert to list
    raw_skills = ans.get("skills")
    if isinstance(raw_skills, str):
        raw_skills = [raw_skills]
    if not raw_skills:
        raw_skills = [ans.get("skill", "")]

    # if normalized skill in SKILL_LOOKUP and not in set seen, 
    # add to seen and add to skills
    skills = []
    seen = set()
    for name in raw_skills:
        got = canonical_skill(name)
        if got:
            canon_skill = got[0]
            if canon_skill not in seen: #no repeats
                seen.add(canon_skill)
                skills.append(canon_skill)

    # message sent to llm
    if not skills:
        msg = ("Your 'skills' are not in the taxonomy. Use EXACT skill names from "
               "the list and respond again with a valid 'answer' JSON object.")
        return False, msg

    ans["skills"] = skills
    ans["skill"] = skills[0]
    ans["capability"] = canonical_skill(skills[0])[1]
    try:
        conf = float(ans.get("confidence", 0.5))
        ans["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        ans["confidence"] = 0.5
    ans["summary"] = str(ans.get("summary", "")).strip()
    return True, None # Valid answer, no error msg


# ---- 5. local keyword classifier (no API) and harm filter ---------------------

def use_local_classifier():
    """ 
    If config has no "classifier" key, default is "local". If classifier is
    local, return true.
    """
    return CONFIG.get("classifier", "local") == "local"

DECLINE_TEXT = (
    "I can't help with that. I won't recommend models for causing harm or getting "
    "around their safeguards. If you're evaluating safety, describe it as a "
    "robustness task, e.g. \"which model best resists harmful or jailbreak "
    "requests\" or \"most harmless content generation\".")

def is_harm_seeking(text):
    """True for clearly harm-seeking or safeguard-evading requests.

    Safety-evaluation phrasing (resist, detect, robust...) is exempt so
    legitimate red-team / moderation questions still get recommendations.
    """
    if re.search(KEYWORDS.get("_safety_eval_words", "$^"), text):
        return False
    return any(re.search(p, text) for p in KEYWORDS.get("_harm_patterns", []))

def classify_local(messages):
    """Match the latest user message to skills with keyword scoring. No API calls.

    Each distinct keyword that appears (at a word start, so "summar" matches
    "summarize"/"summary") scores 1; a skill's own name scores 3.
    Keeps skills scoring at least a third of the top score, up to 5.
    """
    text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            text = norm(m.get("content", ""))
            break
    if is_harm_seeking(text):
        q = {"type": "question", "text": DECLINE_TEXT}
        return q, json.dumps(q), None
    scored = []
    matched = {}
    for cap in TAXONOMY["capabilities"]:
        for skill in cap["skills"]:
            hits = [kw for kw in KEYWORDS.get(skill, [])
                    if re.search(r"(?<![a-z0-9])" + re.escape(kw), text)]
            score = len(hits) + (3 if norm(skill) in text else 0)
            if score:
                scored.append((score, skill))
                matched[skill] = hits
    if not scored:
        q = {"type": "question", "text":
             "I couldn't match that to a skill. Describe what the model needs to do "
             "(e.g. write code, summarize, plan steps, detect emotion, extract data), "
             "or click a skill in the sidebar."}
        return q, json.dumps(q), None
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[0][0]
    skills = [s for sc, s in scored if sc >= max(1, top / 3)][:5]
    ans = {
        "type": "answer",
        "skills": skills,
        "skill": skills[0],
        "capability": canonical_skill(skills[0])[1],
        "confidence": round(min(0.95, 0.35 + 0.15 * top), 2),
        "summary": "Matched on: " + ", ".join(
            sorted({k for s in skills for k in matched[s]}) or skills),
    }
    clean = {k: v for k, v in ans.items() if k != "recommendations"}
    return ans, json.dumps(clean), None

# ---- 6. main entry point -------------------------------------------------------

def classify(messages):
    """Run the model, enforce the JSON contract, and validate the answer."""
    if use_local_classifier():
        return classify_local(messages)
    sys_prompt = build_system_prompt()
    full = [{"role": "system", "content": sys_prompt}] + list(messages)

    raw = call_llm(full)
    parsed, err = extract_json(raw)

    if parsed is None:
        full.append({"role": "assistant", "content": raw})
        full.append({"role": "user", "content":
                     "You must respond with ONLY a JSON object. Follow the format exactly."})
        raw = call_llm(full)
        parsed, err = extract_json(raw)

    if parsed is None:
        return None, raw, "The model did not return valid JSON: " + (err or "")

    ptype = parsed.get("type")
    if ptype == "question":
        parsed["text"] = str(parsed.get("text", "")).strip()
        return parsed, json.dumps(parsed), None

    if ptype == "answer":
        ok, msg = validate_and_normalize_answer(parsed)
        if not ok:
            full.append({"role": "assistant", "content": raw})
            full.append({"role": "user", "content": msg})
            raw2 = call_llm(full)
            parsed2, _ = extract_json(raw2)
            if parsed2 and parsed2.get("type") == "answer" and validate_and_normalize_answer(parsed2)[0]:
                clean2 = {k: v for k, v in parsed2.items() if k != "recommendations"}
                return parsed2, json.dumps(clean2), None
            return None, raw, msg or "invalid classification"
        clean = {k: v for k, v in parsed.items() if k != "recommendations"}
        return parsed, json.dumps(clean), None

    return None, raw, "Unknown response type: " + str(ptype)