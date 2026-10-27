"""
Zero-dependency local chatbot that classifies a task into the skills it needs
and recommends the best-benchmarked models for them.

It:
  1) Converses briefly with the user to gather context.
  2) Classifies the item into one or more skills from taxonomy.json.
  3) Responds only in strict JSON and validates against the real taxonomy
     so the model cannot drift off-target.

Talks to a local model via an OpenAI-compatible server (Ollama or LM Studio).
Uses only the Python standard library.

Run:
    python server.py            # start server only
    python server.py --open     # start server and open the browser
"""

import json
import os
import re
import sys
import threading
import time
import webbrowser
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Resolve asset directory (works both as a script and when frozen by PyInstaller).
if getattr(sys, "frozen", False):
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(name):
    path = os.path.join(BASE_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_json("config.json")
TAXONOMY = load_json("taxonomy.json")
BENCHMARK = load_json("benchmark.json")
KEYWORDS = load_json("keywords.json")


def norm(s):
    """Normalize a string for case/whitespace-insensitive matching."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


# Lookup tables for validating the model's output against the real taxonomy.
SKILL_LOOKUP = {}
CAP_LOOKUP = {}
for cap in TAXONOMY["capabilities"]:
    CAP_LOOKUP[norm(cap["name"])] = cap["name"]
    for skill in cap["skills"]:
        SKILL_LOOKUP[norm(skill)] = (skill, cap["name"])


def canonical_skill(s):
    return SKILL_LOOKUP.get(norm(s))


def canonical_capability(s):
    return CAP_LOOKUP.get(norm(s))


SKILL_TO_CAP = {}
for _cap in TAXONOMY["capabilities"]:
    for _skill in _cap["skills"]:
        SKILL_TO_CAP[_skill] = _cap["name"]


EXCLUDED_FAMILIES = [norm(f) for f in CONFIG.get("excluded_families", [])]
MIN_WEIGHT = CONFIG.get("min_weight", 30)
MAX_SKILLS = CONFIG.get("max_skills", 8)
MAX_SUBTASKS = CONFIG.get("max_subtasks", 4)
WEAK_THRESHOLD = CONFIG.get("weak_threshold", 0.7)


def is_excluded(name):
    """True if a model name belongs to a blocked (Chinese-developed) family."""
    n = norm(name)
    return any(re.search(r"(?<![a-z0-9])" + re.escape(f), n) for f in EXCLUDED_FAMILIES)


if is_excluded(CONFIG["backend"].get("model", "")):
    raise SystemExit("config.json backend model '%s' is on the excluded list "
                     "(excluded_families). Choose a different model." % CONFIG["backend"]["model"])

# Only models that are allowed ever get ranked or recommended.
MODELS = [m for m in BENCHMARK["models"] if not is_excluded(m["name"])]
BEST = {}
for _m in MODELS:
    for _s, _v in _m["skills"].items():
        BEST[_s] = max(BEST.get(_s, 0), _v or 0)


def build_system_prompt():
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
        "- You MUST respond with ONLY a single JSON object. No other text, no markdown.",
        "",
        "FOLLOW-UP QUESTIONS (be as specific as possible about the skills needed):",
        "- Ask a question only when the answer would change which skills or weights you pick.",
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
        "- Include only skills with weight >= %d, at most %d skills." % (MIN_WEIGHT, MAX_SKILLS),
        "",
        "LARGE PROJECTS:",
        "- If the task has 2 or more distinct parts that need different kinds of skills",
        "  (e.g. coding AND emotional tone AND planning), also return 'subtasks': a list of at",
        "  most %d parts, each with a short name and its own weighted skills." % MAX_SUBTASKS,
        "- Omit 'subtasks' for a single focused task.",
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
        '"confidence":0.9,"summary":"one short sentence"}',
        "- Skill names must be EXACT names from the taxonomy, most important first.",
        "- 'confidence' is a number between 0.0 and 1.0.",
    ]
    return "\n".join(lines)


def _load_env_file():
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


# Per-request key supplied by the browser (X-Api-Key header). Held in memory
# for the duration of the request only; never written to disk or logged.
_request = threading.local()


def get_api_key():
    user_key = getattr(_request, "key", "")
    if user_key:
        return user_key
    return get_server_key()


def get_server_key():
    key = CONFIG["backend"].get("api_key", "")
    if isinstance(key, str) and key.startswith("env:"):
        return os.environ.get(key[4:], "") or ""
    return key or os.environ.get("OLLAMA_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")


def _headers():
    h = {"Content-Type": "application/json"}
    key = get_api_key()
    if key:
        h["Authorization"] = "Bearer " + key
    return h


def call_llm(messages):
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
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Some engines reject response_format; retry without it.
        body = dict(payload)
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=_headers(),
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))

    return resp["choices"][0]["message"]["content"]


def extract_json(text):
    """Pull the first balanced JSON object out of a model response."""
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


def normalize_weights(raw):
    """Accept skills as names, {skill, weight} dicts or a {skill: weight} map.

    Returns [{"skill", "weight"}] with canonical names and weights 1-100, strongest
    first. Skills under min_weight are dropped (the top skill is always kept).
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


def validate_and_normalize_answer(ans):
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

    try:
        conf = float(ans.get("confidence", 0.5))
        ans["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        ans["confidence"] = 0.5
    ans["summary"] = str(ans.get("summary", "")).strip()
    return True, None


def fit(model, sw):
    """Weighted share (0-1) of the best available score the model reaches.

    Each skill's score is taken relative to the best allowed model on that skill, so
    skills with very different score scales count equally per unit of weight.
    """
    total = sum(x["weight"] for x in sw)
    got = 0.0
    for x in sw:
        best = BEST.get(x["skill"], 0)
        if best:
            got += x["weight"] * (model["skills"].get(x["skill"]) or 0) / best
    return got / total if total else 0.0


def get_recommendations(sw, limit=3):
    """Rank allowed models by weighted fit across the needed skills."""
    ranked = sorted(MODELS, key=lambda m: (-fit(m, sw), -m["overall"]))
    recs = []
    for m in ranked[:limit]:
        scores = {x["skill"]: m["skills"].get(x["skill"]) for x in sw}
        recs.append({
            "name": m["name"],
            "fit": round(100 * fit(m, sw), 1),
            "overall": m["overall"],
            "scores": scores,
            "relative": {s: round(100 * (v or 0) / BEST[s], 1) if BEST.get(s) else 0
                         for s, v in scores.items()},
        })
    return recs


def build_plan(subtasks, sw):
    """Compare the best single model with a per-part split for multi-part projects.

    A part gets its own model only when the best single model reaches under
    weak_threshold of that part's best model; otherwise the single model covers it.
    """
    if len(subtasks) < 2:
        return None
    by_name = {m["name"]: m for m in MODELS}
    single = get_recommendations(sw, 1)[0]["name"]
    parts, split = [], False
    for st in subtasks:
        best = get_recommendations(st["skills"], 1)[0]
        single_fit = round(100 * fit(by_name[single], st["skills"]), 1)
        weak = best["name"] != single and single_fit < WEAK_THRESHOLD * best["fit"]
        split = split or weak
        parts.append({
            "name": st["name"], "skills": st["skills"],
            "model": best["name"] if weak else single,
            "fit": best["fit"] if weak else single_fit,
            "single_model_fit": single_fit, "weak": weak,
        })
    return {"single_model": single, "split": split, "subtasks": parts}


def build_explanation(ans):
    sw, top, plan = ans["weights"], ans["recommendations"][0], ans.get("plan")
    need = ", ".join("%s (%d)" % (x["skill"], x["weight"]) for x in sw)
    parts = ["Based on your needs, your use case calls for " + need + "."]
    parts.append("For a single model, we recommend %s, which reaches %.0f%% of the best "
                 "available score across these weighted skills." % (top["name"], top["fit"]))
    weak = [s for s, p in top["relative"].items() if p < 100 * WEAK_THRESHOLD]
    if weak:
        parts.append("Its weaker spot%s: %s." % ("s" if len(weak) > 1 else "", ", ".join(weak)))
    if plan and plan["split"]:
        assign = "; ".join("%s for %s" % (p["model"], p["name"]) for p in plan["subtasks"])
        parts.append("Because the parts of this project need different strengths, we "
                     "recommend splitting it: " + assign + ".")
    elif plan:
        parts.append("The parts of this project are close enough that one model covers them all.")
    if ans.get("unsupported_note"):
        parts.append(ans["unsupported_note"])
    return " ".join(parts)


def finish_answer(ans):
    """Attach recommendations, split plan and explanation to a validated answer."""
    ans["recommendations"] = get_recommendations(ans["weights"])
    ans["plan"] = build_plan(ans.get("subtasks", []), ans["weights"])
    ans["explanation"] = build_explanation(ans)
    extra = ("recommendations", "plan", "explanation")
    clean = {k: v for k, v in ans.items() if k not in extra}
    return ans, json.dumps(clean), None


def use_local_classifier():
    return CONFIG.get("classifier", "local") == "local"


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
    """Phrases in the text that ask for skills outside the taxonomy (images, speech...)."""
    found = []
    for p in KEYWORDS.get("_unsupported_patterns", []):
        m = re.search(p, text)
        if m and m.group(0) not in found:
            found.append(m.group(0))
    return found


def _msg_type(m):
    try:
        return json.loads(m.get("content", "")).get("type")
    except Exception:
        return None


def _examples(skill, n=3):
    return ", ".join(KEYWORDS.get(skill, [])[:n]) or skill


def follow_up_question(sw, hits):
    """A specific question when the first message leaves the skills ambiguous, else None.

    Three cases: a big multi-capability project (ask for its parts), two close skills
    from different capabilities (ask which matters more), or a single weak keyword match
    (ask which sibling skill is meant).
    """
    caps = []
    for x in sw:
        c = SKILL_TO_CAP[x["skill"]]
        if c not in caps:
            caps.append(c)
    if len(caps) >= 3:
        return ("This looks like a larger project touching " + ", ".join(caps[:3]) +
                ". List its main parts (for example: 'backend code', 'customer-facing "
                "writing', 'planning') so I can match the right model to each part.")
    if len(sw) >= 2 and sw[1]["weight"] >= 80 and SKILL_TO_CAP[sw[0]["skill"]] != SKILL_TO_CAP[sw[1]["skill"]]:
        a, b = sw[0]["skill"], sw[1]["skill"]
        return ("Which matters more for this task: %s (e.g. %s) or %s (e.g. %s)? "
                "Or are both equally important?" % (a, _examples(a), b, _examples(b)))
    if len(sw) == 1 and hits <= 1:
        s = sw[0]["skill"]
        cap = SKILL_TO_CAP[s]
        siblings = [k for k in next(c["skills"] for c in TAXONOMY["capabilities"]
                                    if c["name"] == cap) if k != s][:2]
        if siblings:
            return ("I read this as %s. Is that the core of it, or is it closer to %s?"
                    % (s, " or ".join(siblings)))
    return None


def classify_local(messages):
    """Match the user's messages to weighted skills with keyword scoring. No API calls.

    Each distinct keyword that appears (at a word start, so "summar" matches
    "summarize"/"summary") scores 1; a skill's own name scores 3. Skill weights are each
    score as a percent of the top score; skills under min_weight are dropped. All user
    messages are scored together so answers to follow-up questions add to the picture.
    """
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    text = norm(" ".join(users))
    latest = norm(users[-1]) if users else ""
    asked = sum(1 for m in messages if m.get("role") == "assistant" and _msg_type(m) == "question")
    if is_harm_seeking(latest):
        q = {"type": "question", "text": DECLINE_TEXT}
        return q, json.dumps(q), None
    unsupported = find_unsupported(text)
    for phrase in unsupported:
        text = text.replace(phrase, " ")  # don't score skills off out-of-scope phrases
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
        if unsupported:
            q = {"type": "question", "text": UNSUPPORTED_TEXT % ", ".join(unsupported)}
        else:
            q = {"type": "question", "text":
                 "I couldn't match that to a skill. Describe what the model needs to do "
                 "(e.g. write code, summarize, plan steps, detect emotion, extract data), "
                 "or click a skill in the sidebar."}
        return q, json.dumps(q), None
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[0][0]
    sw = [{"skill": s, "weight": round(100 * sc / top)} for sc, s in scored]
    sw = normalize_weights(sw)
    if asked == 0:
        q = follow_up_question(sw, len(matched[sw[0]["skill"]]))
        if q:
            q = {"type": "question", "text": q}
            return q, json.dumps(q), None

    groups = {}
    for x in sw:
        groups.setdefault(SKILL_TO_CAP[x["skill"]], []).append(x)
    ans = {
        "type": "answer",
        "skills": sw,
        "subtasks": ([{"name": c, "skills": g} for c, g in groups.items()]
                     if len(groups) >= 2 else []),
        "confidence": round(min(0.95, 0.35 + 0.15 * top), 2),
        "summary": "Matched on: " + ", ".join(
            sorted({k for x in sw for k in matched[x["skill"]]}) or [x["skill"] for x in sw]),
    }
    if unsupported:
        ans["unsupported_note"] = (
            "Note: we have no benchmark data for %s, so this recommendation covers only "
            "the text-based parts of your task." % ", ".join(unsupported))
    validate_and_normalize_answer(ans)
    return finish_answer(ans)


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
    if ptype in ("question", "unsupported"):
        parsed = {"type": "question", "text": str(parsed.get("text", "")).strip()}
        return parsed, json.dumps(parsed), None

    if ptype == "answer":
        ok, msg = validate_and_normalize_answer(parsed)
        if not ok:
            full.append({"role": "assistant", "content": raw})
            full.append({"role": "user", "content": msg})
            raw2 = call_llm(full)
            parsed2, _ = extract_json(raw2)
            if parsed2 and parsed2.get("type") == "answer" and validate_and_normalize_answer(parsed2)[0]:
                return finish_answer(parsed2)
            return None, raw, msg or "invalid classification"
        return finish_answer(parsed)

    return None, raw, "Unknown response type: " + str(ptype)


class Handler(BaseHTTPRequestHandler):
    server_version = "ModelAdvisor/1.0"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _bind_user_key(self):
        _request.key = (self.headers.get("X-Api-Key") or "").strip()

    def do_GET(self):
        self._bind_user_key()
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(500, "index.html not found")
            return
        if path == "/api/taxonomy":
            self._send(200, TAXONOMY)
            return
        if path == "/api/config":
            lm = CONFIG["backend"]
            self._send(200, {
                "model": lm.get("model", "local-model"),
                "temperature": lm.get("temperature", 0.0),
            })
            return
        if path == "/api/health":
            self._send(200, self._lmstudio_status())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        self._bind_user_key()
        if self.path.split("?")[0] != "/api/chat":
            self._send(404, {"error": "not found"})
            return
        body = self._read_body()
        messages = body.get("messages", [])
        if not messages:
            self._send(400, {"error": "no messages"})
            return
        try:
            result, content, error = classify(messages)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                pass
            if e.code in (402, 403, 429):
                self._send(429, {"quota": True, "error":
                    "The shared free usage has run out. Please enter your own Ollama API key "
                    "using the \"API key\" button in the header. "
                    "Note: you get a free API key by creating an Ollama account on their website."})
                return
            self._send(502, {"error": "Model API error (%d): %s" % (e.code, detail[:500])})
            return
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            self._send(503, {"error":
                "Cannot reach the model server at " + CONFIG["backend"]["base_url"] +
                ". Reason: " + str(reason)})
            return
        except Exception as e:
            self._send(500, {"error": str(e)})
            return

        if error:
            self._send(200, {"ok": False, "error": error, "content": content})
            return
        self._send(200, {"ok": True, "result": result, "content": content})

    def _lmstudio_status(self):
        if use_local_classifier():
            return {"lmstudio": True, "endpoint": "local", "local": True,
                    "server_key": False, "using_user_key": False}
        lm = CONFIG["backend"]
        base = lm["base_url"].rstrip("/")
        for endpoint in ("/models", "/api/tags"):
            try:
                req = urllib.request.Request(base + endpoint, headers=_headers())
                with urllib.request.urlopen(req, timeout=4) as r:
                    json.loads(r.read().decode("utf-8"))
                return {"lmstudio": True, "endpoint": endpoint,
                        "server_key": bool(get_server_key()),
                        "using_user_key": bool(getattr(_request, "key", ""))}
            except Exception:
                continue
        return {"lmstudio": False, "endpoint": None,
                "server_key": bool(get_server_key()),
                "using_user_key": bool(getattr(_request, "key", ""))}

    def log_message(self, fmt, *args):
        pass  # keep console quiet


def run(open_browser=False):
    host = CONFIG["host"]
    port = CONFIG["port"]
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = "http://%s:%d" % (host, port)
    print("Model Advisor running at", url)
    print("Model endpoint:", CONFIG["backend"]["base_url"])
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)),
                         daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run(open_browser="--open" in sys.argv or "--browser" in sys.argv)
