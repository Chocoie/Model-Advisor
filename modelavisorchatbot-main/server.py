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


def validate_and_normalize_answer(ans):
    raw_skills = ans.get("skills")
    if isinstance(raw_skills, str):
        raw_skills = [raw_skills]
    if not raw_skills:
        raw_skills = [ans.get("skill", "")]

    skills = []
    seen = set()
    for name in raw_skills:
        got = canonical_skill(name)
        if got:
            canon_skill = got[0]
            if canon_skill not in seen:
                seen.add(canon_skill)
                skills.append(canon_skill)

    if not skills:
        msg = ("Your 'skills' are not in the taxonomy. Use EXACT skill names from "
               "the list and respond again with a valid 'answer' JSON object.")
        return False, msg

    ans["skills"] = skills
    ans["skill"] = skills[0]
    ans["capability"] = SKILL_TO_CAP[skills[0]]
    try:
        conf = float(ans.get("confidence", 0.5))
        ans["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        ans["confidence"] = 0.5
    ans["summary"] = str(ans.get("summary", "")).strip()
    return True, None


def get_recommendations(skills):
    """Rank models by their average rank across one or more skills (lower = better)."""
    models = BENCHMARK["models"]
    avg_rank = {}
    for m in models:
        total = 0
        for skill in skills:
            score = m["skills"].get(skill)
            if score is None:
                total += len(models)
            else:
                better = sum(1 for o in models if (o["skills"].get(skill) or 0) > score)
                total += better + 1
        avg_rank[m["name"]] = total / len(skills)

    ordered = sorted(models, key=lambda m: (avg_rank[m["name"]], -m["overall"]))
    return [{
        "name": m["name"],
        "avg_rank": round(avg_rank[m["name"]], 2),
        "overall": m["overall"],
        "scores": {s: m["skills"].get(s) for s in skills},
    } for m in ordered[:3]]


def use_local_classifier():
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
        "capability": SKILL_TO_CAP[skills[0]],
        "confidence": round(min(0.95, 0.35 + 0.15 * top), 2),
        "summary": "Matched on: " + ", ".join(
            sorted({k for s in skills for k in matched[s]}) or skills),
    }
    ans["recommendations"] = get_recommendations(skills)
    clean = {k: v for k, v in ans.items() if k != "recommendations"}
    return ans, json.dumps(clean), None


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
                parsed2["recommendations"] = get_recommendations(parsed2["skills"])
                clean2 = {k: v for k, v in parsed2.items() if k != "recommendations"}
                return parsed2, json.dumps(clean2), None
            return None, raw, msg or "invalid classification"
        parsed["recommendations"] = get_recommendations(parsed["skills"])
        clean = {k: v for k, v in parsed.items() if k != "recommendations"}
        return parsed, json.dumps(clean), None

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
