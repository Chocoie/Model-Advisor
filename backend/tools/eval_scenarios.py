"""Score the skill classifier against the human consensus from the scenario study.

Run from backend/:

    python tools/eval_scenarios.py [--mode keyword|retrieval|llm] [--n 60] [--seed 0]
                                   [--model gemma4:31b] [--defs] [--no-examples] [--out results.jsonl]

keyword    the offline keyword classifier (no API calls)
retrieval  nearest-scenario voting only (shows how much the examples alone carry)
llm        the real chat classifier, with few-shot examples (needs OLLAMA_API_KEY;
           the scenario being tested is excluded from its own examples and the blend)

--model        override the Ollama model in data/config.json
--defs         add the skill definitions to the prompt (use_skill_definitions)
--no-examples  leave the similar-scenario examples out of the prompt
--out          append one JSON line with the settings and scores to this file

Follow-up questions are counted separately, not scored. Reports, per scenario averaged:
top-1 hit (is our #1 skill in the human set?), precision, recall, and Jaccard of the
predicted vs human skill sets, plus errors and seconds per scenario.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from services import llm_classifier as lc  # noqa: E402
from services import scenarios  # noqa: E402

NO_QUESTIONS = "\n\n(Do not ask follow-up questions. Give your best answer now.)"


def predict(mode, text, no_questions=False, examples=True):
    """Predicted skills, None if the classifier asked a question, or raises on API errors."""
    if mode == "retrieval":
        votes = Counter()
        for sc in scenarios.similar_scenarios(text, 8, exclude=text):
            for x in sc["skills"]:
                votes[x["skill"]] += x["weight"]
        return [s for s, _ in votes.most_common(4)]
    msgs = [{"role": "user", "content": text + (NO_QUESTIONS if no_questions else "")}]
    if mode == "keyword":
        ans, _, _ = lc.classify_local(msgs)
    else:
        ans, _, err = lc.classify(msgs, exclude=text, examples=examples)
        if err:
            raise RuntimeError(err)
    return ans.get("skills", []) if ans and ans.get("type") == "answer" else None


def score_one(a, sc):
    gold = {x if isinstance(x, str) else x["skill"] for x in sc["skills"]}
    t0 = time.time()
    asked_first = False
    try:
        pred = predict(a.mode, sc["text"], examples=not a.no_examples)
        if pred is None and a.force:
            asked_first = True
            pred = predict(a.mode, sc["text"], no_questions=True, examples=not a.no_examples)
    except Exception as e:  # API errors, timeouts, bad JSON after retries
        return {"error": repr(e)[:200], "secs": time.time() - t0}
    out = {"secs": time.time() - t0, "asked_first": asked_first, "text": sc["text"], "pred": pred, "gold": sorted(gold)}
    if pred is None:
        out["asked"] = True
        return out
    hit = set(pred) & gold
    out.update(top1=bool(pred and pred[0] in gold),
               p=len(hit) / len(pred) if pred else 0,
               r=len(hit) / len(gold),
               j=len(hit) / len(set(pred) | gold) if pred else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="keyword", choices=["keyword", "retrieval", "llm"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed; use a new one for a fresh sample")
    ap.add_argument("--model", help="Ollama model to use instead of the one in config.json")
    ap.add_argument("--defs", action="store_true", help="include skill definitions in the prompt")
    ap.add_argument("--no-examples", action="store_true", help="no similar-scenario examples in the prompt")
    ap.add_argument("--force", action="store_true",
                    help="if the classifier asks a question, re-ask with 'answer now' and score that")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls (llm mode)")
    ap.add_argument("--out", help="append a JSON line with settings and scores to this file")
    ap.add_argument("--file", help="score this JSON file ({scenarios: [{text, skills: [names]}]}) "
                                   "instead of the human-ranked scenarios, e.g. data/short_prompts.json")
    ap.add_argument("--show", action="store_true", help="print each prompt with predicted vs expected skills")
    a = ap.parse_args()

    if a.model:
        lc.CONFIG["backend"]["model"] = a.model
    lc.CONFIG["use_skill_definitions"] = a.defs

    if a.file:
        with open(a.file, encoding="utf-8") as f:
            rows = json.load(f)["scenarios"]
    else:
        rows = [sc for sc in scenarios.SCENARIOS if sc["skills"]]
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.n]
    workers = a.workers if a.mode == "llm" else 1
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda sc: score_one(a, sc), rows))

    if a.show:
        for r in results:
            mark = "ASKED" if r.get("asked") else ("ok " if r.get("top1") else "MISS") if "p" in r else "ERR"
            print("%-5s %s" % (mark, r.get("text", "")))
            print("      got:  %s\n      want: %s" % (r.get("pred"), r.get("gold")))
    scored = [r for r in results if "p" in r]
    n = len(scored)
    summary = {
        "mode": a.mode, "file": a.file,
        "model": lc.CONFIG["backend"]["model"] if a.mode == "llm" else None,
        "defs": a.defs, "examples": not a.no_examples, "force": a.force,
        "seed": a.seed, "n": len(rows), "scored": n,
        "asked": sum(1 for r in results if r.get("asked") or r.get("asked_first")),
        "errors": sum(1 for r in results if "error" in r),
        "top1": round(sum(r["top1"] for r in scored) / n, 3) if n else None,
        "precision": round(sum(r["p"] for r in scored) / n, 3) if n else None,
        "recall": round(sum(r["r"] for r in scored) / n, 3) if n else None,
        "jaccard": round(sum(r["j"] for r in scored) / n, 3) if n else None,
        "secs_per_scenario": round(sum(r["secs"] for r in results) / len(results), 2) if results else None,
    }
    errs = [r["error"] for r in results if "error" in r]
    if errs:
        print("first error:", errs[0])
    print(json.dumps(summary))
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
