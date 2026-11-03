"""Score the skill classifier against the human consensus from the scenario study.

    python tools/eval_scenarios.py [--mode keyword|retrieval|llm] [--n 200]

keyword    the offline keyword classifier (no API calls)
retrieval  nearest-scenario voting only (shows how much the examples alone carry)
llm        the real chatbot classifier, with few-shot examples (needs an API key;
           the scenario being tested is excluded from its own examples)
Follow-up questions are counted separately, not scored. Reports, per scenario averaged: top-1 hit (is our #1 skill in the human set?),
precision, recall, and Jaccard of the predicted vs human skill sets.
"""

import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import server  # noqa: E402


NO_QUESTIONS = "\n\n(Do not ask follow-up questions. Give your best answer now.)"


def predict(mode, text, no_questions=False):
    if mode == "retrieval":
        votes = Counter()
        for sc in server.similar_scenarios(text, 8, exclude=text):
            for x in sc["skills"]:
                votes[x["skill"]] += x["weight"]
        return [s for s, _ in votes.most_common(4)]
    msgs = [{"role": "user", "content": text + (NO_QUESTIONS if no_questions else "")}]
    if mode == "keyword":
        ans, _, _ = server.classify_local(msgs)
    else:
        orig_ex, orig_sim = server.examples_block, server.similar_scenarios
        # keep the scenario under test out of both the prompt examples and the neighbour blend
        server.examples_block = lambda q, k=4, exclude=None: orig_ex(q, k, exclude=text)
        server.similar_scenarios = lambda q, k=4, exclude=None: orig_sim(q, k, exclude=text)
        try:
            ans, _, _ = server.classify(msgs)
        finally:
            server.examples_block, server.similar_scenarios = orig_ex, orig_sim
    return ans.get("skills", []) if ans and ans.get("type") == "answer" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="keyword", choices=["keyword", "retrieval", "llm"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed; use a new one for a fresh sample")
    ap.add_argument("--no-questions", action="store_true",
                    help="llm: re-run scenarios where the chatbot asked a question, "
                         "telling it to answer anyway, and score only those")
    a = ap.parse_args()
    rows = [sc for sc in server.SCENARIOS if sc["skills"]]
    random.Random(a.seed).shuffle(rows)
    tot = Counter()
    for sc in rows[:a.n]:
        gold = {x["skill"] for x in sc["skills"]}
        pred = predict(a.mode, sc["text"])
        if pred is None:      # the chatbot asked a follow-up question instead of answering
            tot["asked"] += 1
            if not a.no_questions:
                continue
            pred = predict(a.mode, sc["text"], no_questions=True)
            if pred is None:
                tot["still_asked"] += 1
                continue
        elif a.no_questions:
            continue          # only the previously-asked scenarios are retested
        hit = set(pred) & gold
        tot["n"] += 1
        tot["top1"] += bool(pred and pred[0] in gold)
        tot["p"] += len(hit) / len(pred) if pred else 0
        tot["r"] += len(hit) / len(gold)
        tot["j"] += len(hit) / len(set(pred) | gold) if pred else 0
    n = tot["n"]
    if tot["still_asked"]:
        print("still asked a question despite the instruction: %d" % tot["still_asked"])
    if tot["asked"]:
        print("asked a follow-up instead of answering: %d of %d (scored on the rest)"
              % (tot["asked"], n + tot["asked"]))
    model = server.CONFIG["backend"]["model"] if a.mode == "llm" else "n/a"
    print("%s (model: %s) on %d scenarios: top-1 %.2f  precision %.2f  recall %.2f  Jaccard %.2f"
          % (a.mode, model, n, tot["top1"] / n, tot["p"] / n, tot["r"] / n, tot["j"] / n))


if __name__ == "__main__":
    main()
