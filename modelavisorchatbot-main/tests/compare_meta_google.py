"""Compare Meta (Llama / Cogito-Llama) and Google (Gemma) models in the benchmark.

    python tests/compare_meta_google.py

Benchmark sizes only overlap at the small end (Llama 3B/8B vs Gemma 4B/9B), so the
comparison is done per size bucket; Gemma-2 27B has no Meta counterpart.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402

FAMILIES = {"Meta": ("llama",), "Google": ("gemma",)}
BUCKETS = [("1-4B", 0, 4), ("5-10B", 5, 10), ("11B+", 11, 1000)]

SCENARIOS = {
    "coding assistant": "write python code and debug functions",
    "support chatbot": "detect user emotion and reply with cultural understanding",
    "report summarizer": "summarize long reports and extract entities",
    "project planner": "plan step by step and break down tasks with constraints",
    "safety review": "resist harmful requests and reduce political bias",
}


def family(name):
    n = name.lower()
    for fam, keys in FAMILIES.items():
        if any(k in n for k in keys):
            return fam
    return None


def size_b(name):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)B", name)
    return float(m.group(1)) if m else None


def in_bucket(m, lo, hi):
    s = size_b(m["name"])
    return s is not None and lo <= s <= hi


def best_of(fam, lo, hi, sw):
    pool = [m for m in server.MODELS if family(m["name"]) == fam and in_bucket(m, lo, hi)]
    return max(pool, key=lambda m: server.fit(m, sw)) if pool else None


def main():
    print("Models in benchmark:")
    for fam in FAMILIES:
        names = [m["name"] for m in server.MODELS if family(m["name"]) == fam]
        print("  %-7s %s" % (fam, ", ".join(names)))
    wins = {"Meta": 0, "Google": 0, "tie": 0}
    for label, lo, hi in BUCKETS:
        print("\n== size bucket %s ==" % label)
        for scen, text in SCENARIOS.items():
            # Include a prior question turn so the local classifier answers directly.
            res, _, _ = server.classify_local([
                {"role": "user", "content": text},
                {"role": "assistant", "content": '{"type": "question", "text": "?"}'},
                {"role": "user", "content": "yes"}])
            if res["type"] != "answer":
                print("  %-18s (needs follow-up, skipped)" % scen)
                continue
            sw = res["weights"]
            meta, goog = best_of("Meta", lo, hi, sw), best_of("Google", lo, hi, sw)
            if not (meta and goog):
                print("  %-18s only one family has a model in this bucket" % scen)
                continue
            fm, fg = server.fit(meta, sw), server.fit(goog, sw)
            win = "tie" if abs(fm - fg) < 0.02 else ("Meta" if fm > fg else "Google")
            wins[win] += 1
            print("  %-18s Meta %-28s %.0f%%   Google %-22s %.0f%%   -> %s" % (
                scen, meta["name"], 100 * fm, goog["name"], 100 * fg, win))
    print("\nTally:", wins)


if __name__ == "__main__":
    main()
