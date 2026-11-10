# Classifier evaluation against human skill rankings

Date: 2026-11-03. Script: `tools/eval_scenarios.py`. Data: `data/scenarios.json`, built by
`tools/build_scenario_data.py` from the scenario-ranking study (3,000 scenarios, 2-3 human
annotators each, consensus weights mapped onto our 34-skill taxonomy).

## Method

- Gold = the human consensus skill set for a scenario.
- Prediction = the skills the classifier returns for the scenario text.
- Metrics, averaged per scenario: top-1 (is our #1 skill in the human set), precision,
  recall, Jaccard.
- Sample: scenarios shuffled with a fixed seed, first 60 used (200 for keyword and retrieval
  in the first run).
- LLM runs use `gemma4:31b` via Ollama Cloud, temperature 0, with the 4 most similar scenarios
  (excluding the one being tested) added to the prompt.
- The LLM sometimes asks a follow-up question instead of answering. Those are not scored in
  the normal mode. `--no-questions` re-runs only those scenarios with an instruction to answer
  anyway.

## Results

| Classifier | Model | Scenarios | Top-1 | Precision | Recall | Jaccard |
|---|---|---|---|---|---|---|
| Keyword (offline fallback, since removed) | none (regex) | 200 | 0.18 | 0.18 | 0.09 | 0.08 |
| Retrieval only (nearest-scenario vote) | none (tf-idf) | 200 | 0.67 | 0.57 | 0.58 | 0.41 |
| Retrieval only | none (tf-idf) | 60 | 0.75 | 0.61 | 0.59 | 0.44 |
| LLM + examples, answered unprompted | gemma4:31b | 19 of 60 | 0.95 | 0.66 | 0.78 | 0.57 |
| LLM + examples, forced to answer (`--no-questions`) | gemma4:31b | 37 | 0.84 | 0.58 | 0.68 | 0.48 |
| **After tuning**: LLM + examples + fewer questions + 5-skill cap + neighbour blend (seed 0) | gemma4:31b | 55 of 60 | 0.84 | 0.67 | 0.68 | 0.52 |
| After tuning, fresh sample (seed 1) | gemma4:31b | 94 of 100 | 0.82 | 0.59 | 0.66 | 0.45 |
| Retrieval only, same fresh sample (seed 1) | none (tf-idf) | 100 | 0.76 | 0.57 | 0.60 | 0.41 |

The LLM asked a follow-up question on about two-thirds of the scenarios (41 of 60 in one run,
37 in another; the runs are not fully reproducible). When forced, it answered every one.

## Changes made after the first round

1. Prompt now prefers answering; a detailed first message (40+ words) that still gets a
   question is re-asked once with "answer now". Follow-ups dropped from ~40 of 60 to 5-6.
2. LLM answers keep at most 5 skills with weight >= 50 (`llm_max_skills`, `llm_min_weight`
   in `config.json`).
3. Neighbour blend: for inputs of 15+ words, a skill under weight 80 that none of the 8 most
   similar human-ranked scenarios chose is scaled by 0.7. Skills are never added.
4. Eval script: `--seed` for fresh samples; the tested scenario is excluded from prompt examples
   and from the blend. Held-out evaluation on the GPU server's model has not been run yet.
5. Skills with no benchmark (role playing, translation, creative writing, spatial, tone
   adaptation) now produce a "no benchmark data" note on LLM answers too.

Effect: precision and Jaccard rose and the chatbot no longer stalls on detailed inputs. The
fresh sample is lower than seed 0 (0.59 vs 0.67 precision), so expect roughly 0.6 precision
and 0.45-0.5 Jaccard. The LLM beats retrieval alone on every metric.

## Caveats

- Small samples (19 and 37 scenarios); treat the LLM numbers as indicative.
- Humans picked only from the 10 skills GPT-4.1-mini suggested, so the gold labels are biased
  toward that model's choices.
- Scenarios are paragraph-length. Real users type shorter prompts, so scores on live input
  will likely be lower.
- Seven human-chosen skills have no benchmarked equivalent and were dropped from the gold
  set (Response Adaptation, Creative Generation, Spatial Reasoning, Role Playing, Machine
  Translation, Divergent Thinking, Novel Concept Formation). Response Adaptation was picked
  1,997 times, the largest coverage gap.
- The mapping of the other 14 non-taxonomy skills (`data/scenario_skill_map.json`) is a
  judgment call and has not been reviewed.

## Reproduce

```
python tools/eval_scenarios.py --mode retrieval --n 200
python tools/eval_scenarios.py --mode llm --n 60                  # needs an API key
python tools/eval_scenarios.py --mode llm --n 60 --no-questions
```
