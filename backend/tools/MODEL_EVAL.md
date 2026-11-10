# Chat model evaluation

Which Ollama Cloud model should run the conversational agent? Date: 2026-11-08.
Script: `tools/eval_scenarios.py`. Gold labels: human consensus skills from the scenario-ranking
study (`data/scenarios.json`, 3,000 scenarios mapped onto our 34 skills).

## Method
- Each scenario's text is sent to the real classifier (`services/llm_classifier.py`) with the 4
  most similar *other* scenarios as examples, the neighbour blend, and the 5-skill cap.
- `--force`: if the model asks a follow-up question, it is told to answer anyway, so every
  scenario is scored. "Asked" counts how often it asked first.
- Metrics, averaged per scenario: **top-1** (is our #1 skill in the human set), **precision**,
  **recall**, **Jaccard** (overlap of predicted vs human skill sets).
- Candidates: open-weight models from American companies only (Google gemma, OpenAI gpt-oss,
  NVIDIA nemotron).

## Round 1: all candidates (60 scenarios, seed 0, definitions off)

| Model | Top-1 | Precision | Recall | Jaccard | Asked | sec/scenario |
|---|---|---|---|---|---|---|
| **gemma4:31b** | 0.83 | **0.68** | 0.68 | **0.52** | 4 | **1.3** |
| nemotron-3-ultra | 0.83 | 0.61 | **0.70** | 0.50 | 2 | 18.2 |
| nemotron-3-super | 0.82 | 0.62 | 0.64 | 0.47 | 0 | 9.1 |
| nemotron-3-nano:30b | 0.80 | 0.62 | 0.64 | 0.47 | 0 | 19.4 |
| gpt-oss:120b | **0.85** | 0.60 | 0.64 | 0.47 | 2 | 2.9 |
| gpt-oss:20b | 0.80 | 0.61 | 0.62 | 0.46 | 2 | 6.6 |

Baselines on 200 scenarios: retrieval only (vote of the 8 nearest scenarios, no LLM) top-1 0.67,
precision 0.57, recall 0.58, Jaccard 0.41.

## Round 2: top 3, fresh sample (100 scenarios, seed 1), skill definitions off / on

| Model | Definitions | Top-1 | Precision | Recall | Jaccard | sec/scenario |
|---|---|---|---|---|---|---|
| gemma4:31b | off | 0.82 | 0.59 | 0.66 | 0.45 | 1.4 |
| gemma4:31b | on | 0.80 | 0.59 | 0.69 | 0.47 | 1.6 |
| gpt-oss:120b | off | 0.72 | 0.56 | 0.67 | 0.43 | 3.3 |
| gpt-oss:120b | on | 0.74 | 0.56 | 0.68 | 0.44 | 3.2 |
| nemotron-3-ultra | off | 0.78 | 0.56 | 0.68 | 0.44 | 16.9 |
| nemotron-3-ultra | on | 0.75 | 0.57 | 0.71 | 0.46 | 20.4 |

## Decision
**gemma4:31b with skill definitions on** (`data/config.json`: `backend.model`, `use_skill_definitions`).
- Best precision and Jaccard in both rounds, and 2-12x faster than the alternatives (about 1.5 s
  per classification), which matters for a chat UI.
- Skill definitions raised Jaccard for all three models (+0.02 gemma4, +0.01 gpt-oss, +0.02
  nemotron), mostly through recall. Small but consistent, so they are on.
- nemotron-3-ultra has slightly higher recall but lower precision and takes 17-20 s per answer.
- gpt-oss:120b had the best top-1 on round 1 but dropped to 0.72-0.74 on the fresh sample.

## Short startup-style prompts (`data/short_prompts.json`)
Real users type one line ("predict which users will churn"), not a paragraph. 30 short startup
requests, hand-labeled with only their 1-3 must-have skills (so precision is not meaningful here).
gemma4:31b, definitions on.

| Setup | Asked a question first | Top-1 | Must-haves found (recall) |
|---|---|---|---|
| Retrieval only (no LLM) | - | 0.27 | 0.55 |
| Bot before answer-first | 25 of 30 | 1.00 (5 answered) | 1.00 |
| Same, forced to answer | - | 0.93 | 1.00 |
| Same, forced, no scenario examples | - | 0.93 | 0.98 |
| **Bot with answer-first** | **0 of 30** | **0.90** | **0.98** |

The bot asked a clarifying question on most short prompts even though its forced answer was
right ~93% of the time. Now, on the first message (4+ words), it answers with its best guess and
adds its question as "To fine-tune this: ..." (`answer_first_min_words` in `config.json`).
Paragraph scenarios are unchanged (seed 1, n=100: top-1 0.79, precision 0.59, recall 0.68,
Jaccard 0.46). Scenario examples neither help nor hurt short prompts.

```
python tools/eval_scenarios.py --mode llm --file data/short_prompts.json --n 30 --defs --show
```

## Caveats
- 60-100 scenarios per run; differences under ~0.03 are within noise.
- Human annotators chose from GPT-4.1-mini's top-10 skills, so gold labels lean toward that model.
- Scenarios are paragraph-length; real users type shorter prompts, so live accuracy will be lower.
- Not tested: Mistral (not American; also needs a paid Ollama plan), clef / beam / muse-spark-open
  (origin unverified), and the families in `blocked_backend_families`.

## Reproduce (from backend/)
```
python tools/eval_scenarios.py --mode retrieval --n 200
python tools/eval_scenarios.py --mode llm --n 100 --seed 1 --force --model gemma4:31b --defs
```
