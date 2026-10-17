import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
# ----- 1. Data loading and processing ----------------------------------------
def load_json(name):
    """ Load a JSON file from data/ and return it as Python dicts/lists."""
    with open(DATA_DIR/ name, encoding="utf-8") as f:
        return json.load(f)

def load_csv(name,  encoding="utf-8"):
    """ Load a CSV file from data/ as a list of dicts."""
    with open(DATA_DIR/ name, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))

# load files
    
RESULTS = load_json("results.json")
SKILLS = load_csv("skill_definitions.csv", encoding="cp1252")
MAPPING = load_csv("dataset_skill_mapping_updated.csv")
METRICS = load_csv("metrics.csv")

# list of skills in mapping
VALID_SKILLS = {
    row["Skill"] for row in MAPPING
}
# set of metrics where a lower score is considered better 
LOWER_IS_BETTER = {
    m["metric_key"] for m in METRICS if m["lower_is_better"].lower() == "true"
    }

# mapping names that normalize() can't match automatically
ALIASES = {
    "CNN/DailyMail": ["cnndm"],
    "Disinformation (climate & covid)": ["disinformation:climate", 
                                         "disinformation:covid"],
}

def normalize(name):
    """ 
    Make dataset names comparable across files: lowercase, drop "mmlu:",
    remove spaces, _ and -.
    e.g. "BoolQ" -> "boolq", "mmlu:college_biology" -> "collegebiology"
    """

    name = name.lower()
    if name.startswith("mmlu:"):
        name = name[len("mmlu:"):]
    for ch in " _-":
        name = name.replace(ch, "")
    return name

def get_standardized_keys(dataset_name):
    """ 
    Translate a dataset name from the mapping CSV into its key(s) in SCORES.
    Always returns a list (aliases can have several keys).
    e.g. "BoolQ" -> ["boolq"], "CNN/DailyMail" -> ["cnndm"]
    """
    if dataset_name in ALIASES:
        return ALIASES[dataset_name]
    return [normalize(dataset_name)]

def build_score_table():
    """
    Build {model: {dataset: score}} from RESULTS, using the first metric
    at temperature 0.2. Lower-is-better metrics are flipped (1 - score)
    so higher always means better.
    e.g. {"gpt-4o": {"boolq": 0.65, "anatomy": 0.49}, ...}
    """
    table = {}
    for model, groups in RESULTS.items():
        table[model] = {}
        for group, datasets in groups.items():
            for dataset, metrics in datasets.items():
                metric, scores_by_temp = next(iter(metrics.items())) 
                score = scores_by_temp["0.2"]
                if metric.split(" (")[0] in LOWER_IS_BETTER:
                    score = (1 - score) #flip score
                table[model][normalize(dataset)] = score
    return table

SCORES = build_score_table()

# ----- 2. Skill score ----------------------------------------

def datasets_for_skill(skill):
    """ Return the names of every dataset that tests this skill. """
    return [row["Dataset"] for row in MAPPING if row["Skill"] == skill]


def skill_score(model, skill):
    """ 
    A model's score on one skill: the sum of its scores on that skill's datasets.
    Datasets with no score (e.g. EntityMatching) are skipped.
    """
    total = 0
    for dataset in datasets_for_skill(skill):
        score_keys = get_standardized_keys(dataset)
        if all(k in SCORES[model] for k in score_keys): # if this model has a score for every key of this dataset
            total += sum(SCORES[model][k] for k in score_keys) / len(score_keys)
    return total

# ----- 3. Skill ranking ----------------------------------------

def rank_skill(skill):
    """ 
    Rank all models on one skill, best first.
    Returns [(model, score), ...], e.g. [("llama", 30.1), ("phi", 27.8), ...]
    """
    rank_by_skill = [(model, skill_score(model, skill)) for model in SCORES]
    rank_by_skill.sort(key= lambda x:x[1], reverse=True)
    return rank_by_skill

def rank_skills(skills, top_n=3):
    """
    Rank models across several skills by average rank (lower = better).
    Returns the top_n models as dicts: {"model", "avg_rank", "scores"}.
    Raises ValueError if skills is empty or contains an unknown skill.
    """
    if not skills:
        raise ValueError("No skills given")
    for skill in skills:
       if skill not in VALID_SKILLS:
           raise ValueError(f"Unknown skill: {skill}")

    rank_total = {model : 0 for model in SCORES}
    skill_scores = {model: {} for model in SCORES}

    for skill in skills:
        for position, (model, score) in enumerate(rank_skill(skill), start=1):
            rank_total[model] += position
            skill_scores[model][skill] = round(score, 3)
    results = [
        {"model": m, "avg_rank": round(rank_total[m] / len(skills), 3), "scores": skill_scores[m]}
        for m in SCORES
    ]
    results.sort(key= lambda x:x["avg_rank"])
    return results[:top_n]
