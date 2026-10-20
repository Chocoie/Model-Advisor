"""
process_ranking_data.py: loads the sponsor's data from data/ and cleans it.

Normalizes dataset names, resolves aliases, takes the first metric at
temperature 0.2, and flips lower-is-better metrics (1 - score).
"""
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