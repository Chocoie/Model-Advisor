import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"

def load_json(name):
    """ loads json file and returns dictionnary / list """
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)

def load_csv(name, encoding="utf-8"):
    """ loads csv file and returns dictionnary """
    with open(DATA_DIR / name, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


""" load files """
RESULTS = load_json("results.json") 
SKILLS = load_csv("skill_definitions.csv", encoding="cp1252")
MAPPING = load_csv("dataset_skill_mapping_updated.csv")
METRICS = load_csv("metrics.csv")  


""" set of metric names where lower_is_better = true 
    (scores get flipped later) """
LOWER_IS_BETTER = {
    m["metric_key"] for m in METRICS if m["lower_is_better"].lower() == "true"
}

""" mapping names that normalize() can't match automatically"""
ALIASES = {
    "CNN/DailyMail": ["cnndm"],
    "Disinformation (climate & covid)": ["disinformation:climate", 
                                         "disinformation:covid"],
}

""" valid skill name (has datasets) """
VALID_SKILLS = {row["Skill"] for row in MAPPING}


def normalize(name):
    """ normalizes names to lower caser, no special char 
        BoolQ -> boolq, legal_support -> legalsupport """
    
    name = name.lower()

    if name.startswith("mmlu:"): 
        name = name[len("mmlu:"):]
    for ch in " _-":
        name = name.replace(ch, "")
    return name


def result_keys(dataset_name):
    """ checks for aliases, otherwise normalize """

    if dataset_name in ALIASES:
        return ALIASES[dataset_name]
    return [normalize(dataset_name)]


def build_score_table():
    """ builds a table of each model's score on each dataset: 
    {model: {dataset: score}} """

    table = {}
    for model, groups in RESULTS.items():
        table[model] = {}
        for group, datasets in groups.items():
            for dataset, metrics in datasets.items():
                metric, temps = next(iter(metrics.items()))  # first metric
                score = temps["0.2"]
                if metric in LOWER_IS_BETTER:
                    score = 1 - score  # flip so higher is always better
                table[model][normalize(dataset)] = score
    return table

""" build score table """
SCORES = build_score_table()


def datasets_for_skill(skill):
    """ returns a list of datasets mapped to a skill"""
    return [row["Dataset"] for row in MAPPING if row["Skill"] == skill]


def skill_score(model, skill):
    """ sum of the model's scores on every dataset linked to this skill """

    total = 0
    for dataset in datasets_for_skill(skill):
        keys = result_keys(dataset)
        if all(k in SCORES[model] for k in keys):  
            total += sum(SCORES[model][k] for k in keys) / len(keys)
    return total


def rank_skill(skill):
    """ all models, best first: [(model, score), ...]"""

    ranked = [(model, skill_score(model, skill)) for model in SCORES]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked

def rank_skills(skills, top_n=3):
    """Top models across several skills, by average rank (lower = better)"""

    if not skills:
        raise ValueError("No skills given")
    for skill in skills:
        if skill not in VALID_SKILLS:
            raise ValueError(f"Unknown skill: {skill}")

    rank_total = {model: 0 for model in SCORES}
    skill_scores = {model: {} for model in SCORES}

    for skill in skills:
        for position, (model, score) in enumerate(rank_skill(skill), start=1):
            rank_total[model] += position
            skill_scores[model][skill] = round(score, 3)

    results = [
        {"model": m, "avg_rank": rank_total[m] / len(skills), "scores": skill_scores[m]}
        for m in SCORES
    ]
    results.sort(key=lambda r: r["avg_rank"])
    return results[:top_n]

if __name__ == "__main__":
    for skill in ["Emotion Inference", "Moral / Ethical Reasoning"]:
        print(f"\n{skill}")
        for pos, (m, s) in enumerate(rank_skill(skill)[:12], start=1):
            print(pos, round(s, 4), m)