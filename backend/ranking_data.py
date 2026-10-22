"""
ranking_data.py: ranks models for the skills found by the Chat Agent.

A model's skill score = sum of its scores on that skill's datasets.
Models are ranked per skill, then combined by average rank (lower is better).
The scores come from process_ranking_data.py.
"""

from pathlib import Path
from process_ranking_data import MAPPING, SCORES, VALID_SKILLS, get_standardized_keys

# ----- 1. Skill score ----------------------------------------

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

# ----- 2. Skill ranking ----------------------------------------

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
