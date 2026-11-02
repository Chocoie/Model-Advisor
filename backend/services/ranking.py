"""
Model ranking logic.
Scores each model on a skill from its benchmark results and returns the top models
for one or more skills.
"""

from services.data_loader import MAPPING, SCORES, VALID_SKILLS, get_standardized_keys

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

def rank_skills(weights, top_n=3):
    """
    Rank models by weighted fit: each skill score is divided by the best model's
    score on that skill, then averaged using the skill weights.
    weights: [{"skill", "weight"}]. Returns [{"model", "fit", "scores"}], fit in %.
    Raises ValueError if weights is empty or contains an unknown skill.
    """
    if not weights:
        raise ValueError("No skills given")
    for w in weights:
        if w["skill"] not in VALID_SKILLS:
            raise ValueError(f"Unknown skill: {w['skill']}")
        
    # get skill scores of best model
    best = {w["skill"]: max(skill_score(m, w["skill"]) for m in SCORES) for w in weights}
    total = sum(w["weight"] for w in weights) 

    results = []
    for m in SCORES:
        # get each skil's scores
        scores = {w["skill"]: round(skill_score(m, w["skill"]), 3) for w in weights}
        # divide by best score for skill to scale score to [0,1]
        fit = sum(w["weight"] * scores[w["skill"]] / best[w["skill"]]
                  for w in weights if best[w["skill"]]) / total
        results.append({"model": m, "fit": round(100 * fit, 1), "scores": scores})
    results.sort(key=lambda x: -x["fit"])
    return results[:top_n]