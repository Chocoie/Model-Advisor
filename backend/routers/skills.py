"""
Skill routes (/skills): list valid skills and rank models for a set of skills.
"""

from fastapi import APIRouter, HTTPException
from schemas import RankingRequest, RankingResponse
from services.ranking import VALID_SKILLS, rank_skills

router = APIRouter(prefix="/skills", tags=["skills"])

@router.get("")
def list_skills():
    return sorted(VALID_SKILLS)

@router.post("/ranking", response_model=RankingResponse)
def get_ranking(request: RankingRequest):
    try:
        results = rank_skills(request.skills, request.top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RankingResponse(results=results)



