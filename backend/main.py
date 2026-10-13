from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ranking_data import rank_skills, VALID_SKILLS

app = FastAPI()

# Allow the frontend (Next.js runs on port 3000) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# What the frontend sends
class MessageRequest(BaseModel):
    message: str

# What the backend sends back
class MessageResponse(BaseModel):
    reply: str

class RankingRequest(BaseModel):
    skills: list[str]
    top_n: int = 3


class ModelRanking(BaseModel):
    model: str
    avg_rank: float
    scores: dict[str, float]


class RankingResponse(BaseModel):
    results: list[ModelRanking]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/message", response_model=MessageResponse)
def send_message(request: MessageRequest):
    return MessageResponse(reply=f"Test response: {request.message}")


@app.get("/skills")
def list_skills():
    return sorted(VALID_SKILLS)


@app.post("/rankings", response_model=RankingResponse)
def get_rankings(request: RankingRequest):
    try:
        results = rank_skills(request.skills, request.top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RankingResponse(results=results)