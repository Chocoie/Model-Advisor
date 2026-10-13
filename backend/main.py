from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ranking_data import rank_skills, VALID_SKILLS
from conversations import create_conversation, get_messages, add_message
from chat_agent import next_step, explain

app = FastAPI()

# Allow the frontend (Next.js runs on port 3000) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Messages ---

# A user message (used by /message and the conversation endpoint)
class MessageRequest(BaseModel):
    message: str


# Reply from the test endpoint (/message), used by the frontend for now
class MessageResponse(BaseModel):
    reply: str


# --- Rankings ---

class RankingRequest(BaseModel):
    skills: list[str]
    top_n: int = 3


class ModelRanking(BaseModel):
    model: str
    avg_rank: float
    scores: dict[str, float]


class RankingResponse(BaseModel):
    results: list[ModelRanking]


# --- Conversations ---

class ConversationResponse(BaseModel):
    conversation_id: str


class Message(BaseModel):
    role: str
    content: str


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: list[Message]


# Reply in a conversation: skills/results stay empty when it's a follow-up question
class ConversationReply(BaseModel):
    reply: str
    skills: list[str] = []
    results: list[ModelRanking] = []


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


# Test endpoint: TODO remove once the frontend uses the conversation endpoints
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


@app.post("/conversations", response_model=ConversationResponse)
def new_conversation():
    return ConversationResponse(conversation_id=create_conversation())


@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
def read_conversation(conversation_id: str):
    messages = get_messages(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistory(conversation_id=conversation_id, messages=messages)


@app.post("/conversations/{conversation_id}/messages", response_model=ConversationReply)
def send_conversation_message(conversation_id: str, request: MessageRequest):
    if get_messages(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    add_message(conversation_id, "user", request.message)
    step = next_step(get_messages(conversation_id))

    # Agent needs more info: ask a follow-up question
    if step["type"] == "question":
        reply = step["question"]
        add_message(conversation_id, "assistant", reply)
        return ConversationReply(reply=reply)

    # Agent found the skills: rank models and explain
    results = rank_skills(step["skills"])
    reply = explain(step["skills"], results)
    add_message(conversation_id, "assistant", reply)
    return ConversationReply(reply=reply, skills=step["skills"], results=results)