from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ranking_data import rank_skills, VALID_SKILLS
from conversations import create_conversation, get_messages, add_message

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


class ConversationResponse(BaseModel):
    conversation_id: str


class Message(BaseModel):
    role: str
    content: str


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: list[Message]



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

@app.post("/conversations", response_model=ConversationResponse)
def new_conversation():
    return ConversationResponse(conversation_id=create_conversation())


@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
def read_conversation(conversation_id: str):
    messages = get_messages(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistory(conversation_id=conversation_id, messages=messages)

@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
def send_conversation_message(conversation_id: str, request: MessageRequest):
    if get_messages(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    add_message(conversation_id, "user", request.message)

    reply = f"Test response: {request.message}"  # placeholder until the chat agent (Task 4)

    add_message(conversation_id, "assistant", reply)
    return MessageResponse(reply=reply)