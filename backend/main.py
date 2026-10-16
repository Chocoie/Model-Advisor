from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field # object validity and python <-> json 

from user import create_user, authenticate_user, create_session, get_user_email
from ranking_data import VALID_SKILLS, rank_skills
from conversations import create_conversation, get_messages, add_message
from chat_agent import next_action, explain_results


app = FastAPI()
bearer = HTTPBearer()

# Allow the frontend (Next.js runs on port 3000) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- User ---
class UserCredentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

# --- Messages ---
# # A user message (used by /message and the conversation endpoint)
class MessageRequest(BaseModel):
    message: str

# --- Rankings ---
class RankingRequest(BaseModel): # Top n models ranking request
    skills: list[str]
    top_n: int = Field(default=3, ge=2, le= 30)

class ModelRanking(BaseModel): # One model only
    model: str
    avg_rank: float
    scores: dict[str, float]

class RankingResponse(BaseModel): #  Top n models results
    results: list[ModelRanking]  # [{model, avg_rank, scores}, {...}]

# --- Conversations ---
class NewConversation(BaseModel):
    conversation_id: str

class Message(BaseModel):
    role: str
    content: str

class ConversationHistory(BaseModel):
    conversation_id: str
    messages: list[Message]

# Reply in a conversation: skills/results stay empty when it's a follow-up question
class MessageResponse(BaseModel):
    reply: str
    skills: list[str] = []
    results: list[ModelRanking] = []

# --- Helper ---

def get_current_user(credentials: HTTPAuthorizationCredentials=Depends(bearer)):
    email = get_user_email(credentials.credentials)
    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return email

# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/skills")
def list_skills():
    return sorted(VALID_SKILLS)

@app.post("/rankings", response_model=RankingResponse)
def get_ranking(request: RankingRequest):
    try:
        results = rank_skills(request.skills, request.top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RankingResponse(results=results)

@app.post("/auth/register")
def register_user(user: UserCredentials):
    if not create_user(user.email, user.password):
        raise HTTPException(status_code=409, detail="Email already registered")
    return{"message" : "User registered"}

@app.post("/auth/login")
def verify_user(user: UserCredentials):
    if not authenticate_user(user.email, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_session(user.email)
    return {"token": token,"email" : user.email}

@app.get("/auth/me")
def valid_token(current_user: str = Depends(get_current_user)):
    return {"email": current_user}


@app.post("/conversations", response_model=NewConversation)
def new_conversation():
    conversation_id = create_conversation()
    return NewConversation(conversation_id=conversation_id)

@app.get("/conversations/{conversation_id}", response_model=ConversationHistory)
def get_conversation(conversation_id: str):
    conversation_history = get_messages(conversation_id)
    if conversation_history is None: 
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistory(conversation_id=conversation_id, messages=conversation_history)

@app.post("/conversations/{conversation_id}/message", response_model=MessageResponse)
def send_message(conversation_id: str, request: MessageRequest):
    if get_messages(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    add_message(conversation_id, "user", request.message) #add user msg to conv
    conversation_history = get_messages(conversation_id)
    action = next_action(conversation_history) # get chat agent next action

    if action["type"] == "question":
        reply = action["question"]
        add_message(conversation_id, "assistant", reply)
        return MessageResponse(reply=reply)
    
    skills = action["skills"]
    results = rank_skills(skills)
    reply = explain_results(skills, results)
    add_message(conversation_id, "assistant", reply)

    return MessageResponse(reply=reply, skills=skills, results=results)