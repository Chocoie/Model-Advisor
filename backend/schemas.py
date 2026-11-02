"""
Pydantic models for API requests and responses.
Grouped by area: auth, conversations, and skill rankings.
"""

from pydantic import BaseModel, EmailStr, Field # object validity and python <-> json 

# --- User ---
class UserCredentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

# --- Messages (1) ---
# A user message 
class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

# --- Rankings ---
class SkillWeight(BaseModel):
    skill: str
    weight: int = Field(default=100, ge=1, le=100)
    
class RankingRequest(BaseModel): # Top n models ranking request
    skills: list[str] = []
    weights: list[SkillWeight] | None = None 
    top_n: int = Field(default=3, ge=2, le= 30)

class ModelRanking(BaseModel): # One model only
    model: str
    avg_rank: float | None = None
    fit: float | None = None   
    scores: dict[str, float]

class RankingResponse(BaseModel): #  Top n models results
    results: list[ModelRanking]  # [{model, avg_rank, scores}, {...}]

# --- Message (2) ---
# LLM response: skills/results stay empty when it's a follow-up question
class MessageResponse(BaseModel):
    reply: str
    skills: list[str] = []
    results: list[ModelRanking] = []

# --- Conversations ---
class NewConversation(BaseModel):
    conversation_id: str

class Message(BaseModel):
    role: str
    content: str

class ConversationHistory(BaseModel):
    conversation_id: str
    messages: list[Message]

class ConversationSummary(BaseModel):
    conversation_id: str
    title: str 

class ConversationsList(BaseModel):
    conversations: list[ConversationSummary]