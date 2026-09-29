from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/message", response_model=MessageResponse)
def send_message(request: MessageRequest):
    return MessageResponse(reply=f"Test response: {request.message}")