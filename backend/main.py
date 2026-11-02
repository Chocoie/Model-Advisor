"""
Entry point for the ModelAdvisor backend.
Creates the FastAPI app, sets up CORS for the frontend, and registers the routers.
Run with: uvicorn main:app --reload
"""

#imports
from fastapi import FastAPI#, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from routers import auth, conversations, skills
app = FastAPI()

# Allow the frontend (Next.js runs on port 3000) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Health Checkpoint

@app.get("/health")
def health():
    return {"status": "ok"}


# Routes
app.include_router(skills.router)
app.include_router(auth.router)
app.include_router(conversations.router)