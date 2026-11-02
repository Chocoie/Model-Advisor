"""
Auth routes (/auth): register, login, logout, and get the current user.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from schemas import UserCredentials
from store.user import create_user, authenticate_user, create_session, end_session
from dependencies import get_current_user, bearer

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register")
def register_user(user: UserCredentials):
    if not create_user(user.email, user.password):
        raise HTTPException(status_code=409, detail="Email already registered")
    token = create_session(user.email)
    return{"message" : "User registered", "token": token, "email": user.email}

@router.post("/login")
def user_login(user: UserCredentials):
    if not authenticate_user(user.email, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_session(user.email)
    return {"token": token,"email" : user.email}

@router.post("/logout")
def user_logout(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if end_session(credentials.credentials) is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"message": "Logged out"}

@router.get("/auth/me")
def valid_token(current_user: str = Depends(get_current_user)):
    return {"email": current_user}
