"""
Shared FastAPI dependencies used across routers.
get_current_user: reads the Bearer token and returns the logged-in user's email.
"""

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from store.user import get_user_email


bearer = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials=Depends(bearer)):
    email = get_user_email(credentials.credentials)
    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return email