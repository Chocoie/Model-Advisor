"""
user.py: user accounts and login sessions.

Passwords are hashed with bcrypt, emails are stored in lowercase.
Login creates a random session token; logout deletes it.
"""

import bcrypt
import secrets

USERS = {} # Shape: {email: password_hash} TODO replace with db
SESSIONS = {}   # Shape: {token: email}   TODO replace with db

def create_user(email: str, password: str):
    """ Create new user with email and hashed password"""
    email = email.lower()
    if email in USERS:
        return False
    hash_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    USERS[email] = hash_password
    return True

def authenticate_user(email : str, password: str):
    """ Authenticate user with email and password"""
    email = email.lower()
    if email not in USERS:
        return False
    return bcrypt.checkpw(password.encode(), USERS[email])

def create_session(email: str):
    """ Creates a token if user exists."""
    email = email.lower()
    if email not in USERS:
        return None
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = email
    return token

def end_session(token: str):
    """ Deletes token if it exists"""
    return SESSIONS.pop(token, None)

def get_user_email(token: str):
    """ Returns email linked to token"""
    return SESSIONS.get(token)
