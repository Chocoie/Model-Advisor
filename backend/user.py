import bcrypt
import secrets

USERS = {} # Shape: {email: password_hash} TODO replace with db
SESSIONS = {}   # Shape: {token: email}   TODO replace with db

def create_user(email: str, password: str):
    """ Create new user with email and hashed password"""
    if email in USERS:
        return False
    hash_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    USERS[email] = hash_password
    return True

def authenticate_user(email : str, password: str):
    """ Authenticate user with email and password"""
    if email not in USERS:
        return False
    return bcrypt.checkpw(password.encode(), USERS[email])

def create_session(email: str):
    """ Creates a token for user"""
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = email
    return token

def get_user_email(token: str):
    """ Returns email linked to token"""
    return SESSIONS.get(token)
