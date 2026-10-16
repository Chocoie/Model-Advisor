import bcrypt

#{{"email": hash_password}, ...}
USERS = {} # TODO delete when db ready and replace related action with queries

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
