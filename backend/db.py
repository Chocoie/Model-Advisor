import psycopg
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "data" / ".env")
DB_URL = os.environ.get("DATABASE_URL")

if not DB_URL:
    raise RuntimeError("DATABASE_URL not set in data/.env")

def get_connection():
    return psycopg.connect(DB_URL)
