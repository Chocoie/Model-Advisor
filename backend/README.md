# ModelAdvisor Backend

FastAPI backend for ModelAdvisor.

## Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Run
uvicorn main:app --reload

Then open http://localhost:8000/docs

## Endpoints

### GET /health
Checks that the backend is running.

Response:
```json
{"status": "ok"}
```

### POST /message
Temporary test endpoint for frontend ↔ backend communication.
Echoes the message back (no LLM connected yet).

Request:
```json
{"message": "hello"}
```

Response:
```json
{"reply": "Test response: hello"}
```

Full interactive docs: http://localhost:8000/docs