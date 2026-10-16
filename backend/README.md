## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn main:app --reload
```
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

### GET /skills
Returns the list of valid skill names.

Response:
```json
["Analogical Reasoning", "Argument Mining", "Causal Reasoning", "..."]
```

### POST /rankings
Returns the top models for the given skills, ranked by average rank across skills (lower is better).

Request:
```json
{
  "skills": ["Emotion Inference", "Moral / Ethical Reasoning"],
  "top_n": 3
}
```
`top_n` is optional (default: 3).

Response:
```json
{
  "results": [
    {
      "model": "mistralai_Mistral-Small-24B-Instruct-2501",
      "avg_rank": 4.0,
      "scores": {
        "Emotion Inference": 1.688,
        "Moral / Ethical Reasoning": 0.756
      }
    }
  ]
}
```
(Shortened to 1 result here; the real response returns `top_n` results.)

Errors: `400` if a skill name is unknown or the list is empty.


## User
### POST /auth/register
Creates a new user.

Request:
```json
{
  "email": "user@example.com",
  "password": "password"
}
```

Response:
```json
{
  "message": "User registered"
}
```

Errors:
- `409` if the email is already registered
- `422` if the email is invalid, or the password is shorter than 8 or longer than 72 characters

### POST /auth/login 
Authenticates user for login

Request:
```json
{
  "email": "user@example.com",
  "password": "password"
}
```

Response:
```json
{
  "token": "sxgnvJSPIM17ND4Xy_2fLxt3lIsoqI8cuqhhueHS2II",
  "email": "user@example.com"
}
```

Errors:
- `401` if the email or password is invalid
- `422` if the email is invalid, or the password is shorter than 8 or longer than 72 characters

### GET /auth/me
Returns the email of the logged-in user. Useful to check if a saved token is still valid.

Request header:
```
Authorization: Bearer <token>
```

Response:
```json
{
  "email": "user@example.com"
}
```

Errors:
- `401` if the token is invalid
- `403` if the header is missing

### Conversations
> Stored in memory for now: conversations are lost when the server restarts.
> Replies come from a placeholder keyword-based agent (`chat_agent.py`) until the LLM agent is ready.

### POST /conversations
Creates a new empty conversation.

Response:
```json
{"conversation_id": "3bde8170-cab8-428e-88fd-1b551fa9c391"}
```

### GET /conversations/{conversation_id}
Returns the full message history.

Response:
```json
{
  "conversation_id": "3bde8170-cab8-428e-88fd-1b551fa9c391",
  "messages": [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Can you tell me more about the task? What should the model do?"}
  ]
}
```
Errors: `404` if the conversation doesn't exist.

### POST /conversations/{conversation_id}/message
Adds a user message and returns the reply. Both are saved in the history.
The reply is either a follow-up question, or the detected skills and the top models.

Request:
```json
{"message": "I need a model to solve math word problems"}
```

Response (follow-up question):
```json
{"reply": "Can you tell me more about the task? What should the model do?", "skills": [], "results": []}
```

Response (recommendation):
```json
{
  "reply": "Skills detected: Quantitative Reasoning\nTop models:\n1. ...",
  "skills": ["Quantitative Reasoning"],
  "results": [
    {"model": "...", "avg_rank": 1.0, "scores": {"Quantitative Reasoning": 11.129}}
  ]
}
```
`results` uses the same format as `/rankings`.

Errors: `404` if the conversation doesn't exist.


Note: users, tokens and conversations are stored in memory for now, so they are lost when the server restarts.

Full interactive docs: http://localhost:8000/docs