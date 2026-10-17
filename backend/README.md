## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### LLM mode (optional)
The chat uses the LLM team's skill classifier (`llm_classifier.py`). The mode is set by `"classifier"` in `data/config.json`:
- `"local"`: keyword matching only, no API key needed
- `"llm"`: uses the LLM (Ollama). Needs a key in `data/.env`:
```
  OLLAMA_API_KEY=your_key_here
```
  If the LLM is unreachable or there's no key, the chat falls back to keyword matching. `.env` is ignored by git, so never commit your key.

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
`top_n` is optional (default 3, between 2 and 30).

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

Errors:
- `400` if a skill name is unknown or the list is empty
- `422` if `top_n` is outside 2–30

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
{"message": "User registered"}
```

Errors:
- `409` if the email is already registered
- `422` if the email is invalid, or the password is shorter than 8 or longer than 72 characters

### POST /auth/login
Checks the email and password and returns a token. Send the token with requests that need a logged-in user.

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

### POST /auth/logout
Ends the session: the token stops working.

Request header:
```
Authorization: Bearer <token>
```

Response:
```json
{"message": "Logged out"}
```

Errors:
- `401` if the token is missing or invalid

### GET /auth/me
Returns the email of the logged-in user. Useful to check if a saved token is still valid.

Request header:
```
Authorization: Bearer <token>
```

Response:
```json
{"email": "user@example.com"}
```

Errors:
- `401` if the token is missing or invalid

## Conversations
All conversation endpoints require the token header:
```
Authorization: Bearer <token>
```
Each conversation belongs to the user who created it. Other users get a `404`.

### POST /conversations
Creates a new empty conversation for the logged-in user.

Response:
```json
{"conversation_id": "3bde8170-cab8-428e-88fd-1b551fa9c391"}
```

Errors:
- `401` if the token is missing or invalid

### GET /conversations
Lists the logged-in user's conversations. The title is the first message (first 50 characters), or "New Conversation" if it's empty.

Response:
```json
{
  "conversations": [
    {"conversation_id": "3bde8170-cab8-428e-88fd-1b551fa9c391", "title": "I need a model for customer support"},
    {"conversation_id": "a91c22e4-5b0f-4c1e-9d2a-7f3e8b6c1d00", "title": "New Conversation"}
  ]
}
```

Errors:
- `401` if the token is missing or invalid

### GET /conversations/{conversation_id}
Returns the full message history.

Response:
```json
{
  "conversation_id": "3bde8170-cab8-428e-88fd-1b551fa9c391",
  "messages": [
    {"role": "user", "content": "I need a model for customer support"},
    {"role": "assistant", "content": "Could you specify the primary tasks the model will handle?..."}
  ]
}
```

Errors:
- `401` if the token is missing or invalid
- `404` if the conversation doesn't exist or belongs to another user

### POST /conversations/{conversation_id}/message
Adds a user message and returns the reply. Both are saved in the history.
The reply is either a follow-up question, or the detected skills and the top models.

Request:
```json
{"message": "I need a model to summarize long reports"}
```

Response (follow-up question):
```json
{"reply": "Could you specify the primary tasks the model will handle?...", "skills": [], "results": []}
```

Response (recommendation):
```json
{
  "reply": "Skills detected:\n- Data Interpretation\n- Summarization\nTop Models:\n1. google_gemma-2-9b-it (avg rank 5.5)\n...",
  "skills": ["Data Interpretation", "Summarization"],
  "results": [
    {"model": "google_gemma-2-9b-it", "avg_rank": 5.5, "scores": {"Data Interpretation": 0.961, "Summarization": 1.233}}
  ]
}
```
`results` uses the same format as `/rankings`. In LLM mode, `reply` also ends with a short explanation of the results.

Errors:
- `401` if the token is missing or invalid
- `404` if the conversation doesn't exist or belongs to another user
- `422` if the message is empty or longer than 2000 characters

Note: users, tokens and conversations are stored in memory for now, so they are lost when the server restarts.

Full interactive docs: http://localhost:8000/docs