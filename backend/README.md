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


### POST /conversations
Creates a new empty conversation.

Response:
```json
{
  "conversation_id": "3bde8170-cab8-428e-88fd-1b551fa9c391"
}
```


### GET /conversations/{conversation_id} 
Returns the full message history.

Response
```json
{
  "conversation_id": "75f89191-9457-4e60-b134-dcdfab325af5",
  "messages": [
    {
      "role": "user",
      "content": "hi?"
    },
    {
      "role": "assistant",
      "content": "Test response: hi?"
    }
  ]
}
```
Errors: `404` if the conversation doesn't exist.


### POST /conversations/{conversation_id}/messages
Adds a user message and returns the reply. Both are saved in the history.

Request:
```json
{
    "message": "hi?"
}
```

Response
```json
{
  "reply": "Test response: hi?"
}
```

Full interactive docs: http://localhost:8000/docs