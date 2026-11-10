# Changelog

Model Advisor uses version numbers `MAJOR.MINOR.PATCH`. The current version is
`__version__` in `server.py`. It's shown next to the title in the app, printed at
startup, and returned by `/api/health` and `/api/config`.

To release: bump `__version__`, add a section here, commit, then tag it (`git tag v0.0.2`).

## 0.0.1 (2026-11-10)
First versioned release.
- The LLM (gemma4:31b on Ollama Cloud) does all the categorizing. The keyword classifier is gone.
  Without an API key the chat shows an error instead of a keyword guess.
- Clearer errors when Ollama refuses a key: says why (no credits, bad key, rate limit).
- A refused key (402/401/429) is no longer retried as if it were a format error.
- The message box wraps and grows as you type (Enter sends, Shift+Enter adds a new line).
- Version number shown in black next to the "Model Advisor" title.
- No default API key: the server no longer falls back to `OLLAMA_API_KEY` from `.env`.
  Everyone enters their own key with the "API key" button. `backend.api_key` is empty by
  default and only meant for a private model server that needs a token.
