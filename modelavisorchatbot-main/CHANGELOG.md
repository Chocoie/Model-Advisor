# Changelog

Model Advisor uses version numbers `MAJOR.MINOR.PATCH`. The current version is
`__version__` in `server.py`. It's shown next to the title in the app, printed at
startup, and returned by `/api/health` and `/api/config`.

To release: bump `__version__`, add a section here, commit, then tag it (`git tag v0.0.2`).

## 0.0.1 (2026-11-10)
First versioned release.
- The LLM (gemma4:31b on Ollama Cloud) does all the categorizing. The keyword classifier is gone.
  Without an API key the chat shows an error instead of a keyword guess.
- Clearer errors when Ollama refuses a key: says whether it was your key or the server's,
  and why (no credits, bad key, rate limit).
- A refused key (402/401/429) is no longer retried as if it were a format error.
- The message box wraps and grows as you type (Enter sends, Shift+Enter adds a new line).
- Version number shown in the app header.
