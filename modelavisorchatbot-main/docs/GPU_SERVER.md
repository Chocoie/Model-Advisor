# Running Model Advisor on a GPU server

Notes on moving from the hosted Ollama API (shared key) to a model we host ourselves.
Nothing here is built yet except where marked.

## What changes

| | Hosted Ollama API (now) | Self-hosted GPU server |
|---|---|---|
| Auth | Ollama API key (`OLLAMA_API_KEY`) | None by default; add a token or keep the server private |
| Cost | Free quota per key, then the user's own key | GPU billed per hour whether used or not |
| Abuse looks like | Drained quota / charges | Wasted GPU time, slow replies for real users |
| Model choice | Whatever Ollama's cloud offers | We pull exactly the models we allow |
| Latency | Network round trip to Ollama | Depends on our GPU and model size |

## Config change

The app already speaks the OpenAI-compatible API, so the switch is configuration only
(`config.json`):

```json
"backend": {
  "base_url": "http://<gpu-host>:11434/v1",
  "api_key": "env:MODEL_SERVER_TOKEN",
  "model": "<an allowed model>",
  "temperature": 0.0,
  "timeout_seconds": 300
},
"classifier": "llm"
```

- Ollama on the GPU box listens on port 11434. vLLM and LM Studio use the same
  `/v1/chat/completions` shape (vLLM default port 8000, LM Studio 1234).
- Leave `api_key` empty if the server is private and has no token.
- `classifier` is `"llm"` by default. `"local"` is the keyword matcher and makes no API calls; the app also falls back to it automatically when no API key is set.
- The model name must not match `blocked_backend_families` (the server refuses to start if it does). Blocked families (Qwen, DeepSeek, GLM, Kimi and similar) can still be recommended to users; they just cannot run the chatbot or classifier.
  With our own server we choose which models get pulled, so the exclusion is easy to enforce.

## Securing it

1. **Keep the model server private.** Firewall or VPC rule so only the app server can reach
   port 11434/8000. Never expose Ollama or vLLM directly to the internet (no auth by default).
2. **The app server is the only public entry point.** Put it behind a reverse proxy (HTTPS)
   and, if it is not meant to be public, a login.
3. **If the model server must be reachable over the network,** put a reverse proxy with a
   bearer token in front of it and set that token as `MODEL_SERVER_TOKEN` in `.env`.
4. **Never commit the token.** `.env` is already in `.gitignore`.

## Picking hardware and a model

- Small models (up to about 8B) fit a single 16-24 GB GPU; a 27-32B model needs about 24-48 GB
  at 4-bit quantization, more at full precision.
- The classifier only returns short JSON, so a small, instruction-following model is enough.
  Test it against the cases in `tests/` before committing to one.
- Allowed non-Chinese options already in the benchmark: Gemma, Llama/Cogito, Mistral,
  Falcon, OLMo.

## Checklist for cutover

1. Provision the GPU box and install Ollama (or vLLM).
2. Pull only allowed models; confirm the model name is not on `blocked_backend_families`.
3. Lock the model port to the app server only.
4. Update `config.json` `backend` and set `classifier` to `"llm"`.
5. Set `MODEL_SERVER_TOKEN` in `.env` if a token is used.
6. Run `python -m unittest discover tests`, then try a few real prompts through the UI.
7. Remove the UI "API key" button (or leave it off) if users no longer bring their own key.
