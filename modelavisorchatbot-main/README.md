# Model Advisor

Tell it what task you need an AI model for, and it recommends the best models
for the job.

Works on **Windows** and **Mac** (and Linux).

---

## Easy setup (about 5 minutes)

### Step 1 — Install Python (one time only)

This app needs a free program called **Python** to run. You must install it
**before** you start the app.

1. Go to **<https://www.python.org/downloads/>**
2. Click the big yellow **Download Python** button.
3. Open the file you downloaded and follow the installer.
4. **Windows only:** on the first installer screen, tick the box
   **"Add python.exe to PATH"** before clicking *Install Now*.

### Step 2 — Download this app

1. At the top of this GitHub page, click the green **Code** button.
2. Click **Download ZIP**.
3. Find the downloaded ZIP file, right-click it and choose **Extract All**
   (Mac: double-click it).

### Step 3 — Start the app

- **Windows:** open the extracted folder and double-click **`run.bat`**.
- **Mac:** open the **Terminal** app, type `bash ` (with a space after it),
  drag the **`run.sh`** file into the Terminal window, and press **Enter**.

Your web browser opens with the app. Leave the black window open while you use
it.

### Step 4 — Add your Ollama API key

The app has no built-in key; everyone uses their own.

1. Create an account on Ollama's website and make a key:
   <https://ollama.com/settings/keys>
2. Add usage credits to that account (<https://ollama.com/plans>). Ollama
   charges every cloud request to credits, so a key with no credits gets refused.
3. Click **🔑 API key** at the top of the app page, paste your key and click **Save**.

The key stays in your browser. You can add or change it any time while the app is running.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| A black window appears and closes straight away (Windows) | Python is not installed, or the "Add python.exe to PATH" box was not ticked. Redo Step 1. |
| "Python not found" (Mac) | Install Python from the link in Step 1, then try Step 3 again. |
| "No API key is set" | Add your key (Step 4). |
| "...account behind it has no usage credits" | Your key works but the account has no credits. Add credits at <https://ollama.com/plans>. |
| Red dot / "model offline" | Check your internet connection, or add your key (Step 4). |
| Browser does not open | Open <http://127.0.0.1:8080> yourself. |

---

# More information

Everything below is optional and for advanced users.

## What it does

A chatbot that classifies a task into a fixed taxonomy (the LLM capability
benchmark: 9 capabilities → 34 skills). It converses briefly to gather context,
then returns a strict, validated classification and the top models for it.

`server.py` holds the taxonomy and validates every result against
`taxonomy.json`, so the result can only ever be a category that exists in the
list.

```
launcher  →  starts local server  →  opens browser (http://127.0.0.1:8080)
                                               │
                         browser ──POST /api/chat──▼── Ollama Cloud API
                         (index.html)      (server.py)    (configured backend)
```

> Requirements: Python 3.8+ (standard library only — no `pip install`).

## Where your API key goes

- **In the browser:** the **🔑 API key** button. The key is kept in your
  browser's local storage and sent to the local server with each request. The
  server never writes it to disk.
- There is no default server key. `OLLAMA_API_KEY` in `.env` is ignored.
  `backend.api_key` in `config.json` is only for a private model server that
  needs a token (see `docs/GPU_SERVER.md`).

## Run from the command line

```bash
python3 launcher.py     # Mac / Linux
py -3 launcher.py       # Windows
```

On Mac/Linux you can also make the script executable once with
`chmod +x run.sh` and then run `./run.sh`.

## Use a different provider or a local model

Edit `config.json` → `backend`:

| Provider | `base_url` | `model` |
|---|---|---|
| Ollama Cloud (default) | `https://ollama.com/v1` | `gemma4:31b` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenRouter | `https://openrouter.ai/api/v1` | (any, e.g. `openai/gpt-4o-mini`) |
| Ollama (local) | `http://127.0.0.1:11434/v1` | `qwen2.5:7b-instruct` |
| LM Studio (local) | `http://127.0.0.1:1234/v1` | `local-model` |

For a fully offline local model: install [Ollama](https://ollama.com/), run
`ollama pull qwen2.5:7b-instruct`, set the Ollama (local) `base_url` and
`model` above, set `"api_key": ""`, then start the app.

## Customize the categories

Edit `taxonomy.json` (plain JSON). The bot locks itself to whatever is in that
file. Restart the app after editing.

## Config reference

`config.json`:

- `port` — local web app port (default `8080`).
- `backend.base_url` — model API endpoint.
- `backend.api_key` — empty by default (users enter their own key in the app).
  Set it only for a private model server that needs a token, e.g.
  `"env:MODEL_SERVER_TOKEN"` to read it from the environment.
- `backend.model` — model id.
- `backend.temperature` — `0.0` keeps classification deterministic.

## Build a standalone .exe (Windows)

For teammates without Python:

1. `pip install pyinstaller`
2. Run `build.bat`

Output: `dist/ModelAdvisor.exe`.

## More troubleshooting

- **"Model API error (401)"** — invalid or missing API key.
- **"HTTP 503"** — can't reach the model server URL.
