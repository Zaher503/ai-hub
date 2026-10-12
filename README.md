# ai-hub

One shared AI stack for everything on the Zephyrus: a single Ollama (all LLM models), a single Whisper (all speech models), and a gateway in front that puts every GPU job in one line, swaps what is loaded on the card, and tells each caller how long the wait is.

Every app talks to `http://ai-hub:11434` on the shared Docker network `ai`. The gateway speaks the normal Ollama API, so an app only changes its `OLLAMA_URL`.

## Where it fits

```
Dates (api, worker) ─┐
Foodbox (app)        ├─> ai-hub gateway :11434 ─┬─> ollama   (gemma4:e4b, ...)
Transcriber          ─┘   queue, ETA, model swap └─> whisper  (large-v3, ...)   ──> RTX 3070 (8 GB)
```

| Container | What | Image |
|---|---|---|
| `ollama` | Ollama with every LLM; models live in `/srv/ai-hub/ollama` | `ollama/ollama` |
| `whisper` | faster-whisper behind `/v1/audio/transcriptions`; models come from the shared Hugging Face cache in `/srv/ai-hub/hf`, so nothing is downloaded twice | `ghcr.io/zaher503/ai-hub-whisper` |
| `gateway` | Queue, wait estimates, model switching | `ghcr.io/zaher503/ai-hub-gateway` |
| `model-pull` | Pulls the models listed in `OLLAMA_MODELS`, then exits | `ollama/ollama` |

Only one job uses the GPU at a time. Before a job runs, the gateway frees the card of the other backend (unloads Ollama before Whisper, and Whisper before Ollama).

## Using it from an app

Use the Ollama API exactly as before, against `http://ai-hub:11434`. Optional headers:

| Header | Meaning |
|---|---|
| `X-Priority: high \| normal \| low` | `high` for something a person is waiting on, `low` for bulk work (default `normal`) |
| `X-Client: dates-worker` | Name shown in `/queue` |
| `X-Queue-Ticket: <uuid>` | Lets you poll your place in line while the request is waiting |

Every GPU response carries what the wait was:

| Response header | Meaning |
|---|---|
| `X-Queue-Position` | Jobs ahead when you joined (0 = started right away) |
| `X-Queue-Eta-Seconds` | Estimated wait plus run time at the moment you joined |
| `X-Queue-Wait-Seconds` | How long you actually waited |
| `X-Job-Id` | Job id |

While a request is waiting you cannot see its response yet, so use:

- `GET /queue/estimate?kind=chat|embed|generate|transcribe&model=...&audio_seconds=...&priority=...` returns `{position, wait_seconds, run_seconds, eta_seconds}` for a job you are about to send.
- `GET /queue/tickets/<ticket>` returns the live position and estimate for a request you sent with `X-Queue-Ticket`.
- `GET /queue` returns what is running, who is waiting, what is loaded, and the run times it has learned.

Estimates start from defaults and get better as jobs finish (a moving average per model; Whisper per second of audio). Send `X-Audio-Seconds` with a transcription so its estimate is accurate.

### Whisper

`POST /v1/audio/transcriptions` (multipart: `file`, optional `model`, `language`, `vad_filter`, `beam_size`, `stream`). Returns `{language, duration, text, segments: [{start, end, text, no_speech_prob, avg_logprob}]}`. With `stream=true` it returns newline-delimited JSON, one `segment` line at a time and a final `done` line, so a caller can show progress. Send `X-Model` (and `X-Audio-Seconds`) as headers so the gateway can queue by model.

## Running it

```bash
# One time on the Zephyrus
sudo install -d -o zaher -g zaher /srv/ai-hub /opt/ai-hub
cp .env.example /opt/ai-hub/.env        # then edit
docker compose -f /opt/ai-hub/compose.yaml up -d
```

After that, every push to `main` deploys itself (below).

## CI/CD

`.github/workflows/ci.yml` lints, runs the tests and builds both images. When it passes on `main`, `cd.yml` publishes the images to GHCR and a self-hosted runner on the Zephyrus (label `ai-hub`, installed in `/opt/actions-runner-ai-hub`) copies `compose.yaml` to `/opt/ai-hub`, pulls the new images, restarts the stack and health-checks `http://127.0.0.1:11434/healthz`.

Secrets and data stay on the server: `/opt/ai-hub/.env`, `/srv/ai-hub/ollama`, `/srv/ai-hub/hf`.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-ci.txt
ruff check . && pytest -q
```

## Other apps on the Zephyrus

| App | Repo | Talks to the hub for |
|---|---|---|
| Foodbox (recipes) | `Zaher503/foodbox` | LLM, Whisper |
| Dates (date planner) | `Zaher503/date-planner` | LLM, Whisper |
| Transcriber (interviews) | `Zaher503/transcriber` | Whisper |

Each has its own repo, the same CI/CD pipeline, and its own runner.
