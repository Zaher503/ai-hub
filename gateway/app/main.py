"""The AI hub gateway: one front door for Ollama and Whisper.

Speaks the Ollama API as is, so a client only changes its URL. GPU calls wait in a single line and
run one at a time; the gateway swaps what is loaded on the card and tells each caller how long the
wait is (response headers, GET /queue/estimate, GET /queue/tickets/<id>).
"""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from scheduler import PRIORITY, Job, Scheduler

log = logging.getLogger("gateway")
OLLAMA = os.environ.get("OLLAMA_URL", "http://ollama:11434").rstrip("/")
WHISPER = os.environ.get("WHISPER_URL", "http://whisper:9000").rstrip("/")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
# Headers that must not be copied between the two legs of a proxied call. accept-encoding is dropped so the
# upstream answers uncompressed and the bytes can be streamed through untouched.
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host", "accept-encoding"}
# Ollama calls that use the GPU for real work
OLLAMA_KINDS = {"/api/chat": "chat", "/api/generate": "generate", "/api/embed": "embed", "/api/embeddings": "embed",
                "/v1/chat/completions": "chat", "/v1/completions": "generate", "/v1/embeddings": "embed"}

sched = Scheduler(max_waiting=int(os.environ.get("MAX_WAITING", "200")))
http: httpx.AsyncClient


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http
    http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10))
    yield
    await http.aclose()


app = FastAPI(title="ai-hub gateway", lifespan=lifespan)


def _priority(request: Request) -> int:
    return PRIORITY.get(request.headers.get("x-priority", "normal").lower(), 1)


def _info(job: Job, waited: float, eta_at_admission: dict) -> dict:
    return {
        "X-Job-Id": job.id,
        "X-Queue-Position": str(eta_at_admission["position"]),
        "X-Queue-Wait-Seconds": f"{waited:.1f}",
        "X-Queue-Eta-Seconds": str(eta_at_admission["eta_seconds"]),
    }


async def _unload_ollama() -> None:
    try:
        ps = (await http.get(f"{OLLAMA}/api/ps", timeout=5)).json().get("models", [])
        for m in ps:
            await http.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
        for _ in range(20):  # wait until the card is really free
            if not (await http.get(f"{OLLAMA}/api/ps", timeout=5)).json().get("models"):
                return
            await asyncio.sleep(0.5)
    except (httpx.HTTPError, ValueError) as e:
        log.warning("could not unload Ollama: %s", e)


async def _unload_whisper() -> None:
    try:
        await http.post(f"{WHISPER}/unload", timeout=15)
    except httpx.HTTPError as e:
        log.warning("could not unload Whisper: %s", e)


async def _make_room(job: Job) -> None:
    """Free the card of the other backend before this job loads its own model."""
    resident = sched.resident
    if resident is None or resident[0] == job.backend:
        return
    await (_unload_whisper() if job.backend == "ollama" else _unload_ollama())


async def _wait_turn(request: Request, job: Job) -> bool:
    """True once it is this job's turn; False if the caller hung up while waiting."""
    while not job.start.is_set():
        if await request.is_disconnected():
            sched.finish(job, ok=False)
            return False
        try:
            await asyncio.wait_for(job.start.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
    return job.state == "running"


async def _run_queued(request: Request, job: Job, upstream: httpx.Request) -> Response:
    admission = sched.estimate(job.backend, job.kind, job.model, job.priority, job.audio_seconds)
    try:
        sched.submit(job)
    except OverflowError:
        return JSONResponse({"error": "the AI queue is full, try again later"}, status_code=429, headers={"Retry-After": "60"})
    t0 = time.monotonic()
    if not await _wait_turn(request, job):
        return Response(status_code=499)
    headers = _info(job, time.monotonic() - t0, admission)
    try:
        await _make_room(job)
        upstream_resp = await http.send(upstream, stream=True)
    except httpx.HTTPError as e:
        sched.finish(job, ok=False)
        return JSONResponse({"error": f"{job.backend} unavailable: {e}"}, status_code=502, headers=headers)

    async def body():
        ok = False
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
            ok = upstream_resp.status_code < 400
        finally:
            await upstream_resp.aclose()
            sched.finish(job, ok=ok)

    out = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in HOP}
    return StreamingResponse(body(), status_code=upstream_resp.status_code, headers={**out, **headers})


async def _passthrough(request: Request, base: str, path: str) -> Response:
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    req = http.build_request(request.method, f"{base}{path}", params=request.query_params, content=body, headers=headers)
    try:
        resp = await http.send(req, stream=True)
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream unavailable: {e}"}, status_code=502)

    async def stream():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    out = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}
    return StreamingResponse(stream(), status_code=resp.status_code, headers=out)


# queue API
@app.get("/queue")
async def queue():
    return sched.snapshot()


@app.get("/queue/estimate")
async def estimate(kind: str = "chat", model: str = "", audio_seconds: float | None = None, priority: str = "normal"):
    backend = "whisper" if kind == "transcribe" else "ollama"
    return sched.estimate(backend, kind, model or WHISPER_MODEL, PRIORITY.get(priority, 1), audio_seconds)


@app.get("/queue/tickets/{ticket}")
async def ticket(ticket: str):
    job = sched.tickets.get(ticket)
    if job is None:
        return JSONResponse({"state": "unknown"}, status_code=404)
    return sched.status(job)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# Whisper through the queue
@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    model = request.headers.get("x-model", WHISPER_MODEL)
    audio = request.headers.get("x-audio-seconds")
    job = Job("whisper", "transcribe", model, _priority(request), request.headers.get("x-client", ""),
              float(audio) if audio else None, request.headers.get("x-queue-ticket"))
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    upstream = http.build_request("POST", f"{WHISPER}/v1/audio/transcriptions", content=request.stream(), headers=headers)
    return await _run_queued(request, job, upstream)


@app.get("/whisper/models")
async def whisper_models():
    return await _passthrough_get(WHISPER, "/models")


async def _passthrough_get(base: str, path: str) -> Response:
    try:
        r = await http.get(f"{base}{path}", timeout=10)
        return JSONResponse(r.json(), status_code=r.status_code)
    except (httpx.HTTPError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# Everything else is Ollama
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def ollama(request: Request, path: str):
    full = "/" + path
    kind = OLLAMA_KINDS.get(full)
    if kind is None or request.method != "POST":
        return await _passthrough(request, OLLAMA, full)
    raw = await request.body()
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return await _passthrough(request, OLLAMA, full)
    model = str(data.get("model") or "")
    # Loading or unloading a model with no prompt is not GPU work to queue
    if kind == "generate" and not data.get("prompt") and not data.get("images"):
        return await _passthrough(request, OLLAMA, full)
    job = Job("ollama", kind, model, _priority(request), request.headers.get("x-client", ""), None,
              request.headers.get("x-queue-ticket"))
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    upstream = http.build_request("POST", f"{OLLAMA}{full}", content=raw, headers=headers)
    return await _run_queued(request, job, upstream)
