import asyncio
import json

import httpx
import pytest

import main
from scheduler import Scheduler

calls: list[str] = []
delay = {"chat": 0.0}


def R(status: int, body=None) -> httpx.Response:
    """A response that streams, the way a real upstream does."""
    raw = json.dumps(body).encode() if body is not None else b""
    return httpx.Response(status, headers={"content-type": "application/json"}, stream=httpx.ByteStream(raw))


def upstream(request: httpx.Request) -> httpx.Response:
    calls.append(f"{request.method} {request.url.path}")
    path = request.url.path
    if path == "/api/ps":
        return R(200, {"models": []})
    if path == "/api/chat":
        return R(200, {"message": {"content": "hi"}, "done": True})
    if path == "/api/tags":
        return R(200, {"models": [{"name": "gemma"}]})
    if path == "/unload":
        return R(200, {"unloaded": True})
    if path == "/v1/audio/transcriptions":
        return R(200, {"text": "hello", "segments": []})
    return R(404)


@pytest.fixture
async def client():
    calls.clear()
    main.sched = Scheduler()

    async def slow(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat" and delay["chat"]:
            await asyncio.sleep(delay["chat"])
        return upstream(request)

    main.http = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://hub") as c:
        yield c
    delay["chat"] = 0.0
    await main.http.aclose()


async def test_chat_is_queued_and_reports_headers(client):
    r = await client.post("/api/chat", json={"model": "gemma", "messages": []})
    assert r.status_code == 200 and r.json()["done"] is True
    assert r.headers["x-queue-position"] == "0"
    assert float(r.headers["x-queue-eta-seconds"]) > 0
    assert "x-job-id" in r.headers
    assert main.sched.running is None  # the turn is handed back


async def test_other_ollama_calls_are_not_queued(client):
    r = await client.get("/api/tags")
    assert r.json()["models"][0]["name"] == "gemma"
    assert "x-queue-position" not in r.headers


async def test_unload_request_skips_the_queue(client):
    r = await client.post("/api/generate", json={"model": "gemma", "keep_alive": 0})
    assert "x-queue-position" not in r.headers


async def test_second_caller_waits_for_the_first(client):
    delay["chat"] = 0.3
    first = asyncio.create_task(client.post("/api/chat", json={"model": "gemma", "messages": []}, headers={"x-queue-ticket": "t1"}))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(client.post("/api/chat", json={"model": "gemma", "messages": []}, headers={"x-queue-ticket": "t2"}))
    await asyncio.sleep(0.05)
    waiting = (await client.get("/queue/tickets/t2")).json()
    assert waiting["state"] == "waiting" and waiting["position"] == 1
    snap = (await client.get("/queue")).json()
    assert snap["running"] is not None and len(snap["waiting"]) == 1
    a, b = await asyncio.gather(first, second)
    assert a.headers["x-queue-position"] == "0"
    assert b.headers["x-queue-position"] == "1"
    assert float(b.headers["x-queue-wait-seconds"]) >= 0.15


async def test_estimate_endpoint(client):
    r = await client.get("/queue/estimate", params={"kind": "transcribe", "audio_seconds": 600})
    body = r.json()
    assert body["position"] == 0 and body["run_seconds"] > 10


async def test_whisper_goes_through_the_queue_and_frees_ollama(client):
    main.sched.resident = ("ollama", "gemma")
    r = await client.post("/v1/audio/transcriptions", content=b"audio", headers={"x-audio-seconds": "30"})
    assert r.json()["text"] == "hello"
    assert "GET /api/ps" in calls  # Ollama was checked and unloaded before Whisper ran
    assert json.loads(json.dumps(main.sched.snapshot()))["resident"]["backend"] == "whisper"
