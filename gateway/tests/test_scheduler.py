import asyncio

import pytest

from scheduler import Job, Scheduler


def job(kind="chat", model="gemma", backend="ollama", priority=1, audio=None):
    return Job(backend, kind, model, priority, audio_seconds=audio)


def test_first_job_runs_immediately():
    s = Scheduler()
    a = s.submit(job())
    assert a.state == "running" and s.running is a


def test_fifo_within_priority_and_high_first():
    s = Scheduler()
    a = s.submit(job())
    b = s.submit(job())
    c = s.submit(job(priority=0))
    s.finish(a)
    assert s.running is c
    s.finish(c)
    assert s.running is b


def test_estimate_counts_everyone_ahead():
    s = Scheduler()
    s.submit(job())
    s.submit(job())
    est = s.estimate("ollama", "chat", "gemma")
    assert est["position"] == 2
    assert est["wait_seconds"] > 30  # the running job plus one waiting
    assert est["eta_seconds"] == pytest.approx(est["wait_seconds"] + est["run_seconds"], abs=0.2)


def test_idle_hub_has_no_wait_beyond_the_model_load():
    s = Scheduler()
    est = s.estimate("ollama", "chat", "gemma")
    assert est["position"] == 0
    assert est["wait_seconds"] == s.swap["ollama"]


def test_swapping_backends_costs_time():
    s = Scheduler()
    a = s.submit(job())
    s.finish(a)  # gemma is now loaded
    same = s.estimate("ollama", "chat", "gemma")
    other = s.estimate("whisper", "transcribe", "large-v3", audio_seconds=60)
    assert same["wait_seconds"] == 0
    assert other["wait_seconds"] > 0


def test_whisper_estimate_scales_with_audio():
    s = Scheduler()
    short = s.estimate("whisper", "transcribe", "large-v3", audio_seconds=60)["run_seconds"]
    long = s.estimate("whisper", "transcribe", "large-v3", audio_seconds=600)["run_seconds"]
    assert long > short * 5


def test_cancel_waiting_job_leaves_the_rest():
    s = Scheduler()
    s.submit(job())
    b = s.submit(job())
    c = s.submit(job())
    s.finish(b, ok=False)
    assert s.waiting == [c]


def test_learns_from_finished_jobs():
    s = Scheduler()
    a = s.submit(job())
    a.started -= 10
    s.finish(a)
    assert s.run_ema["chat:gemma"] < 30  # moved towards the 10 s it took


def test_full_queue_is_refused():
    s = Scheduler(max_waiting=1)
    s.submit(job())
    s.submit(job())
    with pytest.raises(OverflowError):
        s.submit(job())


async def test_waiter_wakes_when_turn_comes():
    s = Scheduler()
    a = s.submit(job())
    b = s.submit(job())
    waiter = asyncio.create_task(b.start.wait())
    await asyncio.sleep(0)
    assert not waiter.done()
    s.finish(a)
    await asyncio.wait_for(waiter, 1)
    assert b.state == "running"
