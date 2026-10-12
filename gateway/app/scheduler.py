"""One GPU turn at a time, with a wait-time estimate for everyone in line."""
import asyncio
import itertools
import time
import uuid
from dataclasses import dataclass, field

PRIORITY = {"high": 0, "normal": 1, "low": 2}

# Starting guesses in seconds; real run times replace them as jobs finish
DEFAULT_RUN = {"chat": 30.0, "generate": 30.0, "embed": 2.0, "transcribe": 90.0}
DEFAULT_SWAP = {"ollama": 10.0, "whisper": 6.0}
# Whisper cost per second of audio (large-v3 runs about 15x real time on the 3070)
DEFAULT_PER_AUDIO_SECOND = 0.07
EMA = 0.3


@dataclass
class Job:
    backend: str  # ollama or whisper
    kind: str  # chat, generate, embed, transcribe
    model: str
    priority: int = 1
    client: str = ""
    audio_seconds: float | None = None
    ticket: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    state: str = "waiting"  # waiting, running, done, cancelled
    seq: int = 0
    est_run: float = 0.0
    start: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def target(self) -> tuple[str, str]:
        return (self.backend, self.model)


class Scheduler:
    def __init__(self, max_waiting: int = 200):
        self.waiting: list[Job] = []
        self.running: Job | None = None
        self.resident: tuple[str, str] | None = None
        self.run_ema: dict[str, float] = {}
        self.per_audio_second = DEFAULT_PER_AUDIO_SECOND
        self.swap = dict(DEFAULT_SWAP)
        self.max_waiting = max_waiting
        self.tickets: dict[str, Job] = {}
        self._seq = itertools.count()

    # estimates
    def run_estimate(self, kind: str, model: str, audio_seconds: float | None) -> float:
        if kind == "transcribe" and audio_seconds:
            return max(audio_seconds * self.per_audio_second, 3.0)
        return self.run_ema.get(f"{kind}:{model}", DEFAULT_RUN.get(kind, 30.0))

    def swap_cost(self, current: tuple[str, str] | None, nxt: tuple[str, str]) -> float:
        if current == nxt:
            return 0.0
        return self.swap.get(nxt[0], 8.0)  # nothing loaded, the other backend, or another model on the same one

    def _ahead(self, priority: int, seq: int | None) -> list[Job]:
        """Waiting jobs that run before a job with this priority (and sequence, when it is already queued)."""
        out = [j for j in self.waiting if j.priority < priority or (j.priority == priority and (seq is None or j.seq < seq))]
        return sorted(out, key=lambda j: (j.priority, j.seq))

    def _eta(self, target: tuple[str, str], est: float, priority: int, seq: int | None) -> tuple[int, float]:
        """(jobs ahead, seconds until this job starts)."""
        t = 0.0
        pos = 0
        resident = self.resident
        if self.running is not None:
            r = self.running
            t += max(r.est_run - (time.monotonic() - (r.started or time.monotonic())), 1.0)
            pos += 1
            resident = r.target
        for j in self._ahead(priority, seq):
            t += self.swap_cost(resident, j.target) + j.est_run
            resident = j.target
            pos += 1
        return pos, t + self.swap_cost(resident, target)

    def estimate(self, backend: str, kind: str, model: str, priority: int = 1, audio_seconds: float | None = None) -> dict:
        run = self.run_estimate(kind, model, audio_seconds)
        pos, wait = self._eta((backend, model), run, priority, None)
        return {"position": pos, "wait_seconds": round(wait, 1), "run_seconds": round(run, 1), "eta_seconds": round(wait + run, 1)}

    def status(self, job: Job) -> dict:
        if job.state == "waiting":
            pos, wait = self._eta(job.target, job.est_run, job.priority, job.seq)
        else:
            pos, wait = 0, 0.0
        return {"id": job.id, "state": job.state, "position": pos, "wait_seconds": round(wait, 1),
                "run_seconds": round(job.est_run, 1), "eta_seconds": round(wait + job.est_run, 1),
                "kind": job.kind, "model": job.model, "client": job.client}

    # queue operations
    def submit(self, job: Job) -> Job:
        if len(self.waiting) >= self.max_waiting:
            raise OverflowError("queue is full")
        job.seq = next(self._seq)
        job.est_run = self.run_estimate(job.kind, job.model, job.audio_seconds)
        self.waiting.append(job)
        if job.ticket:
            self.tickets[job.ticket] = job
        self._prune()
        self._dispatch()
        return job

    def _dispatch(self) -> None:
        if self.running is not None or not self.waiting:
            return
        self.waiting.sort(key=lambda j: (j.priority, j.seq))
        job = self.waiting.pop(0)
        job.state = "running"
        job.started = time.monotonic()
        self.running = job
        job.start.set()

    def finish(self, job: Job, ok: bool = True) -> None:
        """Called when a running job ends or a waiting one gives up."""
        if job.state == "waiting":
            self.waiting = [j for j in self.waiting if j is not job]
            job.state = "cancelled"
        elif job.state == "running":
            job.finished = time.monotonic()
            job.state = "done" if ok else "cancelled"
            if self.running is job:
                self.running = None
                self.resident = job.target
                if ok:
                    self._learn(job, job.finished - (job.started or job.finished))
        job.start.set()
        self._dispatch()

    def _learn(self, job: Job, seconds: float) -> None:
        if job.kind == "transcribe" and job.audio_seconds:
            self.per_audio_second += EMA * (seconds / job.audio_seconds - self.per_audio_second)
        else:
            key = f"{job.kind}:{job.model}"
            old = self.run_ema.get(key, seconds)
            self.run_ema[key] = old + EMA * (seconds - old)

    def _prune(self) -> None:
        cutoff = time.monotonic() - 600
        for k in [k for k, j in self.tickets.items() if j.finished and j.finished < cutoff]:
            del self.tickets[k]

    def snapshot(self) -> dict:
        jobs = sorted(self.waiting, key=lambda j: (j.priority, j.seq))
        return {
            "running": self.status(self.running) | {"running_seconds": round(time.monotonic() - (self.running.started or 0), 1)} if self.running else None,
            "waiting": [self.status(j) for j in jobs],
            "resident": {"backend": self.resident[0], "model": self.resident[1]} if self.resident else None,
            "learned": {"run_seconds": {k: round(v, 1) for k, v in self.run_ema.items()},
                        "whisper_seconds_per_audio_second": round(self.per_audio_second, 3)},
        }
