"""Whisper as a service: faster-whisper behind an OpenAI-style transcription endpoint.

Models come from the shared Hugging Face cache (HF_HOME), so nothing is downloaded twice. One model
is loaded at a time; the gateway decides when, and calls /unload to free the card for Ollama.
"""
import gc
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("whisper")
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8_float16")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
HF_HUB = Path(os.environ.get("HF_HOME", "/hf")) / "hub"

app = FastAPI(title="whisper")
_lock = threading.Lock()
_model = None
_name = None


def _load(name: str):
    global _model, _name
    if _model is not None and _name == name:
        return _model
    _unload()
    from faster_whisper import WhisperModel
    t0 = time.monotonic()
    try:
        _model = WhisperModel(name, device=DEVICE, compute_type=COMPUTE)
    except Exception as e:
        if DEVICE != "cuda":
            raise
        log.warning("GPU load failed (%s), using the CPU", e)
        _model = WhisperModel(name, device="cpu", compute_type="int8")
    _name = name
    log.info("loaded %s in %.1fs", name, time.monotonic() - t0)
    return _model


def _unload() -> bool:
    global _model, _name
    had = _model is not None
    _model, _name = None, None
    gc.collect()
    return had


def decode(path: str) -> np.ndarray:
    """16 kHz mono float32 samples via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         check=True, capture_output=True, timeout=1800).stdout
    return np.frombuffer(raw, dtype=np.float32)


def seg_dict(s) -> dict:
    return {"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": s.text.strip(),
            "no_speech_prob": round(float(s.no_speech_prob), 3), "avg_logprob": round(float(s.avg_logprob), 3)}


@app.post("/v1/audio/transcriptions")
def transcriptions(file: UploadFile = File(...), model: str = Form(""), language: str = Form(""),
                   vad_filter: bool = Form(True), beam_size: int = Form(5), stream: bool = Form(False),
                   condition_on_previous_text: bool = Form(False)):
    import tempfile
    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := file.file.read(1 << 20):
            tmp.write(chunk)
        path = tmp.name
    try:
        audio = decode(path)
    except (subprocess.SubprocessError, OSError) as e:
        raise HTTPException(400, f"could not read the audio: {e}")
    finally:
        Path(path).unlink(missing_ok=True)
    duration = audio.size / 16000
    if audio.size < 1600:
        empty = {"language": language or None, "duration": duration, "text": "", "segments": []}
        return JSONResponse(empty)
    name = model or DEFAULT_MODEL

    def run():
        with _lock:
            m = _load(name)
            segs, info = m.transcribe(audio, language=language or None, vad_filter=vad_filter, beam_size=beam_size,
                                      condition_on_previous_text=condition_on_previous_text)
            yield info, None
            for s in segs:
                yield info, seg_dict(s)

    if stream:
        def lines():
            info = None
            texts = []
            for info, seg in run():
                if seg is None:
                    yield json.dumps({"type": "start", "language": info.language, "duration": duration}) + "\n"
                else:
                    texts.append(seg["text"])
                    yield json.dumps({"type": "segment", **seg}) + "\n"
            yield json.dumps({"type": "done", "language": info.language if info else None, "duration": duration}) + "\n"
        return StreamingResponse(lines(), media_type="application/x-ndjson")

    segments, info = [], None
    for info, seg in run():
        if seg is not None:
            segments.append(seg)
    return {"language": info.language if info else None, "duration": round(duration, 2),
            "text": " ".join(s["text"] for s in segments), "segments": segments}


@app.post("/unload")
def unload():
    with _lock:
        return {"unloaded": _unload()}


@app.get("/models")
def models():
    names = sorted(p.name.removeprefix("models--").replace("--", "/") for p in HF_HUB.glob("models--*faster-whisper*"))
    return {"cached": names, "loaded": _name, "default": DEFAULT_MODEL}


@app.get("/healthz")
def healthz():
    return {"ok": True, "loaded": _name}
