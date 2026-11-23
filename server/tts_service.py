"""HTTP service: Arabic question -> LLM answer -> vocalised text -> voice message.

Run with:  uvicorn server.tts_service:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              model metadata and attribution
    POST /api/tts             text -> WAV (browser/mobile simple path)
    POST /api/voice/answer    question -> LLM -> diacritised -> WAV
    POST /api/voice/answer/stream  same, but one WAV per sentence (NDJSON)

The ONNX session is created once, protected by a lock, and never reloaded per
request. The model and G2P are served from ``server/assets``; no remote
repository is contacted at inference time.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
from urllib.parse import quote

import httpx
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .answer import (
    cut_at_sentence,
    generate_answer,
    iter_answer_deltas,
    iter_sentences,
    sanitize,
)
from .diacritize import vocalize

DEFAULT_ASSETS = Path(__file__).resolve().parent / "assets"
ATTRIBUTION = (
    "Arabic Speech Corpus (Halabi 2016), CC BY 4.0. "
    "Halabi, Nawar (2016). Modern standard Arabic phonetics for speech synthesis. "
    "PhD thesis, University of Southampton."
)

APP_PAGE = Path(__file__).resolve().parent / "app.html"





class VoiceEngine:
    """Loads the ONNX model and the Arabic G2P once."""

    def __init__(self, assets: Path | None = None, threads: int = 4):
        self.assets = Path(assets or os.getenv("TINY_VOCA_ASSETS", DEFAULT_ASSETS))
        # Load the exact G2P file used by the browser demo, under its own module
        # name: importing a plain ``arabic_text`` could pick up another copy
        # already present in sys.modules. Dataclasses need a registered module.
        spec = importlib.util.spec_from_file_location(
            "server_arabic_text", self.assets / "arabic_text.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.arabic_text = module
        self.NothingToPronounce = module.NothingToPronounce

        self.config = json.loads((self.assets / "config.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(self.assets / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._lock = threading.Lock()
        self.model = os.getenv("TINY_VOCA_LLM_MODEL", "gpt-4o-mini")

    def phonemize(self, text: str):
        result = self.arabic_text.phonemize(text, diacritized=True)
        return (np.array([result.ids], dtype=np.int64),
                np.array([result.stress], dtype=np.int64))

    def synthesize(self, text: str, speed: float = 0.95, pitch_range: float = 1.35):
        """Vocalised Arabic text -> float32 waveform at the model sample rate."""
        tokens, stress = self.phonemize(text)
        feeds = {
            "tokens": tokens,
            "stress": stress,
            "voice": np.array([0], dtype=np.int64),  # F2, the single Arabic voice
            "speed": np.array([speed], dtype=np.float32),
            "pitch_shift": np.array([0.0], dtype=np.float32),
            "pitch_range": np.array([pitch_range], dtype=np.float32),
        }
        with self._lock:
            audio, _durations = self.session.run(None, feeds)
        # ``durations`` holds one frame count per token, not the sample count.
        # The browser demo plays the audio tensor as-is; do the same here.
        wave = np.asarray(audio[0], dtype=np.float32)
        peak = float(np.abs(wave).max()) if wave.size else 0.0
        if peak > 0.99:
            wave = wave * (0.99 / peak)
        return wave

    def wav_bytes(self, wave: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        # 16-bit PCM (format tag 1) plays everywhere. IEEE float WAV (tag 3)
        # makes some browsers/players report a wrong or truncated duration.
        sf.write(buffer, wave, self.config["sample_rate"], format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def check_voice(self, voice: str):
        if voice not in self.config["voices"]:
            raise HTTPException(400, f"unknown voice {voice!r}; available: {self.config['voices']}")


_engine = None
_answer_cache: dict[str, tuple[str, bytes]] = {}
_stream_cache: dict[str, tuple[list[tuple[str, bytes]], str]] = {}


def _answer_key(body) -> str:
    """One cache identity for both endpoints (model, question, options, history)."""
    return hashlib.sha256(
        f"{get_engine().model}|{body.question.strip()}|{body.max_chars}|{body.voice}|"
        f"{body.speed}|{body.pitch_range}|"
        f"{json.dumps(body.history, ensure_ascii=False, sort_keys=True)}".encode("utf-8")
    ).hexdigest()


def get_engine() -> VoiceEngine:
    global _engine
    if _engine is None:
        _engine = VoiceEngine()
    return _engine


app = FastAPI(title="tiny-voca Arabic voice answers", version="1.0.0")


class TtsIn(BaseModel):
    text: str
    voice: str = "F2"
    speed: float = Field(0.95, ge=0.25, le=4)
    pitch_range: float = Field(1.35, ge=0, le=3)
    already_vocalized: bool = True


class AnswerIn(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list, max_length=12)
    voice: str = "F2"
    speed: float = Field(0.95, ge=0.25, le=4)
    pitch_range: float = Field(1.35, ge=0, le=3)
    max_chars: int = Field(300, ge=40, le=600)


class StreamAnswerIn(AnswerIn):
    """Streaming variant: how many characters per synthesized audio chunk."""
    chunk_chars: int = Field(220, ge=40, le=600)


@app.get("/app", response_class=FileResponse, include_in_schema=False)
@app.get("/demo", response_class=FileResponse, include_in_schema=False)
def demo():
    """Ready-to-use RTL client: a question in writing, the answer in voice."""
    return FileResponse(APP_PAGE, media_type="text/html; charset=utf-8")


@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "language": engine.config["language"],
        "frontend": engine.config["frontend"],
        "voices": engine.config["voices"],
        "sample_rate": engine.config["sample_rate"],
        "step": engine.config["step"],
        "attribution": ATTRIBUTION,
    }


@app.post("/api/tts")
def tts(body: TtsIn):
    engine = get_engine()
    engine.check_voice(body.voice)
    text = body.text if body.already_vocalized else vocalize(body.text)
    text = sanitize(text)
    if not text:
        raise HTTPException(400, "nothing to pronounce after cleanup")
    try:
        wave = engine.synthesize(text, body.speed, body.pitch_range)
    except engine.NothingToPronounce as error:
        raise HTTPException(
            400, "nothing to pronounce: send Arabic text (vocalised text is more accurate)"
        ) from error
    return Response(engine.wav_bytes(wave), media_type="audio/wav")


@app.post("/api/voice/answer")
def voice_answer(body: AnswerIn):
    """Question in text, voice message out. The answer text is returned in a
    percent-encoded header so the client can display it next to the audio."""
    engine = get_engine()
    engine.check_voice(body.voice)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    key = _answer_key(body)
    cached = _answer_cache.get(key)
    if cached is None:
        try:
            answer = sanitize(cut_at_sentence(
                generate_answer(question, history=body.history), body.max_chars
            ))
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        except httpx.TimeoutException as error:
            raise HTTPException(
                504, "the language model request exceeded the timeout"
            ) from error
        except Exception as error:
            raise HTTPException(502, f"answer generation failed: {error}") from error
        if not answer:
            raise HTTPException(502, "the answer contained nothing pronounceable")
        try:
            wave = engine.synthesize(vocalize(answer), body.speed, body.pitch_range)
        except engine.NothingToPronounce as error:
            raise HTTPException(502, "the answer contained nothing pronounceable") from error
        cached = (answer, engine.wav_bytes(wave))
        _answer_cache[key] = cached
    answer, audio = cached
    return Response(
        audio,
        media_type="audio/wav",
        headers={"X-Answer-Text": quote(answer, safe="")},
    )


def _ndjson(event: dict) -> str:
    """One JSON object per line; Arabic stays readable in the wire format."""
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/voice/answer/stream")
def voice_answer_stream(body: StreamAnswerIn):
    """Progressive variant: emit one 16-bit PCM WAV per finished sentence.

    The LLM still generates the full answer (and ``qwen2.5:14b`` is unchanged),
    but each sentence is diacritized and synthesized as soon as it arrives.
    The client reads newline-delimited JSON:

        {"part": 1, "text": "...", "audio": "<base64 WAV>"}
        ...
        {"done": true, "text": "<full answer>", "parts": 2}

    Failures after the response has started are reported as
    ``{"error": "...", "status": 503|504|502}`` events. The original
    ``/api/voice/answer`` endpoint is untouched.
    """
    engine = get_engine()
    engine.check_voice(body.voice)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    key = _answer_key(body) + f"|chunk:{body.chunk_chars}"
    cached = _stream_cache.get(key)

    def events():
        if cached is not None:
            segments, full = cached
            for part, (text, audio) in enumerate(segments, 1):
                yield _ndjson({
                    "part": part,
                    "text": text,
                    "audio": base64.b64encode(audio).decode("ascii"),
                })
            yield _ndjson({"done": True, "text": full, "parts": len(segments)})
            return
        try:
            deltas = iter_answer_deltas(question, history=body.history)
            segments: list[tuple[str, bytes]] = []
            full = ""
            for segment in iter_sentences(deltas, max_chars=body.chunk_chars):
                text = sanitize(cut_at_sentence(segment, body.max_chars))
                if not text:
                    continue
                full = (full + " " + text).strip()
                wave = engine.synthesize(vocalize(text), body.speed, body.pitch_range)
                audio = engine.wav_bytes(wave)
                segments.append((text, audio))
                yield _ndjson({
                    "part": len(segments),
                    "text": text,
                    "audio": base64.b64encode(audio).decode("ascii"),
                })
            if not segments:
                yield _ndjson({
                    "error": "the answer contained nothing pronounceable", "status": 502
                })
                return
            _stream_cache[key] = (segments, full)
            yield _ndjson({"done": True, "text": full, "parts": len(segments)})
        except RuntimeError as error:
            yield _ndjson({"error": str(error), "status": 503})
        except httpx.TimeoutException:
            yield _ndjson({
                "error": "the language model request exceeded the timeout", "status": 504
            })
        except engine.NothingToPronounce:
            yield _ndjson({
                "error": "the answer contained nothing pronounceable", "status": 502
            })
        except Exception as error:
            yield _ndjson({"error": f"answer generation failed: {error}", "status": 502})

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
