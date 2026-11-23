"""Answer generation through an OpenAI-compatible chat endpoint.

The endpoint is configurable so the same code works with a hosted API or a
local server (Ollama, vLLM, llama.cpp...). No key is ever returned to the
client. The prompt asks for short, unvocalised Modern Standard Arabic: the
diacritics are added afterwards by ``server.diacritize``.

Two paths share the same settings and payload:

* ``generate_answer``   - one blocking request, used by ``/api/voice/answer``;
* ``iter_answer_deltas`` - an SSE stream, used by ``/api/voice/answer/stream``.

The streaming path lets the service synthesize the first sentence while the
LLM is still writing the rest. The model itself is unchanged (``qwen2.5:14b``).
"""

from __future__ import annotations

import json
import os
import re

import httpx

SYSTEM_PROMPT = (
    "Answer in Modern Standard Arabic. Use one or two short sentences, at most "
    "40 words. Plain text only: no diacritics, no Latin letters, no emoji, no "
    "Markdown, no quotation marks. Do not mention that you are an AI."
)

# Arabic letters and marks (including diacritics), Arabic digits, Arabic and
# common punctuation, spaces. Everything else (Latin text, emoji, URLs) is
# replaced by a space so the G2P never sees a symbol it cannot pronounce.
_ALLOWED = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
    r"0-9\u0660-\u0669\s.,!؟،؛:\-]"
)
_SPACES = re.compile(r"\s+")

# A chunk ends after one of these marks, but only once the sentence is long
# enough: a stray "." in an abbreviation must not produce a one-word audio.
_SENTENCE_MARKS = "?!.؟؛;\n"
_MIN_CHUNK_CHARS = 12


def sanitize(text: str) -> str:
    """Remove characters the Arabic front end cannot pronounce."""
    clean = "".join(char if _ALLOWED.fullmatch(char) else " " for char in text)
    return _SPACES.sub(" ", clean).strip()


def cut_at_sentence(text: str, max_chars: int = 300) -> str:
    """Trim a reply to ``max_chars``, preferably after a sentence mark."""
    text = _SPACES.sub(" ", text).strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for mark in ("؟", ".", "!", "؛"):
        position = window.rfind(mark)
        if position >= max_chars // 2:
            return window[: position + 1].strip()
    position = window.rfind(" ")
    return (window[:position] if position > 0 else window).strip()


def _messages(question: str, history=None) -> list[dict]:
    """System prompt + the last six turns + the new question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def _request(question: str, history=None, *, stream: bool = False):
    """Return ``(url, headers, payload, timeout)`` for either LLM path.

    Kept in one place so the blocking and streaming endpoints cannot drift:
    same model, prompt, history window, keep-alive and credentials.
    """
    url = os.getenv("TINY_VOCA_LLM_URL", "https://api.openai.com/v1/chat/completions")
    key = os.getenv("TINY_VOCA_LLM_KEY")
    model = os.getenv("TINY_VOCA_LLM_MODEL", "gpt-4o-mini")
    timeout = float(os.getenv("TINY_VOCA_LLM_TIMEOUT", "60"))
    if not key:
        raise RuntimeError(
            "TINY_VOCA_LLM_KEY is not set; for a local server set any value "
            "in it and point TINY_VOCA_LLM_URL at its /v1/chat/completions endpoint"
        )
    payload = {
        "model": model,
        "messages": _messages(question, history),
        "temperature": 0.3,
        "max_tokens": 160,
    }
    if stream:
        payload["stream"] = True
    # Ollama only: keep the model in RAM between questions. The first answer
    # then costs ~2 s instead of ~27 s on this 14B CPU setup. Left unset for
    # hosted APIs, which do not accept this parameter.
    keep_alive = os.getenv("TINY_VOCA_LLM_KEEP_ALIVE")
    if keep_alive:
        # Ollama accepts a Go duration ("5m", "-1m") or a number (-1 = forever).
        # Environment variables are strings, so "-1" must be sent as a number:
        # as a plain string Ollama tries to parse a duration and rejects it.
        try:
            payload["keep_alive"] = int(keep_alive)
        except ValueError:
            payload["keep_alive"] = keep_alive
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    return url, headers, payload, timeout


def generate_answer(question: str, *, client=None, history=None) -> str:
    """Generate a short Arabic reply, optionally aware of the conversation.

    ``history`` is a list of ``{"role": "user"|"assistant", "content": str}``
    turns. Only the last six turns are kept, so the prompt stays small.
    ``client`` is an object with a ``post(url, json=, headers=, timeout=)``
    method; it is only used by the tests. In production the module uses httpx.
    """
    url, headers, payload, timeout = _request(question, history)
    if client is None:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    else:
        response = client.post(url, json=payload, headers=headers, timeout=timeout)
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def iter_answer_deltas(question: str, *, client=None, history=None):
    """Yield text fragments from an OpenAI-compatible SSE chat stream.

    Ollama sends blank-line-separated ``data: {...}`` events whose
    ``choices[0].delta.content`` holds a few characters/tokens. Heartbeat
    comments (``: ping``), non-content chunks, ``[DONE]`` and `event:` lines are
    ignored. ``client`` only exists for the tests; production uses httpx.
    """
    url, headers, payload, timeout = _request(question, history, stream=True)
    if client is None:
        context = httpx.stream("POST", url, json=payload, headers=headers, timeout=timeout)
    else:
        context = client.stream("POST", url, json=payload, headers=headers, timeout=timeout)
    with context as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                yield str(content)


def _next_boundary(text: str, max_chars: int):
    """Index just after the first good chunk boundary in ``text``, or None."""
    for position, char in enumerate(text):
        if char in _SENTENCE_MARKS and len(text[: position + 1].strip()) >= _MIN_CHUNK_CHARS:
            return position + 1
    if len(text) >= max_chars:
        space = text.rfind(" ", 0, max_chars)
        if space >= _MIN_CHUNK_CHARS:
            return space
        return max_chars                       # one very long word: cut anyway
    return None


def iter_sentences(deltas, max_chars: int = 220):
    """Group LLM token deltas into speakable chunks.

    Flush after an Arabic/Latin sentence mark, or at a word boundary when the
    chunk reaches ``max_chars``. Without this, a 30-word answer would keep the
    user waiting ~19 s before the first sound; chunks start sounding as soon as
    the first sentence is complete. The tail is flushed at the end of the
    stream, so no text is ever lost.
    """
    buffer = ""
    for delta in deltas:
        if not delta:
            continue
        buffer += str(delta)
        while True:
            cut = _next_boundary(buffer, max_chars)
            if cut is None:
                break
            segment, buffer = buffer[:cut], buffer[cut:]
            if segment.strip():
                yield segment.strip()
            buffer = buffer.lstrip()
    if buffer.strip():
        yield buffer.strip()
