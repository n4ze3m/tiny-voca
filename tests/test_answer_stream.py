"""Streaming LLM deltas and sentence chunking, without a network.

The two pieces tested here are the ones that make the answer audible early:
``iter_answer_deltas`` parses Ollama's SSE frames, and ``iter_sentences``
groups tokens into speakable pieces instead of waiting for the full reply.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.answer import iter_answer_deltas, iter_sentences  # noqa: E402

ENV = {
    "TINY_VOCA_LLM_URL": "http://127.0.0.1:11434/v1/chat/completions",
    "TINY_VOCA_LLM_KEY": "local",
    "TINY_VOCA_LLM_MODEL": "qwen2.5:14b",
    "TINY_VOCA_LLM_KEEP_ALIVE": "-1",
}


def sse(content: str) -> str:
    """One Ollama/OpenAI SSE data frame."""
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self.lines)


class FakeStreamClient:
    def __init__(self, lines):
        self.lines = lines
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeStreamResponse(self.lines)


class StreamDeltaTests(unittest.TestCase):
    def test_parses_sse_frames_and_ignores_heartbeats_and_done(self):
        lines = [
            ": ping", "", sse("عاصمة"), sse(" مصر"),
            "data: [DONE]", "event: close",
        ]
        client = FakeStreamClient(lines)
        with patch.dict(os.environ, ENV, clear=True):
            pieces = list(iter_answer_deltas("ما عاصمة مصر؟", client=client, history=[
                {"role": "user", "content": "اسمي خالد"},
            ]))
        self.assertEqual(pieces, ["عاصمة", " مصر"])
        method, url, kwargs = client.calls[0]
        self.assertEqual((method, url), ("POST", ENV["TINY_VOCA_LLM_URL"]))
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["model"], "qwen2.5:14b")
        self.assertEqual(kwargs["json"]["keep_alive"], -1)
        roles = [message["role"] for message in kwargs["json"]["messages"]]
        self.assertEqual(roles, ["system", "user", "user"])

    def test_skips_chunks_without_text(self):
        role_only = json.dumps({"choices": [{"delta": {"role": "assistant"}}]})
        empty = json.dumps({"choices": [{"delta": {"content": ""}}]})
        client = FakeStreamClient(["data: " + role_only, "data: " + empty, sse("مرحبا")])
        with patch.dict(os.environ, ENV, clear=True):
            pieces = list(iter_answer_deltas("مرحبا", client=client))
        self.assertEqual(pieces, ["مرحبا"])

    def test_missing_key_raises_before_opening_the_stream(self):
        client = FakeStreamClient([])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TINY_VOCA_LLM_KEY"):
                list(iter_answer_deltas("مرحبا", client=client))
        self.assertEqual(client.calls, [])


class SentenceChunkTests(unittest.TestCase):
    def test_flushes_after_an_arabic_sentence_mark(self):
        deltas = ["عاصمة مصر هي ", "القاهرة. ", "وباريس عاصمة فرنسا."]
        self.assertEqual(list(iter_sentences(deltas)),
                         ["عاصمة مصر هي القاهرة.", "وباريس عاصمة فرنسا."])

    def test_flushes_at_a_word_boundary_when_too_long(self):
        words = ["كلمة" + str(index) for index in range(20)]
        pieces = list(iter_sentences([" ".join(words)], max_chars=30))
        self.assertGreater(len(pieces), 1)
        self.assertEqual(" ".join(pieces), " ".join(words))   # nothing lost
        for piece in pieces:
            self.assertLessEqual(len(piece), 30)

    def test_flushes_the_tail_and_skips_empty_deltas(self):
        pieces = list(iter_sentences([None, "", "مرحبا", " بك"]))
        self.assertEqual(pieces, ["مرحبا بك"])

    def test_short_fragment_with_a_dot_is_not_flushed_early(self):
        pieces = list(iter_sentences(["3.14", " جملة طويلة جدا هنا"]))
        self.assertEqual(pieces, ["3.14 جملة طويلة جدا هنا"])   # no one-word audio


if __name__ == "__main__":
    unittest.main()
