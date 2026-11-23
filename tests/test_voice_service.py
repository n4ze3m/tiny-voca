"""Voice-answer service: answer cleanup, real ONNX synthesis and HTTP endpoints.

The model is loaded once from ``server/assets`` (the exact browser demo model).
The LLM is always replaced by a stand-in: no network and no API key are needed.
"""

import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
import wave as wave_file
from unittest.mock import Mock, patch
from urllib.parse import unquote

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import tts_service
from server.answer import cut_at_sentence, generate_answer, sanitize

ASSETS = ROOT / "server" / "assets"
VOCALIZED = "مَرْحَبًا بِالْعالَمِ."
PLAIN = "مرحبا بالعالم"
_ENGINE = None


def engine():
    """Load the real 21 MB ONNX model once for the whole suite."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = tts_service.VoiceEngine(ASSETS, threads=2)
    return _ENGINE


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}

    def raise_for_status(self):
        return None


class MockLlmTests(unittest.TestCase):
    def test_the_fake_llm_returns_a_valid_arabic_answer(self):
        from server.mock_llm import ANSWER, payload

        data = payload("كيف حالك؟")
        content = data["choices"][0]["message"]["content"]
        self.assertEqual(content, ANSWER)
        self.assertTrue(any("\u0600" <= char <= "\u06ff" for char in content))
        self.assertEqual(data["object"], "chat.completion")


class AnswerTextTests(unittest.TestCase):
    def test_sanitize_keeps_arabic_and_drops_latin_and_emoji(self):
        self.assertEqual(sanitize("مرحبا 😊 hello 42، كيف حالك؟"), "مرحبا 42، كيف حالك؟")
        self.assertEqual(sanitize("مرحبا"), "مرحبا")                     # unvocalised Arabic survives
        self.assertEqual(sanitize(VOCALIZED), VOCALIZED)                 # diacritics survive
        self.assertEqual(sanitize("😊 XD https://x.y"), ": .")   # punctuation survives, letters do not

    def test_cut_at_sentence_prefers_a_mark_in_the_second_half(self):
        text = "أولاً جملة طويلة عن الموضوع. ثم جملة أخرى يجب ألا تظهر في الإجابة."
        self.assertEqual(cut_at_sentence(text, 40), "أولاً جملة طويلة عن الموضوع.")
        self.assertEqual(cut_at_sentence("مرحبا", 10), "مرحبا")          # short text is untouched

    def test_cut_at_sentence_falls_back_to_a_word_boundary(self):
        text = "واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية"
        trimmed = cut_at_sentence(text, 20)
        self.assertLessEqual(len(trimmed), 20)
        self.assertTrue(trimmed)
        self.assertTrue(text.startswith(trimmed))

    def test_generate_answer_uses_the_configured_openai_compatible_endpoint(self):
        env = {
            "TINY_VOCA_LLM_URL": "http://localhost:11434/v1/chat/completions",
            "TINY_VOCA_LLM_KEY": "local-key",
            "TINY_VOCA_LLM_MODEL": "llama3",
            "TINY_VOCA_LLM_TIMEOUT": "5",
        }
        client = Mock(post=Mock(return_value=FakeResponse("  مرحبا بك  ")))
        with patch.dict(os.environ, env):
            answer = generate_answer("كيف حالك؟", client=client)
        self.assertEqual(answer, "مرحبا بك")
        args, kwargs = client.post.call_args
        self.assertEqual(args[0], env["TINY_VOCA_LLM_URL"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer local-key")
        self.assertEqual(kwargs["timeout"], 5.0)
        self.assertEqual(kwargs["json"]["model"], "llama3")
        self.assertEqual(kwargs["json"]["messages"][-1], {"role": "user", "content": "كيف حالك؟"})
        self.assertIn("Modern Standard Arabic", kwargs["json"]["messages"][0]["content"])

    def test_generate_answer_forwards_recent_conversation_turns(self):
        client = Mock(post=Mock(return_value=FakeResponse("نعم")))
        with patch.dict(os.environ, {"TINY_VOCA_LLM_KEY": "k"}):
            generate_answer("وهل تعرفني؟", client=client, history=[
                {"role": "user", "content": "اسمي خالد"},
                {"role": "assistant", "content": "مرحبا خالد"},
            ])
        messages = client.post.call_args.kwargs["json"]["messages"]
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "وهل تعرفني؟")
        self.assertEqual(messages[1]["content"], "اسمي خالد")

    def test_generate_answer_keeps_the_local_model_warm(self):
        client = Mock(post=Mock(return_value=FakeResponse("مرحبا")))
        env = {"TINY_VOCA_LLM_KEY": "local", "TINY_VOCA_LLM_KEEP_ALIVE": "-1"}
        with patch.dict(os.environ, env, clear=True):
            generate_answer("كيف حالك؟", client=client)
        kwargs = client.post.call_args.kwargs
        self.assertEqual(kwargs["json"]["keep_alive"], -1)   # number, not bare string
        self.assertEqual(kwargs["timeout"], 60.0)   # a cold 14B load takes ~27 s

    def test_generate_answer_accepts_a_keep_alive_duration(self):
        client = Mock(post=Mock(return_value=FakeResponse("مرحبا")))
        env = {"TINY_VOCA_LLM_KEY": "local", "TINY_VOCA_LLM_KEEP_ALIVE": "5m"}
        with patch.dict(os.environ, env, clear=True):
            generate_answer("كيف حالك؟", client=client)
        self.assertEqual(client.post.call_args.kwargs["json"]["keep_alive"], "5m")

    def test_generate_answer_omits_keep_alive_for_hosted_apis(self):
        client = Mock(post=Mock(return_value=FakeResponse("مرحبا")))
        with patch.dict(os.environ, {"TINY_VOCA_LLM_KEY": "sk-test"}, clear=True):
            generate_answer("كيف حالك؟", client=client)
        kwargs = client.post.call_args.kwargs
        self.assertNotIn("keep_alive", kwargs["json"])
        self.assertEqual(kwargs["timeout"], 60.0)

    def test_generate_answer_without_key_explains_how_to_set_it(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TINY_VOCA_LLM_KEY"):
                generate_answer("سؤال")


@unittest.skipUnless(importlib.util.find_spec("camel_tools"), "needs camel-tools")
class DiacritizationTests(unittest.TestCase):
    def test_camel_adds_diacritics_and_caches_the_result(self):
        from server.diacritize import vocalize

        plain = "مرحبا بالعالم"
        result = vocalize(plain)
        self.assertNotEqual(result, plain)
        self.assertTrue(any("\u064b" <= char <= "\u065f" for char in result), result)
        self.assertIs(vocalize(plain), result)                           # lru_cache hits

    def test_a_word_without_a_morphology_analysis_keeps_its_spelling(self):
        from server import diacritize

        class Analysis:
            def __init__(self, diac):
                self.analysis = {"diac": diac}

        class Item:
            def __init__(self, word, diac=None):
                self.word = word
                self.analyses = [] if diac is None else [Analysis(diac)]

        fake = Mock()
        fake.disambiguate.return_value = [Item("مارسي"), Item("الكبير", "الْكَبِير"),
                                          Item("كلمةجديدة")]
        with patch.object(diacritize, "_engine", return_value=fake):
            result = diacritize.vocalize("مارسي الكبير كلمةجديدة")
        self.assertEqual(result, "مارسي الْكَبِير كلمةجديدة")


class VoiceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.speech = engine()

    def test_assets_are_the_arabic_rules_model(self):
        config = self.speech.config
        self.assertEqual(config["language"], "ar")
        self.assertEqual(config["frontend"], "rules")
        self.assertEqual(config["voices"], ["F2"])
        self.assertEqual(config["sample_rate"], 24000)
        self.assertEqual(config["step"], 3000)
        self.assertEqual(config["data_source"]["license"], "CC BY 4.0")
        names = [item.name for item in self.speech.session.get_inputs()]
        self.assertEqual(names, config["inputs"])                        # same contract as worker.js
        self.assertEqual(self.speech.session.get_inputs()[0].shape[0], 1)

    def test_only_f2_is_available(self):
        self.assertIsNone(self.speech.check_voice("F2"))
        with self.assertRaises(Exception) as error:
            self.speech.check_voice("M2")
        self.assertEqual(getattr(error.exception, "status_code", None), 400)

    def test_real_synthesis_produces_a_wav_that_can_be_read(self):
        wave = self.speech.synthesize(VOCALIZED)
        self.assertIsInstance(wave, np.ndarray)
        self.assertGreater(len(wave), 2400)                              # more than 0.1 s
        self.assertLessEqual(float(np.abs(wave).max()), 0.99 + 1e-6)
        raw = self.speech.wav_bytes(wave)
        self.assertTrue(raw.startswith(b"RIFF"))
        samples, rate = sf.read(io.BytesIO(raw), dtype="float32")
        self.assertEqual(rate, 24000)
        self.assertEqual(len(samples), len(wave))
        np.testing.assert_allclose(samples, wave, atol=4e-5)             # 16-bit quantisation
        # Any player must read the file: float WAV (tag 3) is not universal.
        with wave_file.open(io.BytesIO(raw)) as readable:
            self.assertEqual(readable.getnframes(), len(wave))
            self.assertEqual(readable.getsampwidth(), 2)
            self.assertEqual(readable.getframerate(), 24000)

    def test_speed_and_pitch_range_change_the_output(self):
        slow = self.speech.synthesize(VOCALIZED, speed=0.7)
        fast = self.speech.synthesize(VOCALIZED, speed=1.3)
        self.assertGreater(len(slow), len(fast))
        self.assertFalse(np.array_equal(slow[:len(fast)], fast))


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.speech = engine()
        cls.client = TestClient(tts_service.app)

    def setUp(self):
        tts_service._engine = self.speech
        tts_service._answer_cache.clear()
        tts_service._stream_cache.clear()

    def wav(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("audio/wav", response.headers["content-type"])
        self.assertEqual(response.content[20:22], b"\x01\x00")            # PCM, not float
        samples, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        self.assertEqual(rate, 24000)
        self.assertGreater(len(samples), 0)
        return samples

    def test_demo_page_calls_the_voice_answer_endpoint(self):
        for path in ("/demo", "/app"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn('lang="ar" dir="rtl"', response.text)
            # The original endpoint and its text header are kept as a fallback.
            self.assertIn('fetch("/api/voice/answer"', response.text)
            self.assertIn('response.headers.get("X-Answer-Text")', response.text)
            self.assertIn('createElement("audio")', response.text)
            self.assertIn("history: history", response.text)

    def test_demo_page_reads_the_streaming_answer(self):
        html = self.client.get("/app").text
        self.assertIn('fetch("/api/voice/answer/stream"', html)
        self.assertIn("response.body.getReader()", html)
        self.assertIn('new TextDecoder("utf-8")', html)
        self.assertIn("event.part", html)
        self.assertIn("event.done", html)
        self.assertIn("event.error", html)
        self.assertIn("base64ToBlob", html)
        self.assertIn('id="stop"', html)
        self.assertIn("function regenerate()", html)

    def test_demo_page_shows_the_answer_as_audio_only(self):
        html = self.client.get("/app").text
        self.assertIn(".bot .text { display: none; }", html)
        self.assertIn("رسالة صوتية", html)
        # nothing was removed: the text header and the history are still used
        self.assertIn('response.headers.get("X-Answer-Text")', html)
        self.assertIn("history.push", html)

    def test_demo_page_has_a_wide_player_with_visible_progress(self):
        html = self.client.get("/app").text
        self.assertIn("width:min(100%, 560px)", html)   # no 300 px intrinsic audio width
        self.assertIn("height:54px", html)
        self.assertIn("timeupdate", html)               # elapsed / total time
        self.assertIn("formatTime(audio.currentTime)", html)
        self.assertIn("class=\"spinner\"", html)        # visible while the LLM works
        self.assertIn("@media (max-width:600px)", html)

    def test_health_reports_language_frontend_voice_and_attribution(self):
        data = self.client.get("/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual((data["language"], data["frontend"]), ("ar", "rules"))
        self.assertEqual(data["voices"], ["F2"])
        self.assertEqual(data["sample_rate"], 24000)
        self.assertIn("Halabi, Nawar (2016)", data["attribution"])

    def test_tts_accepts_vocalized_text(self):
        self.wav(self.client.post("/api/tts", json={"text": VOCALIZED}))

    def test_tts_can_vocalize_plain_text_on_request(self):
        with patch.object(tts_service, "vocalize", return_value=VOCALIZED) as vocalize:
            self.wav(self.client.post("/api/tts", json={"text": PLAIN, "already_vocalized": False}))
        vocalize.assert_called_once_with(PLAIN)

    def test_tts_rejects_unknown_voice_and_empty_text(self):
        self.assertEqual(self.client.post("/api/tts", json={"text": VOCALIZED, "voice": "M2"}).status_code, 400)
        self.assertEqual(self.client.post("/api/tts", json={"text": "😊 XD"}).status_code, 400)

    def test_rules_front_end_also_reads_unvocalized_arabic(self):
        response = self.client.post("/api/tts", json={"text": "مرحبا بالعالم"})
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(self.wav(response)), 0)

    def test_voice_answer_returns_audio_and_the_plain_answer(self):
        answer = "مرحبا بك، كيف يمكنني مساعدتك؟"
        with patch.object(tts_service, "generate_answer", return_value=answer) as generated, \
             patch.object(tts_service, "vocalize", return_value=VOCALIZED) as vocalize:
            response = self.client.post("/api/voice/answer", json={"question": "مرحبا"})
        self.wav(response)
        self.assertEqual(unquote(response.headers["X-Answer-Text"]), answer)
        generated.assert_called_once()
        self.assertEqual(generated.call_args.args, ("مرحبا",))
        self.assertEqual(generated.call_args.kwargs["history"], [])
        vocalize.assert_called_once_with(answer)

    def test_voice_answer_sends_the_conversation_to_the_llm(self):
        answer = "أنا بخير، شكرًا لك."
        turns = [{"role": "user", "content": "مرحبا"},
                 {"role": "assistant", "content": "أهلا بك"}]
        with patch.object(tts_service, "generate_answer", return_value=answer) as generated, \
             patch.object(tts_service, "vocalize", return_value=VOCALIZED):
            response = self.client.post("/api/voice/answer",
                                        json={"question": "كيف حالك؟", "history": turns})
        self.wav(response)
        self.assertEqual(generated.call_args.kwargs["history"], turns)

    def test_voice_answer_caches_identical_questions(self):
        answer = "نعم،那是 صحيح."                                     # Latin/Chinese removed by sanitize
        with patch.object(tts_service, "generate_answer", return_value=answer) as generated, \
             patch.object(tts_service, "vocalize", return_value=VOCALIZED):
            request = {"question": "هل تعمل؟"}
            first = self.client.post("/api/voice/answer", json=request)
            second = self.client.post("/api/voice/answer", json=request)
        self.assertEqual(first.content, second.content)
        self.assertEqual(generated.call_count, 1)
        self.assertEqual(unquote(first.headers["X-Answer-Text"]), "نعم، صحيح.")

    def test_voice_answer_reports_missing_llm_key_as_service_unavailable(self):
        with patch.object(tts_service, "generate_answer", side_effect=RuntimeError("no key")):
            response = self.client.post("/api/voice/answer", json={"question": "مرحبا"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("no key", response.json()["detail"])

    def test_voice_answer_reports_a_llm_timeout_as_gateway_timeout(self):
        import httpx

        with patch.object(tts_service, "generate_answer",
                          side_effect=httpx.TimeoutException("read timeout")):
            response = self.client.post("/api/voice/answer", json={"question": "مرحبا"})
        self.assertEqual(response.status_code, 504)
        self.assertIn("timeout", response.json()["detail"].lower())

    def test_voice_answer_rejects_an_empty_question(self):
        self.assertEqual(self.client.post("/api/voice/answer", json={"question": "  "}).status_code, 400)

    def stream_events(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]

    def test_stream_emits_a_pcm_wav_per_sentence_then_the_full_answer(self):
        deltas = ["عاصمة مصر هي القاهرة. ", "تقع على ضفاف النيل."]
        turns = [{"role": "user", "content": "اسمي خالد"}]
        with patch.object(tts_service, "iter_answer_deltas", return_value=iter(deltas)) as streamed, \
             patch.object(tts_service, "vocalize", return_value=VOCALIZED) as vocalize:
            response = self.client.post("/api/voice/answer/stream",
                                        json={"question": "ما عاصمة مصر؟", "history": turns})
        events = self.stream_events(response)
        self.assertEqual([event["part"] for event in events if "part" in event], [1, 2])
        self.assertEqual(events[-1]["done"], True)
        self.assertEqual(events[-1]["parts"], 2)
        self.assertEqual(events[-1]["text"], "عاصمة مصر هي القاهرة. تقع على ضفاف النيل.")
        self.assertEqual(vocalize.call_count, 2)
        self.assertEqual(streamed.call_args.args, ("ما عاصمة مصر؟",))
        self.assertEqual(streamed.call_args.kwargs["history"], turns)
        for event in events[:2]:
            self.assertTrue(any("\u0600" <= char <= "\u06ff" for char in event["text"]))
            raw = base64.b64decode(event["audio"])
            self.assertEqual(raw[:4], b"RIFF")
            self.assertEqual(raw[20:22], b"\x01\x00")                 # PCM, not float
            with wave_file.open(io.BytesIO(raw)) as readable:
                self.assertEqual(readable.getsampwidth(), 2)
                self.assertEqual(readable.getframerate(), 24000)
                self.assertGreater(readable.getnframes(), 0)

    def test_stream_replays_a_cached_answer_without_the_llm(self):
        deltas = ["مرحبا بك، كيف حالك؟"]
        request = {"question": "مرحبا"}
        with patch.object(tts_service, "iter_answer_deltas", return_value=iter(deltas)) as streamed, \
             patch.object(tts_service, "vocalize", return_value=VOCALIZED):
            first = self.stream_events(self.client.post("/api/voice/answer/stream", json=request))
            second = self.stream_events(self.client.post("/api/voice/answer/stream", json=request))
        self.assertEqual(streamed.call_count, 1)
        self.assertEqual(first, second)

    def test_stream_reports_a_missing_key_as_an_error_event(self):
        with patch.object(tts_service, "iter_answer_deltas",
                          side_effect=RuntimeError("TINY_VOCA_LLM_KEY is not set")):
            events = self.stream_events(
                self.client.post("/api/voice/answer/stream", json={"question": "مرحبا"})
            )
        self.assertEqual(len(events), 1)
        self.assertIn("TINY_VOCA_LLM_KEY", events[0]["error"])
        self.assertEqual(events[0]["status"], 503)
        self.assertNotIn("done", events[0])

    def test_stream_rejects_an_empty_question(self):
        self.assertEqual(self.client.post("/api/voice/answer/stream",
                                          json={"question": "  "}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
