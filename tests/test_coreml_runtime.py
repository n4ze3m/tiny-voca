"""The Core ML pipeline (runtime and command) against the PyTorch model.

A small random model with small buckets keeps the conversion short. Float16 in
the Encoder and Decoder moves F0 by a few cents, and the pulse train
integrates F0, so the waveform drifts in phase: the audio is compared as a
log-mel spectrum, and sample by sample only with PyTorch's F0.
"""

import contextlib
import dataclasses
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from helpers import small_config
from test_infer import phonemize
from tiny_voca.infer import finished, sentences
from tiny_voca.losses import SpectralLoss
from tiny_voca.model import TinyVoca
from tiny_voca.text import NothingToPronounce

try:
    import coremltools  # noqa: F401
    from tiny_voca.coreml import ane, runtime, vocoder as V
    from tiny_voca.coreml import __main__ as command
except ImportError:
    runtime = None

CPU = {"encoder": "CPU_ONLY", "decoder": "CPU_ONLY", "vocoder": "CPU_ONLY"}


def random_model():
    torch.manual_seed(3)
    model = TinyVoca(dataclasses.replace(small_config(), max_frames=400)).eval()
    with torch.no_grad():
        for head in (model.vocoder.harmonic, model.vocoder.noise, model.prosody_head):
            head.weight.normal_(std=0.05)
        model.duration.out.weight.normal_(std=0.02)
        model.duration.out.bias.copy_(torch.tensor([math.log1p(3), 0.2]))
    return model


@unittest.skipIf(runtime is None or sys.platform != "darwin", "coremltools is not installed (or not on macOS)")
class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = random_model()
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        checkpoint = root / "checkpoint.pt"
        torch.save({"step": 7}, checkpoint)
        runtime.export_models(cls.model, checkpoint, root / "models", (32, 64), (128, 256), log=lambda _: None)
        cls.root = root / "models"
        cls.voca = runtime.CoreMLVoca(cls.root, CPU)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_export_directory(self):
        for name in ("Encoder", "Decoder", "Vocoder"):
            self.assertTrue((self.root / f"{name}.mlpackage").is_dir())
            self.assertTrue((self.root / f"{name}.mlmodelc").is_dir())
        meta = self.voca.metadata
        self.assertEqual((meta["step"], meta["max_tokens"], meta["max_frames"]), (7, 64, 256))
        self.assertEqual(meta["models"]["decoder"]["functions"], ["t128", "t256"])
        self.assertEqual(meta["models"]["encoder"]["compute_units"], "CPU_AND_NE")
        self.assertEqual(ane.functions(self.root / "Encoder.mlpackage"), ["n32", "n64"])
        self.assertTrue(np.array_equal(self.voca.phone_tokens, V.PHONE_TOKENS))

    def test_matches_pytorch_with_its_durations_and_noise(self):
        m, c = self.model, self.model.config
        phones = phonemize("Hello there, this is a test of the pipeline!")
        tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
        for voice in (0, 1):
            with self.subTest(voice=voice):
                reference, lengths, details = m.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]), pitch_range=1.3)
                reference = reference[0, :int(lengths[0])].numpy()
                durations = details["durations"][0].numpy()
                noise = torch.randn(len(reference), generator=torch.Generator().manual_seed(0)).numpy()
                audio, ours = self.voca.synthesize(phones.ids, phones.stress, voice, pitch_range=1.3, durations=durations, noise=noise)
                self.assertEqual(audio.shape, reference.shape)
                self.assertEqual(audio.dtype, np.float32)
                f0 = m.frequency(details["prosody"][0, :, 0]).numpy()
                self.assertLess(float(np.abs(1200 * np.log2(ours["f0"] / f0)).max()), 10)          # cents
                mel = SpectralLoss(c.sample_rate)
                distance = float((mel.log_mel(torch.from_numpy(audio)[None]) - mel.log_mel(torch.from_numpy(reference)[None])).abs().mean())
                self.assertLess(distance, 0.1)
                # With PyTorch's F0 the waveform itself matches.
                inputs = V.vocoder_inputs(ours["harmonic"], ours["noise_params"], f0, noise, config=c)
                same_f0 = ane.predict(self.voca._model("vocoder"), inputs)["audio"][0]
                error = np.sum((same_f0.astype(np.float64) - reference) ** 2) / np.sum(reference.astype(np.float64) ** 2)
                self.assertGreater(-10 * math.log10(error), 30)                                   # dB
                free, free_details = self.voca.synthesize(phones.ids, phones.stress, voice, pitch_range=1.3)
                self.assertGreater(float((free_details["durations"] == durations).mean()), 0.9)
                self.assertTrue(np.isfinite(free).all())

    def test_settings_are_checked(self):
        ids = phonemize("Hi.")
        for settings in ({"speed": 0}, {"pitch_range": 4}, {"pitch_shift": math.inf}):
            with self.assertRaises(ValueError):
                self.voca.synthesize(ids.ids, ids.stress, "F2", **settings)
        with self.assertRaises(ValueError):
            self.voca.synthesize(list(range(5, 70)), [0] * 65, "F2")          # more tokens than the largest bucket

    def test_seed_and_voice_names(self):
        ids = phonemize("Hi there.")
        a, _ = self.voca.synthesize(ids.ids, ids.stress, "M2", seed=4)
        b, _ = self.voca.synthesize(ids.ids, ids.stress, 1, seed=4)
        c, _ = self.voca.synthesize(ids.ids, ids.stress, 1, seed=5)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    @patch.object(runtime, "phonemize", phonemize)
    def test_speak_text_joins_sentences(self):
        text = "The first one is here. The second follows"
        wave, spoken = self.voca.speak_text(text, "F2", peak=None)
        self.assertEqual([len(p) for p in spoken], [len(phonemize(s)) for s in sentences(finished(text))])
        parts = [self.voca.synthesize(p.ids, p.stress, "F2", seed=i)[0] for i, p in enumerate(spoken)]
        self.assertTrue(np.array_equal(wave, np.concatenate(parts)))
        limited, _ = self.voca.speak_text(text, "F2")
        self.assertLessEqual(float(np.abs(limited).max()), 0.99 + 1e-6)

    @patch.object(runtime, "phonemize", phonemize)
    def test_long_sentences_are_cut(self):
        # 20 words of 3 phones: more tokens than the largest encoder bucket (64) allows.
        words = " ".join(["word"] * 10)
        _, spoken = self.voca.speak_text(f"{words}, {words}.", "F2", peak=None)
        self.assertGreater(len(spoken), 1)
        self.assertTrue(all(len(p) <= self.voca.max_tokens - 12 for p in spoken))
        # Slow speech: too many frames (UtteranceTooLong) for the largest decoder bucket, so cut again.
        wave, slow = self.voca.speak_text("one two three four five six seven eight nine ten.", "F2", speed=0.25, peak=None)
        self.assertGreater(len(slow), 1)
        self.assertLessEqual(len(wave), len(slow) * self.voca.max_frames * self.model.config.hop)

    @patch.object(runtime, "phonemize", phonemize)
    def test_nothing_to_read(self):
        with self.assertRaises(NothingToPronounce):
            self.voca.speak_text("XD", "F2")

    @patch.object(runtime, "phonemize", phonemize)
    def test_speak_command(self):
        import soundfile as sf
        output = Path(self.directory.name) / "out" / "speech.wav"
        out = io.StringIO()
        with patch.dict(runtime.COMPUTE_UNITS, CPU), contextlib.redirect_stdout(out):
            command.main(["--models", str(self.root), "--text", "One here. Two there!", "--voice", "M2", "--output", str(output)])
        wave, rate = sf.read(output, dtype="float32")
        self.assertEqual(rate, self.model.config.sample_rate)
        expected, _ = self.voca.speak_text("One here. Two there!", "M2")
        self.assertTrue(np.allclose(wave, expected, atol=1e-6))
        self.assertIn("in 2 utterances", out.getvalue())

    def test_cached_models(self):
        checkpoint = Path(self.directory.name) / "checkpoint.pt"
        cache = Path(self.directory.name) / "cache"
        loads = []

        def load(path):
            loads.append(path)
            return self.model

        with patch.object(runtime, "_load_checkpoint", load):
            first = runtime.cached_models(checkpoint, cache, log=lambda _: None, encoder_buckets=(32,), decoder_buckets=(128,))
            again = runtime.cached_models(checkpoint, cache, log=lambda _: None, encoder_buckets=(32,), decoder_buckets=(128,))
        self.assertEqual(first, again)
        self.assertEqual(len(loads), 1)                      # converted once, then read from the cache
        self.assertTrue((first / runtime.METADATA).exists() and (first / "Decoder.mlmodelc").is_dir())
        self.assertEqual([p.name for p in cache.iterdir()], [first.name])   # no partial export is left
        self.assertTrue(first.name.endswith(f"-v{runtime.FORMAT_VERSION}"))

    @patch.object(runtime, "phonemize", phonemize)
    def test_speak_command_plays_without_output(self):
        import soundfile as sf
        played = []

        class Player:
            def __init__(self, arguments):
                played.append(sf.read(arguments[1], dtype="float32"))   # read before the temporary file is removed
                self.arguments = arguments

            def wait(self):
                return 0

        out = io.StringIO()
        with patch.dict(runtime.COMPUTE_UNITS, CPU), patch.object(command.subprocess, "Popen", Player), contextlib.redirect_stdout(out):
            command.main(["--models", str(self.root), "--text", "One here. Two there!", "--voice", "M2"])
        (wave, rate), = played
        self.assertEqual(rate, self.model.config.sample_rate)
        expected, _ = self.voca.speak_text("One here. Two there!", "M2")
        self.assertTrue(np.allclose(wave, expected, atol=1e-6))
        self.assertIn("Playing", out.getvalue())


if __name__ == "__main__":
    unittest.main()
