import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tiny_voca.config import Config
from tiny_voca.coreml import vocoder as V
from tiny_voca.model import TinyVoca, UtteranceTooLong, pulse_train
from tiny_voca.text import PHONE_IDS, SYMBOL_TO_ID, VOWELS

HAS_COREMLTOOLS = importlib.util.find_spec("coremltools") is not None


def model_with_heads():
    """Full-size model (the vocoder contract is 98 + 36 channels) whose heads
    give varied durations and spectra instead of the flat initial ones."""
    torch.manual_seed(0)
    model = TinyVoca(Config()).eval()
    torch.nn.init.normal_(model.duration.out.weight, std=0.05)
    model.duration.out.bias.data[:] = torch.tensor([1.5, 0.0])
    torch.nn.init.normal_(model.vocoder.harmonic.weight, std=0.05)
    torch.nn.init.normal_(model.vocoder.noise.weight, std=0.05)
    return model


def utterance(length=40):
    """Phonemes with word boundaries and punctuation, so pauses can vanish."""
    pool = sorted(PHONE_IDS)[:20] + [SYMBOL_TO_ID[v] for v in VOWELS[:6]] + [SYMBOL_TO_ID["<sp>"], SYMBOL_TO_ID[","]]
    tokens = torch.tensor([[pool[(5 * i + 3) % len(pool)] for i in range(length)]])
    return tokens, torch.arange(length)[None] % 3


def head_outputs(model, hidden):
    with torch.no_grad():
        return model.vocoder.harmonic(hidden)[0].T.numpy(), model.vocoder.noise(hidden)[0].T.numpy()


class HostStepsTest(unittest.TestCase):
    def test_token_tables_match_model_buffers(self):
        model = TinyVoca(Config())
        self.assertTrue(np.array_equal(V.PHONE_TOKENS, model.phone_tokens.numpy()))
        self.assertTrue(np.array_equal(V.VOWEL_TOKENS, model.vowel_tokens.numpy()))

    def test_pulse_train_matches_torch(self):
        c = Config()
        generator = torch.Generator().manual_seed(1)
        for frames in (1, 7, 400):
            f0 = (c.f0_min + (c.f0_max - c.f0_min) * torch.rand(1, frames, generator=generator)).float()
            reference = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit)[0].numpy()
            ours = V.pulse_train(f0[0].numpy(), c.sample_rate, c.hop, c.harmonic_limit)
            self.assertEqual(ours.dtype, np.float32)
            self.assertEqual(ours.shape, reference.shape)
            self.assertLess(float(np.abs(ours - reference).max()), 1e-5)

    def test_durations_match_predict_durations(self):
        model = model_with_heads()
        tokens, stress = utterance()
        mask = torch.ones_like(tokens, dtype=torch.bool)
        with torch.no_grad():
            hidden = model.encoder(model.tokens(tokens) + model.stress(stress) + model.voices(torch.tensor([1]))[:, None], mask)
            raw = model.duration(hidden, mask).float()[0].T.numpy()
            for speed in (1.0, 1.3, 0.7):
                reference = model.predict_durations(hidden, mask, tokens, speed)[0].numpy()
                ours = V.durations(raw, tokens[0].numpy(), speed)
                self.assertTrue(np.array_equal(ours, reference), speed)
        self.assertTrue((reference == 0).any() and (reference > 1).any())   # pauses dropped, lengths varied

    def test_expand_matches_model(self):
        model = model_with_heads()
        tokens, _ = utterance(12)
        durations = torch.tensor([[3, 0, 5, 1, 9, 0, 4, 10, 2, 4, 0, 6]])
        hidden = torch.randn(1, 12, 192)
        voice = torch.tensor([0])
        with torch.no_grad():
            reference, _, lengths = model.expand(hidden, durations, torch.ones_like(durations, dtype=torch.bool), voice)
            frames, position, index = V.expand(hidden[0].T.numpy(), durations[0].numpy())
            self.assertEqual(frames.shape, (192, int(lengths[0])))
            ends = durations.cumsum(1)
            owner = torch.searchsorted(ends, torch.arange(int(lengths[0]))[None], right=True)[0].numpy()
            self.assertTrue(np.array_equal(index, owner))
            ours = torch.tensor(frames.T) + model.frame_position(torch.tensor(position.T)) + model.frame_voices(voice)
        self.assertLess(float((ours - reference[0]).abs().max()), 1e-5)
        self.assertEqual(position.dtype, np.float32)
        self.assertTrue(np.array_equal(V.vowel_mask(tokens[0].numpy(), index), model.vowel_tokens[tokens].gather(1, torch.tensor(index)[None])[0].numpy()))

    def test_expand_rejects_bad_lengths(self):
        with self.assertRaises(ValueError):
            V.expand(np.zeros((4, 3)), [0, 0, 0])
        with self.assertRaises(UtteranceTooLong):
            V.expand(np.zeros((4, 2)), [2000, 1001])

    def test_f0_from_pitch(self):
        model = TinyVoca(Config())
        pitch = torch.linspace(-4, 3, 50)
        reference = model.frequency(pitch[None] + 0.1)[0].numpy()
        self.assertLess(float(np.abs(V.f0_from_pitch(pitch.numpy(), 0.1, model.config) - reference).max()), 1e-3)

    def test_noise_is_seeded(self):
        a, b = V.noise(1000, 3), V.noise(1000, 3)
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, V.noise(1000, 4)))
        self.assertLess(abs(float(V.noise(100000, 0).std()) - 1), 0.02)


class VocoderModuleTest(unittest.TestCase):
    def check(self, run, tolerance):
        model = model_with_heads()
        c = model.config
        generator = torch.Generator().manual_seed(2)
        for frames in (1, 2, 37):
            hidden = torch.randn(1, frames, c.channels, generator=generator)
            f0 = c.f0_min + 300 * torch.rand(1, frames, generator=generator)
            noise = torch.randn(1, frames * c.hop, generator=generator)
            with torch.no_grad():
                reference = model.vocoder(hidden, f0, torch.tensor([frames]), noise=noise)[0].numpy()
            inputs = V.vocoder_inputs(*head_outputs(model, hidden), f0[0].numpy(), noise[0].numpy(), config=c)
            audio = run(model, inputs)
            self.assertEqual(audio.shape, (1, frames * c.hop))
            self.assertLess(float(np.abs(audio[0] - reference).max()), tolerance * float(np.abs(reference).max()), frames)

    def test_module_matches_source_filter(self):
        def run(model, inputs):
            with torch.no_grad():
                return V.vocoder_module(model)(*(torch.tensor(inputs[name]) for name in V.INPUTS)).numpy()
        self.check(run, 1e-5)

    @unittest.skipUnless(HAS_COREMLTOOLS, "coremltools is not installed")
    def test_coreml_matches_source_filter(self):
        models = {}

        def run(model, inputs):
            if "vocoder" not in models:
                directory = tempfile.TemporaryDirectory()
                self.addCleanup(directory.cleanup)
                models["vocoder"] = V.convert(model, Path(directory.name) / "Vocoder.mlpackage", compute_units="CPU_ONLY", max_frames=400)
            return models["vocoder"].predict(inputs)["audio"]
        self.check(run, 1e-4)


if __name__ == "__main__":
    unittest.main()
