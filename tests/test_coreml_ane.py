import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tiny_voca.config import Config
from tiny_voca.coreml import ane
from tiny_voca.model import TinyVoca
from tiny_voca.text import PHONE_IDS, SYMBOL_TO_ID, VOWELS

try:
    import coremltools as ct
except ImportError:
    ct = None


def random_model():
    torch.manual_seed(0)
    model = TinyVoca(Config()).eval()
    torch.nn.init.zeros_(model.duration.out.weight)          # a few frames per token, some pauses present
    model.duration.out.bias.data[:] = torch.tensor([1.5, 0.3])
    for head in (model.vocoder.harmonic, model.vocoder.noise, model.prosody_head):
        torch.nn.init.normal_(head.weight, std=0.05)
    return model


def utterance(length=21):
    ids = sorted(PHONE_IDS) + [SYMBOL_TO_ID[v] for v in VOWELS] + [SYMBOL_TO_ID["<sp>"], SYMBOL_TO_ID["."]]
    tokens = np.array([ids[(7 * i) % len(ids)] for i in range(length)])
    return tokens, np.arange(length) % 3


def close(test, value, reference, tolerance):
    test.assertEqual(value.shape, reference.shape)
    test.assertLess(float(np.abs(value - reference).max()), tolerance * max(1.0, float(np.abs(reference).max())))


class RewriteTest(unittest.TestCase):
    """The ANE-layout modules compute the original model in float32 (no coremltools needed)."""

    @classmethod
    def setUpClass(cls):
        cls.model = random_model()

    def run_pair(self, voice, pitch_range, query_chunk=None, size=None):
        m = self.model
        tokens, stress = utterance()
        n = len(tokens)
        ref = ane._reference(m, tokens, stress, voice, pitch_range=pitch_range)
        encoder, decoder = ane.Encoder(m, query_chunk).eval(), ane.Decoder(m, query_chunk).eval()
        with torch.no_grad():
            hidden, duration = encoder(*ane._torch_inputs(ane.encoder_inputs(tokens, stress, voice, 32)))
        close(self, hidden[0, :, 0, :n].numpy(), ref["hidden"], 1e-5)
        close(self, duration[0, :, :n].numpy(), ref["duration"], 1e-5)
        self.assertTrue(float(hidden[..., n:].abs().max()) == 0.0)
        self.assertTrue(np.array_equal(ane._durations(duration[0, :, :n].numpy(), tokens, m), ref["durations"]))
        total = int(ref["durations"].sum())
        inputs = ane.decoder_inputs(ref["hidden"], ref["durations"], tokens, voice, m, size=size, pitch_range=pitch_range)
        with torch.no_grad():
            pitch, harmonic, noise = decoder(*ane._torch_inputs(inputs))
        close(self, pitch[0, :total].numpy(), ref["pitch"], 1e-5)
        close(self, harmonic[0, :, :total].numpy(), ref["harmonic"], 1e-5)
        close(self, noise[0, :, :total].numpy(), ref["noise_params"], 1e-5)

    def test_matches_original(self):
        for voice, pitch_range in ((0, 1.0), (1, 1.4), (1, 0.0)):
            with self.subTest(voice=voice, pitch_range=pitch_range):
                self.run_pair(voice, pitch_range)

    def test_query_chunks_and_padding(self):
        self.run_pair(1, 1.2, query_chunk=48, size=200)

    def test_expansion_mirrors_model(self):
        m = self.model
        tokens, stress = utterance()
        hidden = torch.randn(1, len(tokens), m.config.channels)
        durations = torch.tensor([[(3 * i) % 5 for i in range(len(tokens))]])
        mask = torch.ones_like(durations, dtype=torch.bool)
        expanded, frame_mask, lengths = m.expand(hidden, durations, mask, torch.tensor([0]))
        frames, position, keep, vowel = ane._expand(hidden[0].T.numpy(), durations[0].numpy(), tokens, m, int(lengths) + 7)
        rebuilt = torch.from_numpy(frames[:, :int(lengths)]).T[None]
        rebuilt = rebuilt + m.frame_position(torch.from_numpy(position[:, :int(lengths)]).T[None]) + m.frame_voices(torch.tensor([0]))[:, None]
        close(self, rebuilt.detach().numpy(), expanded.detach().numpy(), 1e-6)
        self.assertEqual(keep.sum(), int(lengths))
        self.assertFalse(position[:, int(lengths):].any() or frames[:, int(lengths):].any() or vowel[int(lengths):].any())

    def test_bucket(self):
        self.assertEqual(ane.bucket(33, ane.ENCODER_BUCKETS), 64)
        self.assertEqual(ane.bucket(32, ane.ENCODER_BUCKETS), 32)
        with self.assertRaises(ValueError):
            ane.bucket(10 ** 6, ane.DECODER_BUCKETS)


@unittest.skipIf(ct is None or sys.platform != "darwin", "coremltools is not installed (or not on macOS)")
class ConversionTest(unittest.TestCase):
    """Two-bucket packages: functions, I/O names and dtypes, fp16 outputs close to PyTorch.
    The decoder's 512 bucket uses chunked attention queries (QUERY_CHUNK 256)."""

    @classmethod
    def setUpClass(cls):
        cls.model = random_model()
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.encoder = ane.convert_encoder(cls.model, root / "Encoder.mlpackage", (32, 64))
        cls.decoder = ane.convert_decoder(cls.model, root / "Decoder.mlpackage", (128, 512))
        cls.encoder16 = ane.convert_encoder(cls.model, root / "Encoder16.mlpackage", (32,), id_dtype=np.float16)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_functions_and_interface(self):
        self.assertEqual(ane.functions(self.encoder), ["n32", "n64"])
        self.assertEqual(ane.functions(self.decoder), ["t128", "t512"])
        spec = ct.utils.load_spec(str(self.decoder))
        function = {f.name: f for f in spec.description.functions}["t512"]
        self.assertEqual([i.name for i in function.input], list(ane.DECODER_INPUTS))
        self.assertEqual([o.name for o in function.output], list(ane.DECODER_OUTPUTS))
        shapes = {i.name: list(i.type.multiArrayType.shape) for i in function.input}
        self.assertEqual(shapes["frames"], [1, 192, 1, 512])
        self.assertEqual(shapes["pitch_range"], [1])

    def test_fp16_close_to_pytorch(self):
        m = self.model
        tokens, stress = utterance()
        n = len(tokens)
        ref = ane._reference(m, tokens, stress, 1, pitch_range=1.2)
        encoder = ct.models.MLModel(str(self.encoder), compute_units=ct.ComputeUnit.CPU_ONLY, function_name="n32")
        inputs = ane.encoder_inputs(tokens, stress, 1, 32, np.float16)
        out = ane.predict(encoder, inputs)
        self.assertEqual(out["hidden"].shape, (1, 192, 1, 32))
        self.assertEqual(out["duration"].shape, (1, 2, 32))
        close(self, out["hidden"][0, :, 0, :n].astype(np.float32), ref["hidden"], 2e-2)
        close(self, out["duration"][0, :, :n].astype(np.float32), ref["duration"], 2e-2)
        encoder16 = ct.models.MLModel(str(self.encoder16), compute_units=ct.ComputeUnit.CPU_ONLY, function_name="n32")
        out16 = ane.predict(encoder16, ane.encoder_inputs(tokens, stress, 1, 32, np.float16, np.float16))
        close(self, out16["hidden"][0, :, 0, :n].astype(np.float32), ref["hidden"], 2e-2)
        total = int(ref["durations"].sum())
        for size in (128, 512):
            decoder = ct.models.MLModel(str(self.decoder), compute_units=ct.ComputeUnit.CPU_ONLY, function_name=f"t{size}")
            out = ane.predict(decoder, ane.decoder_inputs(ref["hidden"], ref["durations"], tokens, 1, m, size, 1.2, np.float16))
            with self.subTest(size=size):
                close(self, out["pitch"][0, :total].astype(np.float32), ref["pitch"], 2e-2)
                close(self, out["harmonic"][0, :, :total].astype(np.float32), ref["harmonic"], 2e-2)
                close(self, out["noise_params"][0, :, :total].astype(np.float32), ref["noise_params"], 2e-2)

    def test_compute_plan_on_neural_engine(self):
        for path in (self.encoder, self.decoder):
            for name, row in ane.compute_plan_report(path).items():
                with self.subTest(function=name):
                    self.assertGreater(row["ne_share"], 0.9)
                    self.assertGreater(row["ne_cost_share"], 0.9)
        # With float16 ids nothing is left on the CPU (the int32 casts cannot run on the ANE).
        self.assertEqual(ane.compute_plan_report(self.encoder16)["n32"]["ne_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
