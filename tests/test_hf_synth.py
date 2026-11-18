import unittest

import numpy as np

from tiny_voca import hf_synth, postfilter


class HFSynthTest(unittest.TestCase):
    def setUp(self):
        self.sr = postfilter.SAMPLE_RATE
        self.t = np.arange(self.sr) / self.sr
        self.x = sum(0.2 / (k ** 0.7) * np.sin(2 * np.pi * 200 * k * self.t)
                     for k in range(1, 45))
        self.x += 0.01 * np.random.default_rng(0).standard_normal(len(self.t))

    def test_f0_track(self):
        f0, strength = hf_synth.f0_track(self.x)
        self.assertAlmostEqual(float(np.median(f0[strength > 0.5])), 200.0, delta=12.0)

    def test_process_length_finite_peak(self):
        y = hf_synth.process(self.x, voice="F2", preset="hf")
        self.assertEqual(len(y), len(self.x))
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(float(np.abs(y).max()), 0.99 + 1e-9)

    def test_short_clip(self):
        y = hf_synth.process(np.zeros(100), voice="F2", preset="hf")
        self.assertEqual(len(y), 100)
        self.assertTrue(np.isfinite(y).all())

    def test_harmonic_magnitude_reduces_hf_flatness(self):
        # A synthetic harmonic HF with noise between harmonics.
        from tiny_voca import postfilter as pf
        mag = 0.01 * np.ones(len(hf_synth.FREQ))
        for k in range(100, 250):
            j = int(round(k * 200 / hf_synth.DF))
            if 0 < j < len(mag) - 1:
                mag[j - 1:j + 2] = 0.5
        rebuilt = hf_synth.harmonic_magnitude(mag, 200.0, 1.0, start=7000,
                                              floor_ratio=0.2, sigma=2.0)
        self.assertIsNotNone(rebuilt)
        self.assertEqual(len(rebuilt), len(mag))
        hf = hf_synth.FREQ >= 7000

        def flat(v):
            return float(np.exp(np.mean(np.log(v[hf] + 1e-12))) / np.mean(v[hf]))
        self.assertLess(flat(rebuilt), flat(mag))


if __name__ == "__main__":
    unittest.main()
