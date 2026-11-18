import unittest

import numpy as np

from tiny_voca import postfilter


class PostfilterTest(unittest.TestCase):
    def setUp(self):
        self.sr = postfilter.SAMPLE_RATE
        self.t = np.arange(self.sr) / self.sr

    def test_neutral_is_transparent(self):
        x = 0.5 * np.sin(2 * np.pi * 220 * self.t)
        y = postfilter.process(x, voice=None, preset="neutral", tone=False, dehiss=0)
        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(float(np.abs(y).max()), 0.99 + 1e-9)
        self.assertGreater(float(np.corrcoef(x, y)[0, 1]), 0.999)

    def test_short_clip(self):
        y = postfilter.process(np.zeros(100), voice="F2", preset="clean")
        self.assertEqual(len(y), 100)
        self.assertTrue(np.isfinite(y).all())

    def test_dehiss_cleans_pauses_and_keeps_speech(self):
        t = self.t
        speech = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 0.5)
        rng = np.random.default_rng(0)
        noise = 0.003 * rng.standard_normal(len(t))
        x = speech + noise
        y = postfilter.process(x, voice=None, preset="clean", tone=False, dehiss=1.0)
        quiet = t > 0.6
        drop = 20 * np.log10(np.sqrt(np.mean(x[quiet] ** 2)) / np.sqrt(np.mean(y[quiet] ** 2)))
        self.assertGreater(drop, 3.0)
        keep = 20 * np.log10(np.sqrt(np.mean(y[~quiet] ** 2)) / np.sqrt(np.mean(x[~quiet] ** 2)))
        self.assertLess(abs(keep), 1.5)

    def test_tone_curves_are_bounded_and_shaped(self):
        freq = np.fft.rfftfreq(postfilter.N_FFT, 1 / self.sr)
        for voice in ("F2", "M2"):
            for kind in ("voiced", "unvoiced", "silence"):
                gain = postfilter.tone_gain(voice, kind)
                self.assertEqual(len(gain), len(freq))
                self.assertTrue(np.isfinite(gain).all())
                self.assertLessEqual(float(np.abs(gain).max()), 4.0 + 1e-9)
        f2 = postfilter.tone_gain("F2", "voiced")
        self.assertLess(float(f2[np.argmin(abs(freq - 100))]), 0.0)
        self.assertGreater(float(f2[np.argmin(abs(freq - 3800))]), 0.0)

    def test_contrast_preserves_band_energy_and_shapes_peaks(self):
        mag = np.ones((4, postfilter.N_FFT // 2 + 1))
        mag[:, 200] = 10.0
        level = np.full(4, -20.0)
        gain = postfilter.hf_contrast_gain(mag, level, gamma=2.0)
        self.assertGreater(float(gain[0, 200]), 1.0)
        self.assertLess(float(gain[0, 201]), 1.0)
        out = mag * gain
        f = np.fft.rfftfreq(postfilter.N_FFT, 1 / self.sr)
        hf = f >= 3500
        np.testing.assert_allclose(out[:, hf].sum(axis=1), mag[:, hf].sum(axis=1), rtol=0.05)

    def test_peak_limit(self):
        x = np.ones(self.sr) * 2.0
        y = postfilter.process(x, voice=None, preset="neutral", tone=False, dehiss=0)
        self.assertLessEqual(float(np.abs(y).max()), 0.99 + 1e-9)


if __name__ == "__main__":
    unittest.main()
