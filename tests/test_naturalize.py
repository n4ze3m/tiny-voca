import unittest

import numpy as np

from tiny_voca import naturalize, postfilter


class NaturalizeTest(unittest.TestCase):
    def setUp(self):
        self.sr = postfilter.SAMPLE_RATE
        self.t = np.arange(self.sr) / self.sr
        self.x = sum(0.15 / (k ** 0.7) * np.sin(2 * np.pi * 180 * k * self.t)
                     for k in range(1, 60))
        self.x += 0.002 * np.random.default_rng(0).standard_normal(len(self.t))

    def test_process_length_finite_peak(self):
        y = naturalize.process(self.x, voice="F2", preset="natural")
        self.assertEqual(len(y), len(self.x))
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(float(np.abs(y).max()), 0.99 + 1e-9)

    def test_short_clip(self):
        y = naturalize.process(np.zeros(100), voice="F2", preset="natural")
        self.assertEqual(len(y), 100)
        self.assertTrue(np.isfinite(y).all())

    def test_formant_shift_preserves_length(self):
        y = naturalize.formant_shift(self.x, 0.99)
        self.assertEqual(len(y), len(self.x))
        self.assertTrue(np.isfinite(y).all())

    def test_shimmer_and_room_can_be_disabled(self):
        dry = naturalize.process(self.x, voice=None, preset="dry")
        self.assertEqual(len(dry), len(self.x))
        natural = naturalize.process(self.x, voice=None, preset="natural", room=0, breath=0)
        self.assertEqual(len(natural), len(self.x))


if __name__ == "__main__":
    unittest.main()
