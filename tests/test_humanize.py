import unittest

import numpy as np

from tiny_voca import humanize, postfilter


class HumanizeTest(unittest.TestCase):
    def setUp(self):
        self.sr = postfilter.SAMPLE_RATE
        self.t = np.arange(self.sr) / self.sr
        self.x = 0.3 * np.sin(2 * np.pi * 220 * self.t) * (self.t < 0.5)
        self.x += 0.0005 * np.random.default_rng(0).standard_normal(len(self.t))

    def test_length_finite_and_peak(self):
        y = humanize.process(self.x, voice="F2", preset="human")
        self.assertEqual(len(y), len(self.x))
        self.assertTrue(np.isfinite(y).all())
        self.assertLessEqual(float(np.abs(y).max()), 0.99 + 1e-9)

    def test_room_raises_floor(self):
        clean = postfilter.process(self.x, voice=None, preset="warm")
        dry = humanize.process(self.x, voice=None, preset="human", room=0, breath=0, dynamics=0)
        wet = humanize.process(self.x, voice=None, preset="human", room=1, breath=0, dynamics=0)
        quiet = self.t > 0.7
        rms = lambda v: float(np.sqrt(np.mean(v[quiet] ** 2)))
        self.assertLess(np.max(np.abs(dry - clean)), 1e-3)
        self.assertGreater(rms(wet), rms(dry) * 2.5)

    def test_stages_can_be_disabled(self):
        y = humanize.process(self.x, voice=None, preset="human", room=0, breath=0, dynamics=0)
        z = postfilter.process(self.x, voice=None, preset="warm")
        self.assertLess(float(np.max(np.abs(y - z))), 1e-3)

    def test_pink_noise_is_unit_rms(self):
        n = humanize.pink_noise(self.sr, np.random.default_rng(0))
        self.assertAlmostEqual(float(np.sqrt(np.mean(n ** 2))), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
