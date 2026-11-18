"""Programmatic 'human' layer: room, breaths and gentle dynamics.

The vocoder is too dry and too compressed compared with human speech: its
quiet floor is around -72 dBFS, while the human reference here sits around
-57 dBFS, and its 300-1000 Hz body is 3-4 dB weaker.  This module starts from
the ``warm`` spectral filter and adds three non-neural cues:

* a synthetic room (warm low-passed tone + sparse early reflections, no long
  reverb);
* occasional breath noise inside long pauses;
* a slow expander that restores a little p10-p90 dynamic range.

Everything is plain DSP and the speech band is left intact.  Defaults are
subtle; ``--room 0`` / ``--breath 0`` / ``--dynamics 0`` disable a stage.

Usage::

    python -m tiny_voca.humanize --input stock.wav --output human.wav --voice F2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import postfilter as pf

SAMPLE_RATE = pf.SAMPLE_RATE

PRESETS = {
    "subtle": {"room": 0.5, "breath": 0.4, "dynamics": 0.4},
    "human": {"room": 0.7, "breath": 0.7, "dynamics": 0.7},
    "lively": {"room": 1.2, "breath": 1.0, "dynamics": 1.0},
}


# ------------------------------------------------------------------ helpers
def pink_noise(n, rng):
    """1/f noise band-limited to roughly 50 Hz - 9 kHz, unit RMS."""
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    f[0] = f[1]
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(f)
    spec *= 1 / np.sqrt(1 + (50 / f) ** 4)
    spec *= 1 / np.sqrt(1 + (f / 9000) ** 4)
    x = np.fft.irfft(spec, n=n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def room_noise(n, rng):
    """Warm 1/f^0.65 room bed, low-passed at 3 kHz: body without HF haze."""
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    f[0] = f[1]
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / f ** 0.65
    spec *= 1 / np.sqrt(1 + (45 / f) ** 4)
    spec *= 1 / np.sqrt(1 + (f / 3000) ** 4)
    x = np.fft.irfft(spec, n=n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def band_noise(n, low, high, rng):
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec *= 1 / np.sqrt(1 + (low / np.maximum(f, 1e-6)) ** 4)
    spec *= 1 / np.sqrt(1 + (f / high) ** 4)
    x = np.fft.irfft(spec, n=n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def _convolve(x, h):
    n = len(x) + len(h) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), n=size)[:n]


def _wet_rir(seed=0):
    """Early reflections plus a short diffuse tail; direct path excluded."""
    rng = np.random.default_rng(seed)
    length = int(0.22 * SAMPLE_RATE)
    rir = np.zeros(length)
    for delay, gain in ((0.003, -0.12), (0.007, 0.08), (0.011, -0.06),
                        (0.017, 0.05), (0.023, -0.04), (0.031, 0.03)):
        rir[int(delay * SAMPLE_RATE)] += gain
    t = np.arange(length) / SAMPLE_RATE
    tail = rng.standard_normal(length) * np.exp(-t / 0.05) * 0.04
    for i in range(1, length):
        tail[i] = 0.7 * tail[i - 1] + 0.3 * tail[i]
    return rir + tail


def slow_expander(x, strength=1.0):
    """Attenuate quiet speech a little; increases p10-p90 dynamic range."""
    if strength <= 0:
        return x
    window = max(8, int(0.05 * SAMPLE_RATE))
    env = np.sqrt(np.convolve(x ** 2, np.ones(window) / window, mode="same"))
    db = 20 * np.log10(env + 1e-9)
    active = db > -45
    if active.sum() < 100:
        return x
    ref = np.median(db[active])
    delta = np.clip(ref - db, 0, 10)
    gain_db = -0.16 * strength * delta
    gain = np.convolve(10 ** (gain_db / 20), np.ones(int(0.04 * SAMPLE_RATE)) /
                       int(0.04 * SAMPLE_RATE), mode="same")
    return x * gain


def add_breaths(x, strength=1.0, seed=0):
    """Add a quiet breath inside long pauses; does not change the duration."""
    if strength <= 0:
        return x
    rng = np.random.default_rng(seed + 11)
    silence, voiced, unvoiced, level, _ = pf.frame_classes(x)
    active = level > -45
    speech_db = float(np.mean(level[active])) if active.sum() else -22.0
    runs, i = [], 0
    while i < len(silence):
        if silence[i] > 0.8:
            j = i
            while j < len(silence) and silence[j] > 0.8:
                j += 1
            runs.append((i * pf.HOP, j * pf.HOP))
            i = j
        else:
            i += 1
    for start, end in runs:
        if end - start < int(0.28 * SAMPLE_RATE):
            continue
        if rng.random() > 0.55 * min(strength, 1.5):
            continue
        length = int(rng.uniform(0.10, 0.16) * SAMPLE_RATE)
        room = end - start - length - int(0.06 * SAMPLE_RATE)
        if room <= 0:
            continue
        at = start + int(0.05 * SAMPLE_RATE) + rng.integers(0, room)
        if at + length > len(x):
            continue
        env = np.hanning(length) ** 0.8
        noise = band_noise(length, 350, 5000, rng)
        amp = 10 ** ((speech_db - 33) / 20) / (np.sqrt(np.mean(env ** 2)) + 1e-12)
        x[at:at + length] += noise * env * amp * min(strength, 1.5)
    return x


def add_room(x, amount=1.0, seed=0):
    if amount <= 0:
        return x
    rng = np.random.default_rng(seed + 3)
    # quiet room tone: -61 dBFS at amount 0.5, -58 dBFS at 1.0
    level_db = -64.0 + 6.0 * min(amount, 1.5)
    tone = room_noise(len(x), rng) * 10 ** (level_db / 20) * min(amount, 1.5) ** 0.5
    wet = _convolve(x, _wet_rir(seed))[:len(x)]
    return x + tone + wet * 0.35 * amount


# ------------------------------------------------------------------ public
def process(wave, voice=None, preset="human", room=None, breath=None, dynamics=None,
            contrast=None, seed=0, peak=0.99, target_rms=None, base="warm"):
    """Clean + body augment + room/breaths/dynamics.

    ``wave`` is 24 kHz.  ``voice`` selects the tone curve (``"F2"`` / ``"M2"``).
    Explicit ``room`` / ``breath`` / ``dynamics`` override the preset.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    config = PRESETS[preset]
    room = config["room"] if room is None else room
    breath = config["breath"] if breath is None else breath
    dynamics = config["dynamics"] if dynamics is None else dynamics

    x = pf.process(np.asarray(wave, dtype=np.float64).reshape(-1),
                   voice=voice, preset=base, contrast=contrast, peak=1.0)
    x = slow_expander(x, float(np.clip(dynamics, 0, 3)))
    x = add_breaths(x, float(np.clip(breath, 0, 2)), seed)
    x = add_room(x, float(np.clip(room, 0, 2)), seed)
    if target_rms is not None:
        x = pf.normalize_rms(x, target_rms)
    return pf.limit_peak(x, peak)


def _dither(y, rng):
    """TPDF dither at one 16-bit LSB."""
    return y + (rng.random(len(y)) - rng.random(len(y))) / 32768.0


def process_file(source, output, voice=None, preset="human", subtype="PCM_24", seed=0, **kwargs):
    import soundfile as sf
    x, sr = sf.read(source, dtype="float64", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{source}: expected {SAMPLE_RATE} Hz, got {sr}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = process(x, voice=voice, preset=preset, seed=seed, **kwargs)
    if subtype == "PCM_16":
        y = _dither(y, np.random.default_rng(seed))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, y, SAMPLE_RATE, subtype=subtype)
    return y


def main():
    parser = argparse.ArgumentParser(description="Add programmatic room/breath/dynamics to tiny-voca audio")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice", choices=("F2", "M2"), help="Voice of the clip (tone curve)")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="human")
    parser.add_argument("--room", type=float, help="0..1.5 room tone + early reflections")
    parser.add_argument("--breath", type=float, help="0..1.5 breath probability/level")
    parser.add_argument("--dynamics", type=float, help="0..2 slow expander strength")
    parser.add_argument("--contrast", type=float, help="HF contrast gamma (1 = off, 1.5 = default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak", type=float, default=0.99)
    parser.add_argument("--target-rms", type=float, help="Normalise to this dBFS RMS")
    parser.add_argument("--subtype", choices=("FLOAT", "PCM_24", "PCM_16"), default="PCM_24")
    args = parser.parse_args()
    y = process_file(args.input, args.output, voice=args.voice, preset=args.preset,
                     room=args.room, breath=args.breath, dynamics=args.dynamics,
                     contrast=args.contrast, seed=args.seed, peak=args.peak,
                     target_rms=args.target_rms, subtype=args.subtype)
    print(f"wrote {args.output} ({len(y)/SAMPLE_RATE:.2f}s, peak {np.max(np.abs(y)):.3f})")


if __name__ == "__main__":
    main()
