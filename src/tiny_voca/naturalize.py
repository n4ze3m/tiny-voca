"""Full programmatic naturalness chain.

This composes the stages developed on the ``hf-harmonic-formant-synth`` branch:

* ``hf_synth``: 7-12 kHz harmonic rebuild on top of the ``warm`` filter;
* mild **formant warp** (spectral-envelope warp, F0 untouched);
* band-limited **shimmer** (amplitude micro-variation below 5 kHz);
* slow **expander**, sparse **breaths**, and a low-passed **room bed**;
* loudness matching and peak limiting.

The F2 pitch calibration (``pitch_shift=0.07`` at synthesis time) is separate:
it has to happen before post-processing.  See ``postproc/NATURAL.md``.

Usage::

    python -m tiny_voca.naturalize --input raw.wav --output natural.wav --voice F2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import hf_synth, humanize
from . import postfilter as pf

SAMPLE_RATE = pf.SAMPLE_RATE
N_FFT, HOP = pf.N_FFT, pf.HOP
FREQ = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)

PRESETS = {
    "soft": {"formant": 1.00, "shimmer": 0.005, "room": 0.6, "breath": 0.6, "dynamics": 0.6},
    "natural": {"formant": 0.99, "shimmer": 0.008, "room": 0.8, "breath": 0.7, "dynamics": 0.7},
    "formant": {"formant": 0.96, "shimmer": 0.010, "room": 0.8, "breath": 0.7, "dynamics": 0.7},
    "dry": {"formant": 0.99, "shimmer": 0.008, "room": 0.0, "breath": 0.0, "dynamics": 0.7},
}


# ------------------------------------------------------------------ stages
def lowpass(x, cutoff):
    """Zero-phase 8th-order-ish low-pass in the STFT domain."""
    if cutoff <= 0:
        return x
    spectrum = pf.stft(x)
    gain = 1.0 / np.sqrt(1.0 + (FREQ / cutoff) ** 8)
    return pf.istft(spectrum * gain, len(x))


def formant_shift(x, ratio=0.99, lifter=24):
    """Warp the spectral envelope without moving the harmonic comb."""
    if ratio == 1.0:
        return x
    spectrum = pf.stft(x)
    magnitude = np.abs(spectrum)
    cepstrum = np.fft.irfft(np.log(magnitude + 1e-12), n=N_FFT, axis=1)
    keep = np.zeros(N_FFT)
    keep[:lifter] = 1.0
    keep[-lifter:] = 1.0
    envelope = np.exp(np.fft.rfft(cepstrum * keep, axis=1).real)
    residual = magnitude / (envelope + 1e-12)
    source = FREQ / ratio
    warped = np.stack([np.interp(source, FREQ, envelope[i], left=envelope[i, 0],
                                 right=envelope[i, -1])
                       for i in range(envelope.shape[0])])
    new_magnitude = warped * residual
    new_magnitude *= ((magnitude.sum(1, keepdims=True) + 1e-12) /
                      (new_magnitude.sum(1, keepdims=True) + 1e-12))
    silence, voiced, unvoiced, level, _ = pf.frame_classes(x)
    ramp = np.clip((level + 50) / 10, 0.0, 1.0)[:, None]
    ramp *= np.clip((voiced[:, None] - 0.2) / 0.3, 0.0, 1.0)
    blend = magnitude * (1 - ramp) + new_magnitude * ramp
    return pf.istft(spectrum * (blend / (magnitude + 1e-12)), len(x))


def add_shimmer(x, amount=0.008, rate=5.0, cutoff=5000.0, seed=0):
    """Slow amplitude variation applied only below ``cutoff``."""
    if amount <= 0:
        return x
    rng = np.random.default_rng(seed)
    n = len(x)
    noise = rng.standard_normal(n)
    alpha = 1 - np.exp(-2 * np.pi * rate / SAMPLE_RATE)
    smooth = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += alpha * (noise[i] - acc)
        smooth[i] = acc
    smooth /= (np.std(smooth) + 1e-12)
    smooth = np.clip(smooth, -2.5, 2.5)
    _, voiced, _, _, _ = pf.frame_classes(x)
    gate = np.interp(np.arange(n) / SAMPLE_RATE,
                     np.arange(len(voiced)) * HOP / SAMPLE_RATE, voiced,
                     left=0, right=0)
    width = min(481, len(gate) if len(gate) % 2 else len(gate) - 1)
    if width >= 3:
        kernel = np.hanning(width)
        kernel /= kernel.sum()
        gate = np.convolve(gate, kernel, mode="same")
    gain = 1.0 + amount * smooth * gate
    low = lowpass(x, cutoff)
    return x + (gain - 1.0) * low


def add_room(x, amount=0.8, cutoff=2000.0, wet_gain=0.18, seed=0):
    """Low-passed room tone + early reflections; keeps HF clean."""
    if amount <= 0:
        return x
    rng = np.random.default_rng(seed + 3)
    level_db = -64.0 + 6.0 * min(amount, 1.5)
    tone = humanize.room_noise(len(x), rng) * 10 ** (level_db / 20.0)
    tone = lowpass(tone, cutoff)
    wet = humanize._convolve(x, humanize._wet_rir(seed))[:len(x)]
    wet = lowpass(wet, cutoff)
    return x + tone + wet * wet_gain * min(amount, 1.5)


# ------------------------------------------------------------------ public
def process(wave, voice=None, preset="natural", formant=None, shimmer=None,
            room=None, breath=None, dynamics=None, room_cutoff=2000.0,
            wet_gain=0.18, target_rms=-18.5, peak=0.99, seed=0):
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    config = PRESETS[preset]
    formant = config["formant"] if formant is None else float(np.clip(formant, 0.85, 1.1))
    shimmer_amount = config["shimmer"] if shimmer is None else float(np.clip(shimmer, 0, 0.05))
    room = config["room"] if room is None else float(np.clip(room, 0, 2))
    breath = config["breath"] if breath is None else float(np.clip(breath, 0, 2))
    dynamics = config["dynamics"] if dynamics is None else float(np.clip(dynamics, 0, 3))

    x = hf_synth.process(np.asarray(wave, dtype=np.float64).reshape(-1),
                         voice=voice, preset="hf", base="warm", target_rms=None)
    x = formant_shift(x, formant)
    x = add_shimmer(x, shimmer_amount, seed=seed)
    x = humanize.slow_expander(x, dynamics)
    x = humanize.add_breaths(x, breath, seed=seed)
    x = add_room(x, room, cutoff=room_cutoff, wet_gain=wet_gain, seed=seed)
    if target_rms is not None:
        x = pf.normalize_rms(x, target_rms)
    return pf.limit_peak(x, peak)


def process_file(source, output, voice=None, preset="natural", subtype="PCM_24",
                 seed=0, **kwargs):
    import soundfile as sf
    x, sr = sf.read(source, dtype="float64", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{source}: expected {SAMPLE_RATE} Hz, got {sr}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = process(x, voice=voice, preset=preset, seed=seed, **kwargs)
    if subtype == "PCM_16":
        rng = np.random.default_rng(seed)
        y = y + (rng.random(len(y)) - rng.random(len(y))) / 32768.0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, y, SAMPLE_RATE, subtype=subtype)
    return y


def main():
    parser = argparse.ArgumentParser(description="Full programmatic naturalness chain")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice", choices=("F2", "M2"))
    parser.add_argument("--preset", choices=tuple(PRESETS), default="natural")
    parser.add_argument("--formant", type=float, help="envelope warp ratio (1 = off)")
    parser.add_argument("--shimmer", type=float, help="amplitude shimmer amount (0..0.05)")
    parser.add_argument("--room", type=float, help="room amount (0..2)")
    parser.add_argument("--breath", type=float)
    parser.add_argument("--dynamics", type=float)
    parser.add_argument("--room-cutoff", type=float, default=2000.0)
    parser.add_argument("--wet-gain", type=float, default=0.18)
    parser.add_argument("--target-rms", type=float, default=-18.5)
    parser.add_argument("--peak", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subtype", choices=("FLOAT", "PCM_24", "PCM_16"), default="PCM_24")
    args = parser.parse_args()
    y = process_file(args.input, args.output, voice=args.voice, preset=args.preset,
                     formant=args.formant, shimmer=args.shimmer, room=args.room,
                     breath=args.breath, dynamics=args.dynamics,
                     room_cutoff=args.room_cutoff, wet_gain=args.wet_gain,
                     target_rms=args.target_rms, peak=args.peak, seed=args.seed,
                     subtype=args.subtype)
    print(f"wrote {args.output} ({len(y)/SAMPLE_RATE:.2f}s, peak {np.max(np.abs(y)):.3f})")


if __name__ == "__main__":
    main()
