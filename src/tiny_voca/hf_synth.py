"""Experimental HF harmonic/formant resynthesis.

The current post filter raises HF spectral contrast, but the HF band is still a
mixture of harmonics and broadband vocoder noise.  This stage rebuilds the
voiced HF band from a harmonic model:

1. track F0 on every STFT frame;
2. snap each harmonic k*F0 to its local spectral peak (handles small F0 error);
3. smooth the harmonic amplitudes across k so the formant envelope survives;
4. rebuild the 7-12 kHz magnitude as those peaks plus a controlled noise floor;
5. keep the original phase and preserve the original HF energy.

The result is a cleaner harmonic HF with human-like flatness, without a treble
cut.  It is deliberately experimental: the default pipeline does not use it.
See ``postproc/HF_SYNTH.md``.

Usage::

    python -m tiny_voca.hf_synth --input warm.wav --output hf.wav --voice F2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import postfilter as pf

SAMPLE_RATE = pf.SAMPLE_RATE
N_FFT, HOP = pf.N_FFT, pf.HOP
FREQ = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
DF = SAMPLE_RATE / N_FFT

PRESETS = {
    "subtle": {"floor": 0.30, "sigma": 2.5},
    "hf": {"floor": 0.20, "sigma": 2.0},
    "strong": {"floor": 0.10, "sigma": 1.5},
}


# ------------------------------------------------------------------ analysis
def f0_track(x):
    """Frame-wise F0 and normalized autocorrelation strength."""
    frames = pf._frames(x)
    centered = frames - frames.mean(axis=1, keepdims=True)
    centered *= np.hanning(N_FFT)
    size = 2 * N_FFT
    spectrum = np.fft.rfft(centered, n=size, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=size, axis=1)[:, :N_FFT]
    autocorr /= np.maximum(autocorr[:, :1], 1e-12)
    low, high = int(SAMPLE_RATE / 500), int(SAMPLE_RATE / 60)
    band = autocorr[:, low:high]
    lag = low + np.argmax(band, axis=1)
    strength = band.max(axis=1)
    f0 = SAMPLE_RATE / lag
    # 3-frame median smooths octave jumps without flattening real glides
    if len(f0) >= 3:
        padded = np.pad(f0, 1, mode="edge")
        f0 = np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)
    return f0, strength


def _moving_percentile(values, size=17, q=70.0):
    pad = size // 2
    padded = np.pad(values, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.percentile(windows, q, axis=-1)


def _smooth_harmonics(values, window=5):
    if len(values) < window or window < 3:
        return values
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, window // 2, mode="edge"), kernel, mode="valid")


def harmonic_magnitude(mag, f0, strength, start=7000.0, floor_ratio=0.2,
                       sigma=2.0, preserve="total", smooth_k=5, strength_min=0.4):
    """Rebuild the HF magnitude for one frame.

    Returns ``new_mag`` or ``None`` when the frame should be left alone.
    """
    if strength < strength_min or not np.isfinite(f0) or not 55 <= f0 <= 500:
        return None
    floor = _moving_percentile(mag, 17, 70.0)
    peaks = []
    k = max(1, int(np.floor(start / f0)))
    while k * f0 < 11000:
        center = k * f0
        if center < start:
            k += 1
            continue
        # allow for F0 error at high k, but never reach the neighbouring harmonic
        half = max(2, int(round((0.10 * f0 + 0.015 * center) / DF)))
        half = min(half, max(2, int(0.40 * f0 / DF)))
        j0 = int(round(center / DF))
        lo, hi = max(1, j0 - half), min(len(FREQ) - 2, j0 + half + 1)
        j = lo + int(np.argmax(mag[lo:hi]))
        a, b, c = mag[j - 1], mag[j], mag[j + 1]
        d = np.clip(0.5 * (a - c) / (a - 2 * b + c + 1e-12), -0.5, 0.5)
        amplitude = b - 0.25 * (a - c) * d
        peaks.append((j, amplitude))
        k += 1
    if len(peaks) < 3:
        return None
    amplitudes = _smooth_harmonics(np.array([p[1] for p in peaks]), smooth_k)
    clean = floor_ratio * floor
    for (j, _), amplitude in zip(peaks, amplitudes):
        lo = max(1, j - int(4 * sigma))
        hi = min(len(FREQ) - 1, j + int(4 * sigma) + 1)
        index = np.arange(lo, hi)
        clean[lo:hi] += amplitude * np.exp(-0.5 * ((index - j) / sigma) ** 2)
    hf = FREQ >= start
    if preserve == "total":
        old = np.sum(mag[hf] ** 2) + 1e-12
        new = np.sum(clean[hf] ** 2) + 1e-12
    elif preserve == "harmonic":
        old = np.sum(np.array([p[1] for p in peaks]) ** 2) + 1e-12
        new = np.sum(amplitudes ** 2) + 1e-12
    else:
        raise ValueError(f"unknown preserve mode {preserve!r}")
    clean *= np.sqrt(old / new)
    ramp = np.clip((FREQ - start) / 1500.0, 0.0, 1.0)
    new_mag = mag.copy()
    new_mag[hf] = mag[hf] * (1 - ramp[hf]) + clean[hf] * ramp[hf]
    return new_mag


def resynthesize(x, start=7000.0, floor_ratio=0.2, sigma=2.0,
                 preserve="total", smooth_k=5, strength_min=0.4):
    spectrum = pf.stft(x)
    magnitude = np.abs(spectrum)
    f0, strength = f0_track(x)
    count = min(len(f0), spectrum.shape[0])
    new_magnitude = magnitude.copy()
    for i in range(count):
        rebuilt = harmonic_magnitude(magnitude[i], f0[i], strength[i], start=start,
                                     floor_ratio=floor_ratio, sigma=sigma,
                                     preserve=preserve, smooth_k=smooth_k,
                                     strength_min=strength_min)
        if rebuilt is not None:
            new_magnitude[i] = rebuilt
    return pf.istft(spectrum * (new_magnitude / (magnitude + 1e-12)), len(x))


# ------------------------------------------------------------------ public
def process(wave, voice=None, preset="hf", floor=None, sigma=None, start=7000.0,
            preserve="total", base="warm", peak=0.99, target_rms=None,
            dither=False, seed=0):
    """Post-filter (``base``) then rebuild the voiced HF band harmonically."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    config = PRESETS[preset]
    floor = config["floor"] if floor is None else floor
    sigma = config["sigma"] if sigma is None else sigma
    x = pf.process(np.asarray(wave, dtype=np.float64).reshape(-1), voice=voice,
                   preset=base, peak=1.0)
    x = resynthesize(x, start=start, floor_ratio=float(np.clip(floor, 0.0, 1.0)),
                     sigma=float(np.clip(sigma, 0.5, 6.0)), preserve=preserve)
    if target_rms is not None:
        x = pf.normalize_rms(x, target_rms)
    x = pf.limit_peak(x, peak)
    if dither:
        rng = np.random.default_rng(seed)
        x = x + (rng.random(len(x)) - rng.random(len(x))) / 32768.0
    return x


def process_file(source, output, voice=None, preset="hf", subtype="PCM_24", **kwargs):
    import soundfile as sf
    x, sr = sf.read(source, dtype="float64", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{source}: expected {SAMPLE_RATE} Hz, got {sr}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if subtype == "PCM_16":
        kwargs.setdefault("dither", True)
    y = process(x, voice=voice, preset=preset, **kwargs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, y, SAMPLE_RATE, subtype=subtype)
    return y


def main():
    parser = argparse.ArgumentParser(description="Experimental HF harmonic/formant resynthesis")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice", choices=("F2", "M2"), help="Voice of the clip")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="hf")
    parser.add_argument("--base", choices=tuple(pf.PRESETS), default="warm",
                        help="Spectral filter applied first (default: warm)")
    parser.add_argument("--floor", type=float, help="HF noise floor relative to envelope (default 0.2)")
    parser.add_argument("--sigma", type=float, help="Harmonic peak width in FFT bins (default 2)")
    parser.add_argument("--start", type=float, default=7000.0, help="First rebuilt frequency (Hz)")
    parser.add_argument("--preserve", choices=("total", "harmonic"), default="total")
    parser.add_argument("--target-rms", type=float)
    parser.add_argument("--peak", type=float, default=0.99)
    parser.add_argument("--subtype", choices=("FLOAT", "PCM_24", "PCM_16"), default="PCM_24")
    args = parser.parse_args()
    y = process_file(args.input, args.output, voice=args.voice, preset=args.preset,
                     base=args.base, floor=args.floor, sigma=args.sigma, start=args.start,
                     preserve=args.preserve, target_rms=args.target_rms, peak=args.peak,
                     subtype=args.subtype)
    print(f"wrote {args.output} ({len(y)/SAMPLE_RATE:.2f}s, peak {np.max(np.abs(y)):.3f})")


if __name__ == "__main__":
    main()
