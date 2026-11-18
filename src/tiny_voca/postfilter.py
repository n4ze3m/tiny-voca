"""Spectral post-processing for tiny-voca output.

Two measured defects are addressed:

* **Tonal imbalance** (mostly F2): a little too much energy at 160-315 Hz and a
  little too little at 2.5-10 kHz, especially in voiced frames.  A gentle
  voicing-adaptive shelf pair corrects it.  Fricatives get a stronger low-mid
  cut because the vocoder's harmonic excitation keeps running through unvoiced
  frames.
* **Residual hiss** in pauses: the vocoder has no explicit voicing gate, so
  the noise branch keeps running in pauses.  A minimum-statistics spectral
  subtraction stage removes about 13 dB of it while leaving speech essentially
  untouched.
* **Thin low-mid body** (optional, ``warm``): the human reference has 3-4 dB
  more 300-1000 Hz energy.  A broad 180-700 Hz augmentation restores some of
  it without changing the harmonic structure.
* **High-frequency sizzle** (``smooth``/``warm``/``human``): the Qwen target
  itself has flatter HF spectra than human speech.  Raising the HF magnitude
  to a power gamma and renormalising keeps the LTAS but restores harmonic and
  formant contrast (``--contrast``).

Everything is plain STFT magnitude filtering (no neural networks).  The
default preset is conservative; ``neutral`` only high-passes and limits.

Usage::

    python -m tiny_voca.postfilter --input in.wav --output out.wav --voice F2

The analysis that produced these settings is in ``postproc/REPORT.md``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
N_FFT = 1024
HOP = 256
WINDOW = np.hanning(N_FFT)


# --------------------------------------------------------------------- stft
def _frames(x, size=N_FFT, hop=HOP):
    pad = size // 2
    xp = np.pad(x, pad, mode="reflect")
    count = 1 + (len(xp) - size) // hop
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return xp[index]


def stft(x):
    return np.fft.rfft(_frames(x) * WINDOW, axis=1)


def istft(spectrum, length):
    frames = np.fft.irfft(spectrum, n=N_FFT, axis=1) * WINDOW
    out = np.zeros(HOP * (len(frames) - 1) + N_FFT)
    norm = np.zeros_like(out)
    for i, frame in enumerate(frames):
        out[i * HOP:i * HOP + N_FFT] += frame
        norm[i * HOP:i * HOP + N_FFT] += WINDOW ** 2
    out /= np.maximum(norm, 1e-8)
    start = N_FFT // 2
    return out[start:start + length]


# --------------------------------------------------------------------- eq
def _shelf_low(freq, cutoff, gain_db):
    """First-order low shelf (gain at DC, 0 dB at HF)."""
    return gain_db / (1.0 + (freq / cutoff) ** 2)


def _shelf_high(freq, cutoff, gain_db):
    """First-order high shelf (0 dB at DC, gain at HF)."""
    return gain_db * (freq / cutoff) ** 2 / (1.0 + (freq / cutoff) ** 2)


def _peak(freq, center, q, gain_db):
    """Second-order peaking EQ magnitude in dB (RBJ cookbook)."""
    if gain_db == 0:
        return np.zeros_like(freq)
    amp = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * center / SAMPLE_RATE
    alpha = np.sin(w0) / (2 * q)
    cos = np.cos(w0)
    # |H(e^jw)|^2 of the RBJ peaking filter
    b = np.array([1 + alpha * amp, -2 * cos, 1 - alpha * amp])
    a = np.array([1 + alpha / amp, -2 * cos, 1 - alpha / amp])
    z = np.exp(-1j * 2 * np.pi * freq / SAMPLE_RATE)
    num = b[0] + b[1] * z + b[2] * z ** 2
    den = a[0] + a[1] * z + a[2] * z ** 2
    return 20 * np.log10(np.maximum(np.abs(num / den), 1e-9))


def body_gain(freq, amount):
    """Broad 180-700 Hz body augmentation (dB at the 380 Hz peak)."""
    if amount <= 0:
        return np.zeros_like(freq)
    return (_peak(freq, 180, 0.7, 0.35 * amount)
            + _peak(freq, 380, 0.6, amount)
            + _peak(freq, 700, 0.7, 0.40 * amount))


def tone_gain(voice, kind, n_freq=N_FFT // 2 + 1, body=0.0):
    """Spectral gain in dB for one frame class.

    ``kind`` is ``voiced``, ``unvoiced`` or ``silence``.  The values come from
    the per-phoneme measurements in postproc/REPORT.md; ``body`` adds a broad
    low-mid augmentation (the human reference has more 300-1000 Hz energy than
    the vocoder).  Total gain is kept inside +/-5 dB.
    """
    freq = np.fft.rfftfreq((n_freq - 1) * 2, 1 / SAMPLE_RATE)
    if voice == "F2":
        if kind == "voiced":
            gain = _shelf_low(freq, 300, -2.8) + _peak(freq, 3800, 1.0, 2.2)
            gain += body_gain(freq, body)
        elif kind == "unvoiced":
            gain = _shelf_low(freq, 350, -3.6) + _peak(freq, 3200, 1.1, 0.5)
        else:
            gain = _shelf_low(freq, 120, -1.0)
    elif voice == "M2":
        if kind == "voiced":
            gain = _shelf_low(freq, 300, -0.5) + _shelf_high(freq, 6000, 1.0)
            gain += body_gain(freq, body)
        elif kind == "unvoiced":
            gain = _shelf_low(freq, 300, -1.2) + _shelf_high(freq, 4000, 0.2)
        else:
            gain = _shelf_low(freq, 120, -1.0)
    else:
        gain = np.zeros(n_freq)
    gain[freq < 60] = 0.0
    return np.clip(gain, -5.0, 5.0)


# --------------------------------------------------------------------- vad
def frame_classes(x):
    """Soft silence / voiced / unvoiced weights per STFT frame, plus level.

    ``level`` is true frame RMS in dBFS (unwindowed), as in postproc/common.py.
    """
    frames = _frames(x)
    level = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    centered = (frames - frames.mean(axis=1, keepdims=True)) * WINDOW
    size = 2 * N_FFT
    spectrum = np.fft.rfft(centered, n=size, axis=1)
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2, n=size, axis=1)[:, :N_FFT]
    autocorr /= np.maximum(autocorr[:, :1], 1e-12)
    low, high = int(SAMPLE_RATE / 500), int(SAMPLE_RATE / 60)
    peak = autocorr[:, low:high].max(axis=1)
    # soft silence gate: 0 above -35 dBFS, 1 below -47 dBFS
    silence = np.clip((-35.0 - level) / 12.0, 0.0, 1.0)
    voiced = np.clip((peak - 0.25) / 0.25, 0.0, 1.0) * (1.0 - silence)
    unvoiced = (1.0 - voiced) * (1.0 - silence)
    return silence, voiced, unvoiced, level, peak


# --------------------------------------------------------------------- stages
def _smooth_time(gain, width=5):
    if width <= 1:
        return gain
    kernel = np.hanning(width + 2)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(gain, ((width // 2, width // 2), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 0, padded)


def _smooth_freq(gain, width=5):
    if width <= 1:
        return gain
    kernel = np.hanning(width + 2)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(gain, ((0, 0), (width // 2, width // 2)), mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 1, padded)


def hf_contrast_gain(mag, level, gamma=2.0, start=3500.0):
    """Per-frame power-law contrast enhancement above ``start`` Hz.

    The target (and therefore the model) has flatter HF spectra than human
    speech: broadband noise fills the valleys.  Raising the HF magnitude to
    ``gamma`` and renormalising to the original band energy keeps the LTAS but
    increases peak-to-valley contrast, which moves the HF texture toward human
    (lower spectral flatness) without a treble cut.  Silence is left alone.
    """
    if gamma <= 1.0:
        return np.ones_like(mag)
    freq = np.fft.rfftfreq((mag.shape[1] - 1) * 2, 1 / SAMPLE_RATE)
    hf = freq >= start
    band = mag[:, hf]
    new = band ** gamma
    new *= (band.sum(axis=1, keepdims=True) + 1e-12) / (new.sum(axis=1, keepdims=True) + 1e-12)
    gain = np.ones_like(mag)
    gain[:, hf] = new / (band + 1e-12)
    ramp = np.clip((freq - start) / 1500.0, 0.0, 1.0)[None, :]
    active = np.clip((level + 50.0) / 10.0, 0.0, 1.0)[:, None]
    return 1.0 - active * ramp * (1.0 - gain)


def dehiss_gain(power, level, strength=1.0, floor_db=-14.0, smooth_t=3, smooth_f=7):
    """Minimum-statistics spectral subtraction, gated to quiet frames.

    The noise power is the mean spectrum of the quietest fifth of the frames,
    which estimates the noise mean instead of a low percentile (a percentile
    of exponentially distributed periodogram bins is biased far below the
    mean).  ``strength`` 0..2 scales the over-subtraction.  Returns a *power*
    gain in [10**(floor_db/10), 1]; ``process`` takes its square root for the
    magnitude.  Speech frames are left alone: the gate is zero above -35 dBFS
    and ramps on below -47 dBFS.
    """
    if strength <= 0:
        return np.ones_like(power)
    quiet = level <= np.percentile(level, 20)
    if quiet.sum() >= 5:
        noise = power[quiet].mean(axis=0)
    else:
        # no clear pause: compensate the periodogram percentile bias
        noise = np.percentile(power, 10, axis=0) * 6.0
    gain = np.maximum(power - (0.25 + strength) * noise, 0.0) / (power + 1e-12)
    gain = np.maximum(gain, 10 ** (floor_db / 10))
    gain = _smooth_time(gain, smooth_t)
    gain = _smooth_freq(gain, smooth_f)
    gate = np.clip((-35.0 - level) / 12.0, 0.0, 1.0)[:, None]
    return 1.0 - gate * (1.0 - gain)


def limit_peak(x, peak=0.99):
    loudest = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (peak / loudest) if loudest > peak else x


def normalize_rms(x, target_db=-20.0):
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms <= 0:
        return x
    return x * (10 ** (target_db / 20) / rms)


# --------------------------------------------------------------------- public
PRESETS = {
    "neutral": {"tone": False, "dehiss": 0.0, "body": 0.0, "contrast": 1.0, "floor": -14.0, "smooth": (5, 7)},
    "clean": {"tone": True, "dehiss": 1.0, "body": 0.0, "contrast": 1.0, "floor": -14.0, "smooth": (3, 5)},
    "smooth": {"tone": True, "dehiss": 1.0, "body": 0.0, "contrast": 1.5, "floor": -10.0, "smooth": (5, 7)},
    "warm": {"tone": True, "dehiss": 1.0, "body": 3.5, "contrast": 1.5, "floor": -10.0, "smooth": (5, 7)},
    "strong": {"tone": True, "dehiss": 1.8, "body": 0.0, "contrast": 2.0, "floor": -12.0, "smooth": (5, 7)},
}


def process(wave, voice=None, preset="clean", tone=None, dehiss=None, body=None,
            contrast=None, peak=0.99, target_rms=None):
    """Filter a float waveform (numpy array or torch tensor).

    Parameters
    ----------
    wave : 1-D array-like
        Audio at 24 kHz, nominally in [-1, 1].
    voice : ``"F2"``, ``"M2"`` or ``None``
        Selects the fitted tone curve.  ``None`` skips the tone EQ.
    preset : ``"neutral"``, ``"clean"``, ``"strong"``
        Defaults for ``tone`` / ``dehiss``; explicit arguments override.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    config = PRESETS[preset]
    tone = config["tone"] if tone is None else tone
    dehiss = config["dehiss"] if dehiss is None else dehiss
    dehiss = float(np.clip(dehiss, 0.0, 3.0))
    body = config["body"] if body is None else body
    body = float(np.clip(body, 0.0, 5.0))
    contrast = config["contrast"] if contrast is None else contrast
    contrast = float(np.clip(contrast, 1.0, 3.0))
    floor_db = config["floor"]
    smooth_t, smooth_f = config["smooth"]

    x = np.asarray(wave, dtype=np.float64).reshape(-1)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
        short = True
    else:
        short = False

    spectrum = stft(x)
    power = np.abs(spectrum) ** 2
    silence, voiced, unvoiced, level, _ = frame_classes(x)

    gain_db = np.zeros((spectrum.shape[0], spectrum.shape[1]))
    if tone and voice is not None:
        gv = tone_gain(voice, "voiced", spectrum.shape[1], body=body)
        gu = tone_gain(voice, "unvoiced", spectrum.shape[1])
        gs = tone_gain(voice, "silence", spectrum.shape[1])
        gain_db = (voiced[:, None] * gv + unvoiced[:, None] * gu + silence[:, None] * gs)
        gain_db = _smooth_time(gain_db, 5)

    magnitude = 10 ** (gain_db / 20.0)
    if dehiss:
        magnitude *= np.sqrt(dehiss_gain(power, level, dehiss, floor_db=floor_db,
                                          smooth_t=smooth_t, smooth_f=smooth_f))
    if contrast > 1.0:
        # contrast must be measured on the actual signal magnitude, not on the
        # (smooth) filter curve
        magnitude *= hf_contrast_gain(np.abs(spectrum) * magnitude, level, contrast)
    # 2nd-order zero-phase high-pass at 55 Hz: removes sub-bass rumble without
    # touching the lowest voice harmonics (F0 >= 50 Hz is protected by the
    # gentle 12 dB/oct rolloff).
    freq = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    magnitude *= 1.0 / np.sqrt(1.0 + (55.0 / np.maximum(freq, 1e-6)) ** 4)

    y = istft(spectrum * magnitude, len(x))
    if short:
        y = y[:len(np.asarray(wave).reshape(-1))]
    if target_rms is not None:
        y = normalize_rms(y, target_rms)
    return limit_peak(y, peak)


def process_file(source, output, voice=None, preset="clean", subtype="FLOAT", seed=0, **kwargs):
    import soundfile as sf
    x, sr = sf.read(source, dtype="float64", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{source}: expected {SAMPLE_RATE} Hz, got {sr}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = process(x, voice=voice, preset=preset, **kwargs)
    if subtype == "PCM_16":
        rng = np.random.default_rng(seed)
        y = y + (rng.random(len(y)) - rng.random(len(y))) / 32768.0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, y, SAMPLE_RATE, subtype=subtype)
    return y


def main():
    parser = argparse.ArgumentParser(description="Spectral post-filter for tiny-voca 24 kHz audio")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice", choices=("F2", "M2"), help="Voice of the clip (tone curve)")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="clean")
    parser.add_argument("--dehiss", type=float, help="0..2: spectral subtraction strength")
    parser.add_argument("--body", type=float, help="dB of 180-700 Hz body augmentation (0..5)")
    parser.add_argument("--contrast", type=float, help="HF spectral contrast gamma (1 = off, 2 = strong)")
    parser.add_argument("--no-tone", action="store_true", help="Skip the tone EQ")
    parser.add_argument("--target-rms", type=float, help="Normalise to this dBFS RMS")
    parser.add_argument("--peak", type=float, default=0.99)
    parser.add_argument("--subtype", choices=("FLOAT", "PCM_24", "PCM_16"), default="FLOAT")
    args = parser.parse_args()
    y = process_file(args.input, args.output, voice=args.voice, preset=args.preset,
                     tone=False if args.no_tone else None, dehiss=args.dehiss,
                     body=args.body, contrast=args.contrast,
                     target_rms=args.target_rms, peak=args.peak, subtype=args.subtype)
    print(f"wrote {args.output} ({len(y)/SAMPLE_RATE:.2f}s, peak {np.max(np.abs(y)):.3f})")


if __name__ == "__main__":
    main()
