"""Shared audio / spectral helpers for the tiny-voca post-processing study.

Everything here is deliberately dependency-light: numpy + soundfile only.
The analysis runs on CPU and never touches training code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "postproc" / "out"
SR = 24000

# ---------------------------------------------------------------- audio i/o
def load(path, mono=True):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1 and mono:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz, expected {SR}")
    return x


def save(path, x, subtype="FLOAT"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(x, dtype=np.float64), SR, subtype=subtype)


def manifest():
    with open(DATA / "ref" / "manifest.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- framing
def frames(x, size, hop, center=True, pad_mode="reflect"):
    if center:
        x = np.pad(x, size // 2, mode=pad_mode)
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def frame_level(x, size=1024, hop=256):
    """True frame RMS in dBFS (unwindowed), one value per STFT frame."""
    f = frames(x, size, hop)
    return 10.0 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)


def stft_mag(x, n_fft=1024, hop=256, window="hann"):
    w = np.hanning(n_fft) if window == "hann" else np.hamming(n_fft)
    return np.abs(np.fft.rfft(frames(x, n_fft, hop) * w, axis=1))


def stft_complex(x, n_fft=1024, hop=256):
    w = np.hanning(n_fft)
    return np.fft.rfft(frames(x, n_fft, hop) * w, axis=1)


def istft(spec, n_fft=1024, hop=256, length=None):
    w = np.hanning(n_fft)
    full = np.fft.irfft(spec, n=n_fft, axis=1)
    out = np.zeros(hop * (spec.shape[0] - 1) + n_fft)
    norm = np.zeros_like(out)
    for i in range(spec.shape[0]):
        out[i * hop:i * hop + n_fft] += full[i] * w
        norm[i * hop:i * hop + n_fft] += w * w
    out /= np.maximum(norm, 1e-8)
    if length is not None:
        out = out[:length]
    else:
        out = out[n_fft // 2:-(n_fft // 2)]
    return out


def db(x, floor=1e-10):
    return 10.0 * np.log10(np.maximum(x, floor))


# ---------------------------------------------------------------- spectrum
def mel_filterbank(n_mels=80, n_fft=1024, sr=SR, fmin=50.0, fmax=None):
    """HTK-style mel bank, matching the model's 80-band front end closely."""
    fmax = sr / 2 if fmax is None else fmax
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bank = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        left = (bins - lo) / max(mid - lo, 1e-9)
        right = (hi - bins) / max(hi - mid, 1e-9)
        bank[i] = np.clip(np.minimum(left, right), 0.0, None)
    return bank


def logmel(x, n_mels=80, n_fft=1024, hop=256):
    power = stft_mag(x, n_fft, hop) ** 2
    return db(power @ mel_filterbank(n_mels, n_fft).T)


def spectral_stats(mag, n_fft=1024, sr=SR):
    """Per-frame descriptive statistics of a magnitude spectrogram [T, F]."""
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    power = mag ** 2
    total = power.sum(axis=1) + 1e-12
    centroid = (power * f).sum(axis=1) / total
    norm = power / total[:, None]
    bandwidth = np.sqrt((norm * (f - centroid[:, None]) ** 2).sum(axis=1))
    cum = np.cumsum(power, axis=1)
    rolloff = f[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    flatness = np.exp(np.mean(np.log(mag + 1e-12), axis=1)) / (np.mean(mag, axis=1) + 1e-12)
    # band ratios relative to total power
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return (power[:, m].sum(axis=1) / total)
    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "low": band(0, 500),
        "mid": band(500, 4000),
        "high": band(4000, 8000),
        "air": band(8000, 12000),
        "log_total": db(total),
    }


def spectral_tilt(freq, level_db, lo=100.0, hi=8000.0):
    """Least-squares slope of the long-term spectrum, dB per octave."""
    m = (freq >= lo) & (freq <= hi) & np.isfinite(level_db)
    x = np.log2(freq[m] / lo)
    y = level_db[m]
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------- periodicity
def autocorr_f0(x, sr=SR, win=1024, hop=256, fmin=60.0, fmax=500.0, threshold=0.30):
    """Frame-wise F0 and normalized-autocorrelation voicing strength.

    Simple and dependency-free; good enough to separate voiced from unvoiced
    frames and to measure harmonic-to-noise ratio trends.
    """
    lo, hi = int(sr / fmax), int(sr / fmin)
    segs = frames(x, win, hop) * np.hanning(win)
    segs = segs - segs.mean(axis=1, keepdims=True)
    spec = np.fft.rfft(segs, n=2 * win, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :win]
    ac /= np.maximum(ac[:, :1], 1e-12)
    band = ac[:, lo:hi]
    lag = lo + np.argmax(band, axis=1)
    strength = band.max(axis=1)
    f0 = sr / lag
    voiced = strength > threshold
    return f0, strength, voiced


def hnr_from_strength(strength):
    r = np.clip(strength, 1e-4, 1 - 1e-4)
    return 10.0 * np.log10(r / (1.0 - r))


def cepstral_peak_prominence(x, sr=SR, win=1024, hop=256):
    """CPP in dB: peak of the real cepstrum in the pitch quefrency band over a
    fitted baseline. A standard dysphonia / breathiness measure."""
    segs = frames(x, win, hop) * np.hanning(win)
    logmag = np.log(np.abs(np.fft.rfft(segs, axis=1)) + 1e-8)
    ceps = np.fft.rfft(logmag, n=win, axis=1).real
    qlo, qhi = int(sr / 500.0), int(sr / 60.0)   # 48 .. 400 samples
    band = ceps[:, qlo:qhi]
    peak = band.max(axis=1)
    # remove a smoothed baseline (moving average of the cepstrum)
    k = 65
    kernel = np.ones(k) / k
    base = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, band)
    return peak - base.max(axis=1)


def modulation_spectrum(logmel_band, sr=SR, hop=256):
    """Temporal modulation spectrum of one band-energy track (dB, 1 value per
    frame). Returns (freqs_Hz, magnitude). Any fixed frame-rate artifact shows
    up as a spike at the generator's frame rate and its multiples."""
    v = logmel_band - logmel_band.mean()
    v = v * np.hanning(len(v))
    spec = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), hop / sr)
    return f, spec


# ---------------------------------------------------------------- report bits
def summarize(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {"mean": float(v.mean()), "std": float(v.std()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
