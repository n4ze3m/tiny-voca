"""Fit a per-voice spectral correction filter and validate it on held-out clips.

Target:  the Qwen3-TTS reference clips that tiny-voca was trained to imitate.
Design:  1/6-octave smoothed, shrunk, regularised log-spectral gain, converted
         to a linear-phase FIR.  Fitting and evaluation use disjoint clips so
         the reported numbers are honest.

Usage:  .venv/bin/python postproc/fit_filter.py
Writes: postproc/out/filter.json, filter_F2.npy, filter_M2.npy, fit_eval.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import firwin2, freqz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = C.OUT
NFFT, HOP = 1024, 256
FREQ = np.fft.rfftfreq(NFFT, 1 / C.SR)
EDGES = np.array([40, 60, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12000])
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
TAPS = 513
RIDGE = 0.35          # shrink weak evidence toward 0 dB
SMOOTH = 1.1          # gaussian sigma in bands
MAX_DB = 4.0


def spectrum(x, mask=None):
    mag = C.stft_mag(x, NFFT, HOP)
    power = mag ** 2
    level = C.frame_level(x, NFFT, HOP)
    if mask is None:
        mask = level > -45.0
    if mask.sum() < 8:
        mask = np.ones(len(level), bool)
    p = power[mask].mean(axis=0)
    total = p[(FREQ > 80) & (FREQ < 12000)].sum()
    return p / (total + 1e-12)


def bandify(spec):
    return np.array([spec[(FREQ >= lo) & (FREQ < hi)].sum() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def smooth_gain(g_db):
    k = np.exp(-0.5 * (np.arange(-4, 5) / SMOOTH) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(g_db, 4, mode="edge"), k, mode="valid")


def fit_curve(ref_specs, gen_specs):
    """Shrunk, smoothed per-band gain from paired spectra of one voice."""
    r = np.log(np.mean([bandify(s) for s in ref_specs], axis=0) + 1e-12)
    g = np.log(np.mean([bandify(s) for s in gen_specs], axis=0) + 1e-12)
    # paired difference per clip -> standard error of the mean
    d = np.array([np.log(bandify(a) + 1e-12) - np.log(bandify(b) + 1e-12)
                  for a, b in zip(ref_specs, gen_specs)])
    mean = d.mean(axis=0)
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.zeros_like(mean)
    shrink = 1.0 / (1.0 + (RIDGE * se) ** 2 * len(d))
    gain = smooth_gain(mean * shrink) * 10.0 / np.log(10)   # natural log -> dB
    gain = np.clip(gain, -MAX_DB, MAX_DB)
    # low/high ends: no correction below 60 Hz, hold the last value above 10k
    gain[EDGES[:-1] < 60] = 0.0
    return gain, se * 10.0 / np.log(10)


def to_fir(gain_db):
    """Interpolate the band curve onto an FFT grid and design a linear-phase FIR."""
    f = np.linspace(0, C.SR / 2, 2049)
    g = np.interp(np.log2(np.maximum(f, 1.0)), np.log2(CENTERS), gain_db,
                  left=0.0, right=gain_db[-1])
    g = 10 ** (g / 20.0)
    g[f > 11800] = g[np.argmin(np.abs(f - 11800))]
    taps = firwin2(TAPS, f / (C.SR / 2), g, window="hann")
    return taps


def apply_fir(x, taps):
    n = len(x)
    y = np.convolve(x, taps, mode="full")
    delay = (len(taps) - 1) // 2
    return y[delay:delay + n]


def ltAS_error(ref, gen, apply=None):
    """RMS dB error in third-octave bands, level-normalised per clip."""
    r = bandify(spectrum(ref))
    g = bandify(spectrum(apply(gen) if apply else gen))
    # only compare shape: normalise both to the 300-3000 Hz band total
    sel = (CENTERS >= 300) & (CENTERS <= 3000)
    r = r / r[sel].sum(); g = g / g[sel].sum()
    return float(np.sqrt(np.mean((10 * np.log10((g + 1e-12) / (r + 1e-12))) ** 2)))


def main():
    rows = C.manifest()
    by_voice = {"F2": [], "M2": []}
    for row in rows:
        ref = C.load(C.DATA / "ref" / row["ref_file"])
        gen = C.load(C.DATA / "gen" / f"{row['id']}.wav")
        by_voice[row["voice_id"]].append({
            "name": row["id"], "ref": ref, "gen": gen,
            "ref_spec": spectrum(ref), "gen_spec": spectrum(gen),
        })

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, voice in enumerate(("F2", "M2")):
        items = by_voice[voice]
        # stratified split: alternating clips (each source text has F2 and M2)
        train = [x for j, x in enumerate(items) if j % 2 == 0]
        test = [x for j, x in enumerate(items) if j % 2 == 1]
        gain, se = fit_curve([x["ref_spec"] for x in train], [x["gen_spec"] for x in train])
        taps = to_fir(gain)

        def proc(y, taps=taps):
            return apply_fir(y, taps)

        base = np.mean([ltAS_error(x["ref"], x["gen"]) for x in test])
        after = np.mean([ltAS_error(x["ref"], x["gen"], proc) for x in test])
        ref_after = np.mean([ltAS_error(x["ref"], x["ref"], proc) for x in test])

        # systematic (mean) bias: the part a static filter is meant to remove
        def mean_band(items, key, proc=None):
            vals = [bandify(spectrum(proc(x[key]) if proc else x[key])) for x in items]
            v = np.mean(vals, axis=0)
            sel = (CENTERS >= 300) & (CENTERS <= 3000)
            return v / v[sel].sum()
        rmean, gmean, gmean_after = (mean_band(test, "ref"), mean_band(test, "gen"),
                                     mean_band(test, "gen", proc))
        def bias_err(g):
            return float(np.sqrt(np.mean((10 * np.log10((g + 1e-12) / (rmean + 1e-12))) ** 2)))
        bias_before, bias_after = bias_err(gmean), bias_err(gmean_after)

        # full-data curve for the released filter
        gain_all, se_all = fit_curve([x["ref_spec"] for x in items], [x["gen_spec"] for x in items])
        taps_all = to_fir(gain_all)
        results[voice] = {
            "gain_db": gain_all.tolist(), "se_db": se_all.tolist(),
            "bands_hz": EDGES.tolist(), "taps": taps_all.tolist(),
            "heldout_before_db": float(base), "heldout_after_db": float(after),
            "heldout_reference_after_db": float(ref_after),
            "heldout_bias_before_db": bias_before, "heldout_bias_after_db": bias_after,
            "fitted_on": [x["name"] for x in items],
        }
        print(f"{voice}: held-out per-clip LTAS error {base:.3f} -> {after:.3f} dB; "
              f"systematic bias {bias_before:.3f} -> {bias_after:.3f} dB "
              f"({100*(1-bias_after/bias_before):.1f}% lower); ref through filter {ref_after:.3f} dB")

        w, h = freqz(taps_all, worN=2048, fs=C.SR)
        axes[0, i].semilogx(w, 20 * np.log10(np.abs(h) + 1e-12), label="fitted FIR")
        axes[0, i].errorbar(CENTERS, gain, yerr=se, fmt="o", ms=3, lw=0.8, label="bands (train)")
        axes[0, i].axhline(0, color="k", lw=0.6); axes[0, i].set_xscale("log")
        axes[0, i].set_title(f"{voice}: correction filter")
        axes[0, i].set_ylabel("dB"); axes[0, i].set_xlabel("Hz")
        axes[0, i].grid(alpha=0.25); axes[0, i].legend()

        # effect on the average spectrum
        def avg_spec(items, key, proc=None):
            vals = [spectrum(proc(x[key]) if proc else x[key]) for x in items]
            return np.mean(vals, axis=0)
        r = avg_spec(items, "ref")
        g = avg_spec(items, "gen")
        ga = avg_spec(items, "gen", proc)
        axes[1, i].semilogx(FREQ[1:], 10 * np.log10(g[1:] / (r[1:] + 1e-12)), label="before", alpha=0.6)
        axes[1, i].semilogx(FREQ[1:], 10 * np.log10(ga[1:] / (r[1:] + 1e-12)), label="after", alpha=0.8)
        axes[1, i].axhline(0, color="k", lw=0.6); axes[1, i].set_xscale("log")
        axes[1, i].set_ylim(-6, 6)
        axes[1, i].set_title(f"{voice}: gen - ref spectrum")
        axes[1, i].set_xlabel("Hz"); axes[1, i].set_ylabel("dB")
        axes[1, i].grid(alpha=0.25); axes[1, i].legend()

        np.save(OUT / f"filter_{voice}.npy", taps_all)

    json.dump(results, open(OUT / "filter.json", "w"), indent=1)
    fig.tight_layout(); fig.savefig(OUT / "fit_eval.png", dpi=140); plt.close(fig)
    print("wrote", OUT / "filter.json")


if __name__ == "__main__":
    main()
