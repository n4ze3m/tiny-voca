"""Fit and validate a phoneme-class-adaptive spectral filter.

Classes come from the reference alignments (vowel / fricative / stop / nasal /
approximant / affricate / silence). A real post filter cannot see phoneme
labels, but it can estimate three acoustic classes from the signal: silence,
voiced, unvoiced. We therefore validate both

  * oracle class gains (upper bound), and
  * gains collapsed to silence / voiced / unvoiced (the deployable filter),

on a held-out half of the clips.

Usage: .venv/bin/python postproc/fit_class_filter.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = C.OUT
NFFT, HOP = 1024, 256
FREQ = np.fft.rfftfreq(NFFT, 1 / C.SR)
CLASSES = ["vowel", "fricative", "stop", "nasal", "approximant", "affricate", "silence"]
GROUP = {"vowel": "voiced", "nasal": "voiced", "approximant": "voiced",
         "fricative": "unvoiced", "stop": "unvoiced", "affricate": "unvoiced",
         "silence": "silence"}
MAX_DB = 5.0


def smooth_db(g, sigma=1.2):
    """Smooth a per-bin dB curve over log-frequency (in log2 bins)."""
    x = np.log2(np.maximum(FREQ, 20.0))
    grid = np.arange(x[0], x[-1] + 0.02, 0.02)
    v = np.interp(grid, x, g)
    k = np.exp(-0.5 * (np.arange(-int(3 * sigma / 0.02), int(3 * sigma / 0.02) + 1) * 0.02 / sigma) ** 2)
    k /= k.sum()
    v = np.convolve(np.pad(v, len(k) // 2, mode="edge"), k, mode="valid")
    return np.interp(x, grid, v)


def gain_from(r, g):
    r = r / (r[(FREQ > 80) & (FREQ < 12000)].sum() + 1e-12)
    g = g / (g[(FREQ > 80) & (FREQ < 12000)].sum() + 1e-12)
    gain = 10 * np.log10((r + 1e-12) / (g + 1e-12))
    gain = np.clip(smooth_db(gain), -MAX_DB, MAX_DB)
    gain[FREQ < 70] = 0.0
    return gain


def err(r, g):
    r = r / (r[(FREQ > 80) & (FREQ < 12000)].sum() + 1e-12)
    g = g / (g[(FREQ > 80) & (FREQ < 12000)].sum() + 1e-12)
    d = 10 * np.log10((g + 1e-12) / (r + 1e-12))
    m = (FREQ >= 80) & (FREQ < 12000)
    return float(np.sqrt(np.mean(d[m] ** 2)))


def apply_gain(spec, gain_db):
    return spec * 10 ** (gain_db / 10.0)


def main():
    data = pickle.load(open(OUT / "class_spectra.pkl", "rb"))
    result = {}
    for voice in ("F2", "M2"):
        items = data[voice]
        train = [x for i, x in enumerate(items) if i % 2 == 0]
        test = [x for i, x in enumerate(items) if i % 2 == 1]

        # ---- oracle per-class gains from the training half
        gains = {}
        for cls in CLASSES:
            r = [x[1][cls] for x in train if cls in x[1]]
            g = [x[2][cls] for x in train if cls in x[2]]
            if len(r) < 2 or len(g) < 2:
                gains[cls] = np.zeros(len(FREQ)); continue
            gains[cls] = gain_from(np.mean(r, 0), np.mean(g, 0))

        # ---- collapse to three deployable classes
        grouped = {}
        for group in ("voiced", "unvoiced", "silence"):
            members = [c for c in CLASSES if GROUP[c] == group and not np.all(gains[c] == 0)]
            if members:
                grouped[group] = np.mean([gains[c] for c in members], axis=0)
            else:
                grouped[group] = np.zeros(len(FREQ))

        def evaluate(which):
            per_class = {}
            for cls in CLASSES:
                before, after, orac = [], [], []
                for x in test:
                    if cls not in x[1] or cls not in x[2]:
                        continue
                    r, g = x[1][cls], x[2][cls]
                    before.append(err(r, g))
                    gg = grouped[GROUP[cls]] if which == "group" else gains[cls]
                    after.append(err(r, apply_gain(g, gg)))
                    orac.append(err(r, apply_gain(g, gains[cls])))
                if before:
                    per_class[cls] = {"n": len(before), "before": float(np.mean(before)),
                                      "after": float(np.mean(after)), "oracle": float(np.mean(orac))}
            return per_class

        group_eval = evaluate("group")
        oracle_eval = evaluate("oracle")
        b = np.mean([v["before"] for v in group_eval.values()])
        a = np.mean([v["after"] for v in group_eval.values()])
        o = np.mean([v["oracle"] for v in oracle_eval.values()])
        print(f"\n{voice}: held-out class-shape error  before {b:.3f} dB  "
              f"deployable {a:.3f} dB ({100*(1-a/b):.1f}% lower)  oracle {o:.3f} dB")
        for cls, v in group_eval.items():
            print(f"  {cls:12s} n={v['n']:2d}  {v['before']:.2f} -> {v['after']:.2f} "
                  f"({100*(1-v['after']/v['before']):+5.1f}%)  oracle {v['oracle']:.2f}")

        result[voice] = {"class_gain_db": {k: v.tolist() for k, v in gains.items()},
                         "group_gain_db": {k: v.tolist() for k, v in grouped.items()},
                         "eval_group": group_eval, "eval_oracle": oracle_eval}

    np.savez_compressed(OUT / "class_filters.npz",
                        freq=FREQ,
                        **{f"{voice}_{g}": np.array(v)
                           for voice, d in result.items()
                           for g, v in d["group_gain_db"].items()})
    json.dump(result, open(OUT / "class_filter.json", "w"), indent=1)
    print("\nwrote", OUT / "class_filter.json")


if __name__ == "__main__":
    main()
