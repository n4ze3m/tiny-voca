"""Measure tonal ('buzzy') energy inside each phoneme class.

For every analysis frame we remove a wide spectral envelope with a median
filter across frequency and look at the positive residual: harmonic peaks and
narrow tones stand out, broadband fricative noise does not. This isolates the
vocoder's periodic excitation from legitimate noise.

Usage: .venv/bin/python postproc/tonality.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DIZIONE_DATA", str(Path(__file__).resolve().parent.parent / "weights" / "phonemizer"))
os.environ.setdefault("TINY_VOCA_FRONTEND", "own")

import numpy as np
from scipy.ndimage import median_filter

from tiny_voca.infer import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from phones import read_utterances, phone_class, ROOT, NFFT, HOP, FREQ

OUT = C.OUT
CLASSES = ["vowel", "fricative", "stop", "nasal", "approximant", "affricate", "silence"]


def frame_features(wave):
    mag = C.stft_mag(wave, NFFT, HOP)
    db = 20 * np.log10(mag + 1e-9)
    # wide envelope: median across ~400 Hz, independent of pitch
    env = median_filter(db, size=(1, 9), mode="nearest")
    resid = db - env
    peak = np.percentile(resid[:, (FREQ > 80) & (FREQ < 8000)], 99, axis=1)
    level = C.db((mag ** 2).sum(axis=1)) - 10 * np.log10(NFFT)
    flat = np.exp(np.mean(np.log(mag + 1e-9), axis=1)) / (np.mean(mag, axis=1) + 1e-9)
    f0, strength, voiced = C.autocorr_f0(wave, win=NFFT, hop=HOP, threshold=0)
    n = min(len(peak), len(strength), len(level))
    return peak[:n], strength[:n], flat[:n], level[:n]


def collect(wave, labels, durations):
    peak, strength, flat, level = frame_features(wave)
    lab = np.repeat(list(labels), np.asarray(durations))
    n = min(len(lab), len(peak))
    out = {}
    for i in range(n):
        cls = phone_class(lab[i])
        out.setdefault(cls, []).append((peak[i], strength[i], flat[i], level[i]))
    return {k: np.array(v) for k, v in out.items()}


def main():
    rows = C.manifest()
    ref_ann = {}
    for line in open(ROOT / "refs" / "annotations-v4-validation.jsonl"):
        item = json.loads(line)
        ref_ann[(item["text"], item["speaker"])] = item

    model = load_model(ROOT / "weights" / "tiny-voca.pt", "cpu")
    data = {"F2": {"ref": {}, "gen": {}}, "M2": {"ref": {}, "gen": {}}}
    for row in rows:
        key = (row["text"], row["voice_id"])
        if key not in ref_ann:
            continue
        voice = row["voice_id"]
        ann = ref_ann[key]
        ref = C.load(C.DATA / "ref" / row["ref_file"])
        gen = C.load(C.DATA / "gen" / f"{row['id']}.wav")
        for kind, wave, labels, durations in (
            ("ref", ref, ann["symbols"].split(), ann["durations"]),
            ("gen", gen, None, None),
        ):
            if kind == "gen":
                utts = read_utterances(model, row["text"], voice)
                parts = [collect(u["wave"].numpy(), u["symbols"], u["durations"]) for u in utts]
                feats = {}
                for p in parts:
                    for k, v in p.items():
                        feats.setdefault(k, []).append(v)
                feats = {k: np.concatenate(v) for k, v in feats.items()}
            else:
                feats = collect(wave, labels, durations)
            for k, v in feats.items():
                data[voice][kind].setdefault(k, []).append(v)

    print(f"{'class':12s} {'ref peak':>9s} {'gen peak':>9s} {'ref HNR':>8s} {'gen HNR':>8s} {'ref flat':>8s} {'gen flat':>8s}")
    summary = {}
    for voice in ("F2", "M2"):
        print("\n" + voice)
        summary[voice] = {}
        for cls in CLASSES:
            r = np.concatenate(data[voice]["ref"].get(cls, []))
            g = np.concatenate(data[voice]["gen"].get(cls, []))
            if len(r) < 10 or len(g) < 10:
                continue
            rp, rs, rf, _ = r.T
            gp, gs, gf, _ = g.T
            # only frames with real energy
            rp = rp[np.isfinite(rp)]; gp = gp[np.isfinite(gp)]
            row = {
                "ref_peak_db": float(np.mean(rp)), "gen_peak_db": float(np.mean(gp)),
                "ref_ac": float(np.mean(rs)), "gen_ac": float(np.mean(gs)),
                "ref_flat": float(np.mean(rf)), "gen_flat": float(np.mean(gf)),
                "ref_frames": int(len(r)), "gen_frames": int(len(g)),
            }
            summary[voice][cls] = row
            print(f"{cls:12s} {row['ref_peak_db']:9.2f} {row['gen_peak_db']:9.2f} "
                  f"{row['ref_ac']:8.3f} {row['gen_ac']:8.3f} {row['ref_flat']:8.3f} {row['gen_flat']:8.3f}")
    json.dump(summary, open(OUT / "tonality.json", "w"), indent=1)


if __name__ == "__main__":
    main()
