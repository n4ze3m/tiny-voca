"""Apply the post filter to every held-out clip and report before/after metrics.

Usage: .venv/bin/python postproc/evaluate_filter.py
Writes data/filtered/*.wav, postproc/out/filter_metrics.json and compares
against the Qwen3-TTS targets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from tiny_voca import postfilter as pf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = C.OUT
FREQ = np.fft.rfftfreq(1024, 1 / C.SR)


def metrics(x):
    mag = C.stft_mag(x, 1024, 256)
    level = C.frame_level(x, 1024, 256)
    active = level > -45
    f0, strength, voiced = C.autocorr_f0(x, win=2048, hop=256, threshold=0.45)
    n = min(len(voiced), len(level))
    v = voiced[:n] & active[:n]
    quiet = level <= np.percentile(level, 10)
    cpp = C.cepstral_peak_prominence(x)
    st = C.spectral_stats(mag[active], 1024)
    return {
        "peak": float(np.abs(x).max()),
        "floor_db": float(np.mean(level[quiet])),
        "hnr": float(np.mean(C.hnr_from_strength(strength[:n][v]))),
        "cpp": float(np.mean(cpp[:min(len(cpp), n)][v[:min(len(cpp), n)]])),
        "flatness": float(np.mean(st["flatness"])),
        "centroid": float(np.mean(st["centroid"])),
        "hnr_frames": int(v.sum()),
    }


def band_error(ref, gen):
    """dB difference in the robust bands used to design the tone curve."""
    bands = {"low": (80, 315), "mid": (1250, 2500), "presence": (2500, 5000), "high": (5000, 10000)}

    def spec(y):
        mag = C.stft_mag(y, 1024, 256)
        lv = C.frame_level(y, 1024, 256)
        p = (mag[lv > -45] ** 2).mean(0)
        return p / p[(FREQ > 80) & (FREQ < 12000)].sum()
    r, g = spec(ref), spec(gen)
    return {k: float(10 * np.log10((g[(FREQ >= lo) & (FREQ < hi)].sum() + 1e-12) /
                                   (r[(FREQ >= lo) & (FREQ < hi)].sum() + 1e-12)))
            for k, (lo, hi) in bands.items()}


def main():
    rows = C.manifest()
    filtered = C.DATA / "filtered"
    filtered.mkdir(exist_ok=True)
    records = []
    for row in rows:
        ref = C.load(C.DATA / "ref" / row["ref_file"])
        gen = C.load(C.DATA / "gen" / f"{row['id']}.wav")
        out = pf.process(gen, voice=row["voice_id"], preset="clean")
        C.save(filtered / f"{row['id']}.wav", out)
        records.append({
            "id": row["id"], "voice": row["voice_id"],
            "before": metrics(gen), "after": metrics(out), "reference": metrics(ref),
            "band_before": band_error(ref, gen), "band_after": band_error(ref, out),
        })

    summary = {}
    for voice in ("F2", "M2"):
        rs = [r for r in records if r["voice"] == voice]
        summary[voice] = {}
        for key in ("floor_db", "hnr", "cpp", "flatness", "centroid", "peak"):
            summary[voice][key] = {
                "before": float(np.mean([r["before"][key] for r in rs])),
                "after": float(np.mean([r["after"][key] for r in rs])),
                "reference": float(np.mean([r["reference"][key] for r in rs])),
            }
        for band in ("low", "mid", "presence", "high"):
            summary[voice][f"band_{band}"] = {
                "before": float(np.mean([r["band_before"][band] for r in rs])),
                "after": float(np.mean([r["band_after"][band] for r in rs])),
            }
        print(f"\n{voice}")
        for key in ("floor_db", "hnr", "cpp", "flatness", "peak"):
            v = summary[voice][key]
            print(f"  {key:9s} gen {v['before']:8.2f} -> {v['after']:8.2f}   ref {v['reference']:8.2f}")
        for band in ("low", "mid", "presence", "high"):
            v = summary[voice][f"band_{band}"]
            print(f"  {band:9s} bias {v['before']:+6.2f} -> {v['after']:+6.2f} dB")

    json.dump({"summary": summary, "per_clip": records}, open(OUT / "filter_metrics.json", "w"), indent=1)

    # figure: pause-floor spectra before/after and a spectrogram pair
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, voice in zip(axes, ("F2", "M2")):
        for kind, color in (("gen", "tab:red"), ("filtered", "tab:green")):
            specs = []
            for r in records:
                if r["voice"] != voice:
                    continue
                x = C.load((C.DATA / ("gen" if kind == "gen" else "filtered")) / f"{r['id']}.wav")
                mag = C.stft_mag(x, 1024, 256)
                lv = C.frame_level(x, 1024, 256)
                q = lv <= np.percentile(lv, 10)
                specs.append(C.db((mag[q] ** 2).mean(0)))
            s = np.mean(specs, 0)
            ax.semilogx(FREQ[1:], s[1:], label=kind, color=color, lw=1)
        ax.set_title(f"{voice}: quiet-frame spectrum")
        ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "filter_floor.png", dpi=140); plt.close(fig)

    print("\nwrote", filtered, "and", OUT / "filter_metrics.json")


if __name__ == "__main__":
    main()
