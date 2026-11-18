"""Regenerate the committed comparison assets.

Canonical generator for ``assets/*.wav``: same text, voice, seed, loudness and
24-bit format for every file.  The branch-specific stages are imported from the
installed package, so run with the project venv:

    .venv/bin/python postproc/make_assets.py
    .venv/bin/python postproc/make_assets.py --only natural
    .venv/bin/python postproc/make_assets.py --list
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("DIZIONE_DATA", str(Path(__file__).resolve().parent.parent / "weights" / "phonemizer"))
os.environ.setdefault("TINY_VOCA_FRONTEND", "own")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "postproc"))
import common as C  # noqa: E402
from tiny_voca import hf_synth as HS, humanize as H, naturalize as NL, postfilter as pf  # noqa: E402
from tiny_voca.infer import load_model, speak_text  # noqa: E402

ASSETS = ROOT / "assets"
TEXT = ("The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, "
        "and the old lighthouse keeper counted the ships each evening.")
TARGET_RMS = -18.5

ORDER = [
    "stock", "filtered", "smooth", "warm", "human", "hfsynth",
    "warm_m2", "hfsynth_m2",
    "natural", "natural_soft", "natural_formant", "natural_dry", "natural_nopitch",
]


def metrics(x):
    mag = C.stft_mag(x, 1024, 256)
    level = C.frame_level(x)
    active = level > -45
    quiet = level <= np.percentile(level, 10)
    f0, strength, voiced = C.autocorr_f0(x, win=2048, hop=256, threshold=0.5)
    n = min(len(voiced), len(level))
    v = voiced[:n] & active[:n]
    mag = mag[:n]
    freq = np.fft.rfftfreq(1024, 1 / C.SR)

    def flat(lo, hi):
        z = mag[:, (freq >= lo) & (freq < hi)]
        z = z[v]
        if not len(z):
            return float("nan")
        return float(np.mean(np.exp(np.mean(np.log(z + 1e-12), 1)) /
                             (np.mean(z, 1) + 1e-12)))
    return {
        "floor_db": float(np.mean(level[quiet])),
        "dynamic_db": float(np.percentile(level[active], 90) - np.percentile(level[active], 10)),
        "hf_flat_4_8k": flat(4000, 8000),
        "hf_flat_8_12k": flat(8000, 12000),
        "peak": float(np.abs(x).max()),
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)),
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate assets/*.wav")
    parser.add_argument("--only", nargs="+", choices=ORDER,
                        help="Generate only the named files (default: all)")
    parser.add_argument("--list", action="store_true", help="List asset names and exit")
    args = parser.parse_args()
    if args.list:
        print("\n".join(ORDER))
        return
    names = args.only or ORDER
    ASSETS.mkdir(exist_ok=True)

    model = load_model(ROOT / "weights" / "tiny-voca.pt", "cpu")
    raw, _ = speak_text(model, TEXT, "F2", seed=0, peak=None)
    raw = raw.numpy()
    raw_pitch, _ = speak_text(model, TEXT, "F2", seed=0, pitch_shift=0.07, peak=None)
    raw_pitch = raw_pitch.numpy()
    raw_m2, _ = speak_text(model, TEXT, "M2", seed=0, peak=None)
    raw_m2 = raw_m2.numpy()

    def make(name):
        if name == "stock":
            return raw, "F2"
        if name == "filtered":
            return pf.process(raw, voice="F2", preset="clean", target_rms=TARGET_RMS), "F2"
        if name == "smooth":
            return pf.process(raw, voice="F2", preset="smooth", target_rms=TARGET_RMS), "F2"
        if name == "warm":
            return pf.process(raw, voice="F2", preset="warm", target_rms=TARGET_RMS), "F2"
        if name == "human":
            return H.process(raw, voice="F2", preset="human", target_rms=TARGET_RMS), "F2"
        if name == "hfsynth":
            return HS.process(raw, voice="F2", preset="hf", base="warm", target_rms=TARGET_RMS), "F2"
        if name == "warm_m2":
            return pf.process(raw_m2, voice="M2", preset="warm", target_rms=TARGET_RMS), "M2"
        if name == "hfsynth_m2":
            return HS.process(raw_m2, voice="M2", preset="hf", base="warm", target_rms=TARGET_RMS), "M2"
        if name == "natural":
            return NL.process(raw_pitch, voice="F2", preset="natural"), "F2"
        if name == "natural_soft":
            return NL.process(raw_pitch, voice="F2", preset="soft"), "F2"
        if name == "natural_formant":
            return NL.process(raw_pitch, voice="F2", preset="formant"), "F2"
        if name == "natural_dry":
            return NL.process(raw_pitch, voice="F2", preset="dry"), "F2"
        if name == "natural_nopitch":
            return NL.process(raw, voice="F2", preset="natural"), "F2"
        raise ValueError(name)

    report = {}
    for name in names:
        wave, voice = make(name)
        wave = pf.limit_peak(np.asarray(wave, dtype=np.float64), 0.99)
        path = ASSETS / f"tiny_voca_{name}.wav"
        import soundfile as sf
        sf.write(path, wave, C.SR, subtype="PCM_24")
        row = metrics(wave)
        row["voice"] = voice
        report[name] = row
        print(f"{name:16s} {voice}  floor={row['floor_db']:6.1f}  dyn={row['dynamic_db']:5.1f}  "
              f"hf4-8={row['hf_flat_4_8k']:.3f}  hf8-12={row['hf_flat_8_12k']:.3f}  peak={row['peak']:.3f}")

    metrics_path = ASSETS / "metrics.json"
    previous = {}
    if metrics_path.exists() and args.only:
        previous = json.loads(metrics_path.read_text())
    previous.update(report)
    metrics_path.write_text(json.dumps(previous, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {len(names)} file(s) to {ASSETS}")


if __name__ == "__main__":
    main()
