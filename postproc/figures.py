"""Report figures for the post-processing study."""
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


def main():
    # 1. filter curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, voice in zip(axes, ("F2", "M2")):
        for kind, color in (("voiced", "tab:blue"), ("unvoiced", "tab:orange"), ("silence", "tab:gray")):
            g = pf.tone_gain(voice, kind)
            ax.semilogx(FREQ[1:], g[1:], label=kind, color=color)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{voice}: tone correction")
        ax.set_xlabel("Hz"); ax.grid(alpha=0.25); ax.legend()
    axes[0].set_ylabel("dB")
    fig.tight_layout(); fig.savefig(OUT / "filter_curves.png", dpi=140); plt.close(fig)

    # 2. before/after spectrogram of a real clip
    name = "128c0ba2c4e644ad_F2"
    before = C.load(C.DATA / "gen" / f"{name}.wav")
    after = C.load(C.DATA / "filtered" / f"{name}.wav")
    lm_b = C.logmel(before, n_mels=80, hop=256)
    lm_a = C.logmel(after, n_mels=80, hop=256)
    lo = min(np.percentile(lm_b, 5), np.percentile(lm_a, 5))
    hi = max(np.percentile(lm_b, 99), np.percentile(lm_a, 99))
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, lm, title in ((axes[0], lm_b, "before"), (axes[1], lm_a, "after (clean)")):
        ax.imshow(lm.T, origin="lower", aspect="auto", extent=(0, len(before) / C.SR, 0, 80),
                  cmap="magma", vmin=lo, vmax=hi)
        ax.set_title(title); ax.set_ylabel("mel")
    axes[1].set_xlabel("s")
    fig.suptitle(f"{name}: log-mel spectrogram")
    fig.tight_layout(); fig.savefig(OUT / "before_after.png", dpi=140); plt.close(fig)

    # 3. band bias bars
    data = json.load(open(OUT / "filter_metrics.json"))
    bands = ["low", "mid", "presence", "high"]
    labels = ["80-315", "1.25-2.5k", "2.5-5k", "5-10k"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, voice in zip(axes, ("F2", "M2")):
        s = data["summary"][voice]
        before = [s[f"band_{b}"]["before"] for b in bands]
        after = [s[f"band_{b}"]["after"] for b in bands]
        x = np.arange(len(bands))
        ax.bar(x - 0.2, before, 0.4, label="before", color="tab:red")
        ax.bar(x + 0.2, after, 0.4, label="after", color="tab:green")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
        ax.set_title(f"{voice}: generated - reference band bias")
        ax.set_ylabel("dB"); ax.grid(alpha=0.25, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "bias_bars.png", dpi=140); plt.close(fig)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
