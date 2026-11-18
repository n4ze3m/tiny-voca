"""DTW-aligned spectral comparison and static-filter fitting.

The reference (Qwen3-TTS target) and generated clip say the same sentence but
with different timing. DTW on log-mel aligns them so we can separate

  * a consistent spectral bias  -> what a static post filter can remove, from
  * frame-to-frame variance     -> what it cannot remove.

Writes postproc/out/align_*.png, fit.json and the fitted per-voice curves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = C.OUT
NFFT, HOP, NMEL = 1024, 256, 80
FREQ = np.fft.rfftfreq(NFFT, 1 / C.SR)
BANK = C.mel_filterbank(NMEL, NFFT)


def dtw(a, b, band=0.2):
    """Banded DTW between two [T, D] sequences. Returns index pairs."""
    na, nb = len(a), len(b)
    width = max(int(band * max(na, nb)), abs(na - nb) + 1, 8)
    inf = np.inf
    cost = np.full((na + 1, nb + 1), inf)
    cost[0, 0] = 0.0
    back = np.zeros((na + 1, nb + 1), dtype=np.uint8)
    for i in range(1, na + 1):
        lo = max(1, i - width)
        hi = min(nb, i + width)
        ai = a[i - 1]
        prev = cost[i - 1]
        row = cost[i]
        for j in range(lo, hi + 1):
            d = np.mean((ai - b[j - 1]) ** 2)
            # diagonal, vertical, horizontal
            s0 = prev[j - 1]
            s1 = prev[j]
            s2 = row[j - 1]
            if s0 <= s1 and s0 <= s2:
                row[j] = s0 + d; back[i, j] = 0
            elif s1 <= s2:
                row[j] = s1 + d; back[i, j] = 1
            else:
                row[j] = s2 + d; back[i, j] = 2
    i, j = na, nb
    pairs = []
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        step = back[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return np.asarray(pairs[::-1])


def mel_power(x):
    mag = C.stft_mag(x, NFFT, HOP)
    return (mag ** 2) @ BANK.T, mag ** 2


def main():
    rows = C.manifest()
    data = {"F2": [], "M2": []}
    for row in rows:
        name, voice = row["id"], row["voice_id"]
        ref = C.load(C.DATA / "ref" / row["ref_file"])
        gen = C.load(C.DATA / "gen" / f"{name}.wav")
        # loudness-normalise both: DTW/filter should not chase level
        ref = ref / (np.sqrt(np.mean(ref ** 2)) + 1e-12)
        gen = gen / (np.sqrt(np.mean(gen ** 2)) + 1e-12)
        rmel, rp = mel_power(ref)
        gmel, gp = mel_power(gen)
        rl, gl = C.db(rmel), C.db(gmel)
        # energy gate: compare only frames with speech in either side
        pairs = dtw(rl, gl)
        ri, gi = pairs[:, 0], pairs[:, 1]
        keep = (C.db(rp.sum(1))[ri] > 20) & (C.db(gp.sum(1))[gi] > 20)
        ri, gi = ri[keep], gi[keep]
        if len(ri) < 10:
            continue
        data[voice].append({
            "name": name, "ref_mel": rmel[ri], "gen_mel": gmel[gi],
            "ref_pow": rp[ri], "gen_pow": gp[gi],
        })

    fit = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for row_i, voice in enumerate(("F2", "M2")):
        items = data[voice]
        refmel = np.concatenate([d["ref_mel"] for d in items])
        genmel = np.concatenate([d["gen_mel"] for d in items])
        refpow = np.concatenate([d["ref_pow"] for d in items])
        genpow = np.concatenate([d["gen_pow"] for d in items])

        # per-frame optimal gain in each mel band, then average (ratio of means)
        gain = refmel.mean(0) / np.maximum(genmel.mean(0), 1e-12)
        gain_db = 10 * np.log10(gain)
        # smooth in mel index (envelope, not harmonic detail)
        k = np.hanning(9); k /= k.sum()
        smooth = np.convolve(np.pad(gain_db, 4, mode="edge"), k, mode="valid")
        fit[voice] = {"mel_gain_db": smooth.tolist()}

        ref_lsd = 10 * np.log10(refmel / np.maximum(genmel, 1e-12))
        # after static gain
        res = ref_lsd - smooth
        before = np.sqrt(np.mean(ref_lsd ** 2, axis=1))
        after = np.sqrt(np.mean(res ** 2, axis=1))
        # variance explained by the static curve
        v = 1 - np.var(res) / np.var(ref_lsd)

        fit[voice].update({
            "lsd_before_db": float(before.mean()),
            "lsd_after_db": float(after.mean()),
            "frame_var_explained": float(v),
            "aligned_frames": int(len(refmel)),
        })

        mel_centers = C.mel_filterbank(NMEL, NFFT) @ FREQ / (C.mel_filterbank(NMEL, NFFT).sum(1) + 1e-12)

        ax = axes[row_i, 0]
        ax.plot(mel_centers, smooth, label="fitted static gain")
        ax.plot(mel_centers, gain_db, alpha=0.35, label="raw band bias")
        ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log")
        ax.set_xlabel("Hz"); ax.set_ylabel("dB"); ax.grid(alpha=0.25)
        ax.set_title(f"{voice}: ref/gen band bias")

        ax = axes[row_i, 1]
        ax.plot(mel_centers, np.sqrt((ref_lsd ** 2).mean(0)), label="before", color="tab:red")
        ax.plot(mel_centers, np.sqrt((res ** 2).mean(0)), label="after static filter", color="tab:green")
        ax.set_xscale("log"); ax.set_xlabel("Hz"); ax.set_ylabel("RMS dB")
        ax.set_title(f"{voice}: per-band residual"); ax.grid(alpha=0.25); ax.legend()

        ax = axes[row_i, 2]
        ax.hist(before, bins=60, alpha=0.55, label=f"before {before.mean():.2f} dB")
        ax.hist(after, bins=60, alpha=0.55, label=f"after {after.mean():.2f} dB")
        ax.set_xlabel("frame log-spectral distance (dB)"); ax.legend(); ax.grid(alpha=0.25)
        ax.set_title(f"{voice}: frame LSD  (static explains {v*100:.1f}% var)")
        print(voice, fit[voice])

    # linear-frequency version of the gain, for an STFT mask
    for voice in ("F2", "M2"):
        mel_gain = np.array(fit[voice]["mel_gain_db"])
        lo = np.log2(np.maximum(FREQ, 1.0))
        mel_centers = (np.array([bank @ FREQ for bank in [C.mel_filterbank(NMEL, NFFT)][0]])
                       / C.mel_filterbank(NMEL, NFFT).sum(1))
        # interpolate mel gain onto FFT bins in log-frequency, hold at edges
        g = np.interp(lo, np.log2(mel_centers), mel_gain)
        fit[voice]["fft_gain_db"] = g.tolist()

    json.dump(fit, open(OUT / "fit.json", "w"), indent=1)
    fig.tight_layout(); fig.savefig(OUT / "align.png", dpi=140); plt.close(fig)
    print("wrote", OUT / "fit.json")


if __name__ == "__main__":
    main()
