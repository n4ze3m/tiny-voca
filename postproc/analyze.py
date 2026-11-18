"""Frequency / periodicity / modulation analysis: tiny-voca vs. held-out targets.

Writes machine-readable metrics and figures under postproc/out/.

Usage:  .venv/bin/python postproc/analyze.py
"""
from __future__ import annotations

import csv
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
OUT.mkdir(parents=True, exist_ok=True)

N_FFT, HOP = 1024, 256
FREQ = np.fft.rfftfreq(N_FFT, 1 / C.SR)

# Analysis bands (Hz)
BANDS = [(80, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 6000), (6000, 8000), (8000, 10000), (10000, 12000)]


ACTIVE_DBFS = -45.0


def active_mask(x):
    """Frames carrying speech energy, on an absolute dBFS scale (both model and
    reference are peak-normalised to roughly the same level)."""
    mag = C.stft_mag(x, N_FFT, HOP)
    level = C.frame_level(x, N_FFT, HOP)
    return level > ACTIVE_DBFS, level, mag


def ltas(mags, mask):
    """Long-term average power spectrum in dB, over selected frames."""
    p = (mags[mask] ** 2).mean(axis=0)
    return C.db(p)


def harmonic_profile(x):
    """F0-normalised harmonic amplitudes and inter-harmonic level.

    Uses an 85 ms analysis window and a 4x zero-padded FFT so harmonics are
    resolved for F0 >= 60 Hz. Returns (harmonic_db_re_f0, interharmonic_db, n).
    """
    win, hop, nfft = 2048, HOP, 8192
    f0, strength, voiced = C.autocorr_f0(x, win=win, hop=hop, fmin=60, fmax=500, threshold=0.45)
    segs = C.frames(x, win, hop)
    segs = (segs - segs.mean(axis=1, keepdims=True)) * np.hanning(win)
    spec = np.abs(np.fft.rfft(segs, n=nfft, axis=1))
    freq = np.fft.rfftfreq(nfft, 1 / C.SR)
    profiles, inters = [], []
    for i in np.flatnonzero(voiced):
        f = f0[i]
        if not (60 <= f <= 500):
            continue
        k_max = int(min(30, 10000 / f))
        if k_max < 5:
            continue
        db = 20 * np.log10(spec[i] + 1e-12)
        h = []
        for k in range(1, k_max + 1):
            center = k * f
            j = int(np.searchsorted(freq, center))
            if j < 1 or j >= len(freq) - 1:
                break
            # parabolic peak in log magnitude (clamped: flat bins would explode)
            a, b, c = db[j - 1], db[j], db[j + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
            d = float(np.clip(d, -0.5, 0.5))
            h.append(b - 0.25 * (a - c) * d)
        if len(h) < 5:
            continue
        h = np.asarray(h)
        profiles.append(h - h[0])
        mids = []
        for k in range(1, len(h)):
            for frac in (0.25, 0.5, 0.75):
                center = (k + frac) * f
                j = int(np.searchsorted(freq, center))
                mids.append(db[j])
        inters.append(float(np.median(mids)))
    if not profiles:
        return None
    n = min(map(len, profiles))
    return np.mean([p[:n] for p in profiles], axis=0), np.array(inters), n


def modulation(x, hop=32):
    """Average temporal-modulation spectrum of 8 log-mel bands.

    hop 32 samples = 750 Hz frame rate, so amplitude modulation up to 375 Hz
    (the vocoder's noise-branch update rate) is visible. Fixed frame-rate
    artefacts appear as narrow peaks independent of the speech content.
    """
    lm = C.logmel(x, n_mels=40, n_fft=512, hop=hop)   # [T, 40]
    # group into 8 bands of 5 mel channels
    tracks = lm.reshape(lm.shape[0], 8, 5).mean(axis=2)
    spectra = []
    for b in range(8):
        v = tracks[:, b]
        v = v - v.mean()
        v = v * np.hanning(len(v))
        s = np.abs(np.fft.rfft(v))
        s = s / (s.sum() + 1e-12)
        spectra.append(s)
    f = np.fft.rfftfreq(len(tracks), hop / C.SR)
    return f, np.mean(spectra, axis=0), tracks


def metrics(x):
    mag = C.stft_mag(x, N_FFT, HOP)
    mask, level, _ = active_mask(x)
    if mask.sum() < 4:
        mask[:] = True
    st = C.spectral_stats(mag[mask], N_FFT)
    f0, strength, voiced = C.autocorr_f0(x, win=2048, hop=HOP)
    # voicing is only meaningful where there is energy
    voiced = voiced & (level > ACTIVE_DBFS)
    hnr = C.hnr_from_strength(strength)
    cpp = C.cepstral_peak_prominence(x)
    # cpp frames are hop-aligned with level (same 1024/256 framing)
    n = min(len(cpp), len(level), len(voiced))
    v = voiced[:n]
    if v.sum() < 4:
        v = np.zeros(n, bool); v[:] = True
    # frame-to-frame log-mel roughness
    lm = C.logmel(x, n_mels=80, hop=HOP)
    delta = np.abs(np.diff(lm, axis=0)).mean(axis=1)
    # noise floor: quietest 10% of frames, true dBFS
    quiet = level <= np.percentile(level, 10)
    qmag = mag[:len(level)][quiet]
    floor = C.db((qmag ** 2).mean(axis=0)) if len(qmag) else np.full(len(FREQ), np.nan)
    row = {
        "seconds": len(x) / C.SR,
        "peak": float(np.abs(x).max()),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "crest": float(np.abs(x).max() / (np.sqrt(np.mean(x ** 2)) + 1e-12)),
        "dc": float(np.mean(x)),
        "voiced_frac": float(voiced.mean()),
        "f0_mean": float(np.mean(f0[voiced])) if voiced.sum() else np.nan,
        "f0_std": float(np.std(f0[voiced])) if voiced.sum() else np.nan,
        "f0_p10": float(np.percentile(f0[voiced], 10)) if voiced.sum() else np.nan,
        "f0_p90": float(np.percentile(f0[voiced], 90)) if voiced.sum() else np.nan,
        "hnr": float(np.mean(hnr[:n][v])),
        "cpp": float(np.mean(cpp[:n][v])),
        "flatness": float(np.mean(st["flatness"])),
        "centroid": float(np.mean(st["centroid"])),
        "bandwidth": float(np.mean(st["bandwidth"])),
        "rolloff": float(np.mean(st["rolloff"])),
        "tilt_db_oct": C.spectral_tilt(FREQ, C.db((mag[mask] ** 2).mean(axis=0))),
        "delta_lm": float(np.mean(delta)),
        "floor_db": float(np.mean(level[quiet])),
    }
    for lo, hi in BANDS:
        m = (FREQ >= lo) & (FREQ < hi)
        row[f"band_{lo}_{hi}"] = float(np.mean(
            (mag[mask][:, m] ** 2).sum(axis=1) / ((mag[mask] ** 2).sum(axis=1) + 1e-12)))
    return row, mag, mask, floor, cpp, voiced, st


def main():
    rows = C.manifest()
    records = []
    ltas_by_group = {"ref": {"F2": [], "M2": []}, "gen": {"F2": [], "M2": []}}
    profiles = {"ref": [], "gen": []}
    inter = {"ref": [], "gen": []}
    mods = {"ref": [], "gen": []}
    mod_f = np.arange(0.0, 375.0 + 0.5, 1.0)
    crops = []

    for row in rows:
        voice = row["voice_id"]
        for kind, directory in (("ref", C.DATA / "ref"), ("gen", C.DATA / "gen")):
            path = directory / row["ref_file"] if kind == "ref" else directory / f"{row['id']}.wav"
            if not path.exists():
                print("missing", path)
                continue
            x = C.load(path)
            m, mag, mask, floor, cpp, voiced, st = metrics(x)
            m["file"] = row["id"]
            m["voice"] = voice
            m["kind"] = kind
            m["text"] = row["text"]
            records.append(m)
            ltas_by_group[kind][voice].append(ltas(mag, mask))
            hp = harmonic_profile(x)
            if hp is not None:
                profiles[kind].append(hp[0])
                inter[kind].append(hp[1])
            f, ms, _ = modulation(x)
            mods[kind].append(np.interp(mod_f, f, ms, left=0.0, right=0.0))
            if row["id"] in ("fb6c75bb716b7ac1_F2", "12f0f0cc9c17d6c2_F2"):
                crops.append((row["id"], kind, x))

    # ---------------- CSV
    keys = [k for k in records[0] if k not in ("text",)]
    with open(OUT / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in keys})

    # ---------------- summary table
    def agg(kind, voice, key):
        v = [r[key] for r in records if r["kind"] == kind and r["voice"] == voice]
        return C.summarize(v)

    summary = {}
    for voice in ("F2", "M2"):
        for key in ("hnr", "cpp", "flatness", "centroid", "rolloff", "tilt_db_oct",
                    "delta_lm", "f0_mean", "voiced_frac", "floor_db", "crest"):
            summary.setdefault(key, {})[voice] = {
                "ref": agg("ref", voice, key), "gen": agg("gen", voice, key)}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

    # ---------------- LTAS plots
    for voice in ("F2", "M2"):
        ref = np.mean(ltas_by_group["ref"][voice], axis=0)
        gen = np.mean(ltas_by_group["gen"][voice], axis=0)
        # normalise both to their 200-1000 Hz mean so only shape matters
        m = (FREQ >= 200) & (FREQ <= 1000)
        refn, genn = ref - ref[m].mean(), gen - gen[m].mean()
        np.savez(OUT / f"ltas_{voice}.npz", freq=FREQ, ref=refn, gen=genn)
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
        ax[0].semilogx(FREQ[1:], refn[1:], label="Qwen3-TTS target", lw=1.4)
        ax[0].semilogx(FREQ[1:], genn[1:], label="tiny-voca", lw=1.4)
        ax[0].set_ylabel("dB (norm. 0.2-1 kHz)")
        ax[0].set_title(f"Long-term average spectrum, {voice}")
        ax[0].legend(); ax[0].grid(alpha=0.25)
        ax[1].semilogx(FREQ[1:], genn[1:] - refn[1:], color="crimson", lw=1.2)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_ylabel("gen - ref (dB)")
        ax[1].set_xlabel("Hz"); ax[1].grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(OUT / f"ltas_{voice}.png", dpi=140); plt.close(fig)

    # combined diff plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for voice, color in (("F2", "tab:blue"), ("M2", "tab:orange")):
        d = np.load(OUT / f"ltas_{voice}.npz")
        ax.semilogx(d["freq"][1:], d["gen"][1:] - d["ref"][1:], label=voice, color=color)
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
    ax.set_title("tiny-voca minus Qwen3-TTS target, long-term spectrum")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(OUT / "ltas_diff.png", dpi=140); plt.close(fig)

    # ---------------- harmonic profile
    def profile_mean(kind):
        n = min(len(p) for p in profiles[kind])
        return np.mean([p[:n] for p in profiles[kind]], axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, color in (("ref", "tab:blue"), ("gen", "tab:red")):
        arr = profile_mean(kind)
        ax.plot(np.arange(1, len(arr) + 1), arr, marker="o", ms=3, label=kind, color=color)
    ax.set_xlabel("harmonic number"); ax.set_ylabel("dB re. F0")
    ax.set_title("Average harmonic amplitude profile (voiced frames)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "harmonic_profile.png", dpi=140); plt.close(fig)
    hm = {}
    for kind in profiles:
        arr = profile_mean(kind)
        hm[kind] = float(-np.polyfit(np.arange(1, len(arr) + 1), arr, 1)[0])
    it = {k: float(np.mean(np.concatenate(v))) for k, v in inter.items()}

    # ---------------- modulation spectrum
    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, color in (("ref", "tab:blue"), ("gen", "tab:red")):
        m = np.mean(mods[kind], axis=0)
        ax.plot(mod_f, m, label=kind, color=color, lw=1.2)
    for rate in (93.75, 187.5, 375.0):
        ax.axvline(rate, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 380); ax.set_xlabel("modulation frequency (Hz)")
    ax.set_ylabel("normalised magnitude")
    ax.set_title("Average temporal modulation spectrum (8 mel bands)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(OUT / "modulation.png", dpi=140); plt.close(fig)
    mod_peak = {}
    for kind in mods:
        m = np.mean(mods[kind], axis=0)
        # power in narrow bands around generator frame rates
        mod_peak[kind] = {str(rate): float(np.sum(m[(mod_f > rate - 3) & (mod_f < rate + 3)]))
                          for rate in (93.75, 187.5, 375.0)}

    # ---------------- spectrogram pairs
    pairs = {}
    for name, kind, x in crops:
        pairs.setdefault(name, {})[kind] = x
    pairs = [(name, v) for name, v in pairs.items() if "ref" in v and "gen" in v]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(11, 2.4 * len(pairs)), squeeze=False)
    for i, (name, v) in enumerate(pairs):
        allv = np.concatenate([C.logmel(v[k], n_mels=80, hop=HOP).ravel() for k in ("ref", "gen")])
        lo, hi = np.percentile(allv, 10), np.percentile(allv, 99)
        for j, kind in enumerate(("ref", "gen")):
            lm = C.logmel(v[kind], n_mels=80, hop=HOP)
            axes[i, j].imshow(lm.T, origin="lower", aspect="auto",
                              extent=(0, len(v[kind]) / C.SR, 0, 80), cmap="magma",
                              vmin=lo, vmax=hi)
            axes[i, j].set_title(f"{name} {kind}")
            axes[i, j].set_ylabel("mel")
            axes[i, j].set_xlabel("s")
    fig.suptitle("log-mel spectrograms (same text, same voice)")
    fig.tight_layout(); fig.savefig(OUT / "spectrograms.png", dpi=140); plt.close(fig)

    # ---------------- steady vowel diagnostics
    steady = {}
    for path in sorted((C.DATA / "steady").glob("*.wav")):
        x = C.load(path)
        mag = C.stft_mag(x, 1024, 256)
        mask, level, _ = active_mask(x)
        if mask.sum() < 4:
            mask[:] = True
        steady[path.stem] = {
            "ltas": ltas(mag, mask), "level": level,
        }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, voice in enumerate(("F2", "M2")):
        ax = axes[i, 0]
        for path in sorted((C.DATA / "steady").glob(f"{voice}_*.wav")):
            ax.semilogx(FREQ[1:], steady[path.stem]["ltas"][1:], lw=1, label=path.stem)
        ax.set_title(f"steady vowels, {voice}"); ax.set_xlabel("Hz"); ax.set_ylabel("dB")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        ax = axes[i, 1]
        p = C.DATA / "steady" / f"{voice}_ɑ.wav"
        x = C.load(p)
        seg = x[int(1.0 * C.SR):int(1.2 * C.SR)]
        ax.plot(np.arange(len(seg)) / C.SR, seg, lw=0.7)
        ax.set_title(f"{voice} steady /ɑː/, 0.2 s"); ax.set_xlabel("s")
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "steady.png", dpi=140); plt.close(fig)

    # ---------------- print report data
    report = {
        "harmonic_slope_db_per_harmonic": hm,
        "interharmonic_level_db": it,
        "modulation_band_energy": mod_peak,
        "summary": summary,
    }
    json.dump(report, open(OUT / "report.json", "w"), indent=1)
    print(json.dumps({"harmonic_slope": hm, "interharmonic": it}, indent=1))
    for key in ("hnr", "cpp", "tilt_db_oct", "flatness", "delta_lm", "centroid", "floor_db"):
        print(f"\n{key}")
        for voice in ("F2", "M2"):
            r, g = summary[key][voice]["ref"], summary[key][voice]["gen"]
            print(f"  {voice}: ref {r['mean']:.3f}  gen {g['mean']:.3f}  (delta {g['mean']-r['mean']:+.3f})")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
