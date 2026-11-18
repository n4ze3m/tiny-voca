"""Per-phoneme spectral comparison using the reference forced alignments.

Regenerates each validation clip while recording the model's predicted phone
durations, then compares average spectra per phone class. Writes
postproc/out/phones.json and phone_heatmap.png.

Usage: .venv/bin/python postproc/phones.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("DIZIONE_DATA", str(Path(__file__).resolve().parent.parent / "weights" / "phonemizer"))
os.environ.setdefault("TINY_VOCA_FRONTEND", "own")

import numpy as np
import torch

from tiny_voca.infer import load_model, sentences, finished, shorter, _voice
from tiny_voca.text import phonemize, NothingToPronounce, CONSONANTS, VOWELS, SPECIALS, PUNCTUATION
from tiny_voca.model import UtteranceTooLong

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = C.OUT
NFFT, HOP = 1024, 256
FREQ = np.fft.rfftfreq(NFFT, 1 / C.SR)
EDGES = [80, 160, 315, 630, 1250, 2500, 5000, 10000, 12000]


def phone_class(symbol):
    if symbol in SPECIALS or symbol in PUNCTUATION or symbol in (" ", ""):
        return "silence"
    if symbol in VOWELS:
        return "vowel"
    if symbol in ("s", "z", "ʃ", "ʒ", "f", "v", "θ", "ð", "h"):
        return "fricative"
    if symbol in ("p", "b", "t", "d", "k", "ɡ", "ʔ"):
        return "stop"
    if symbol in ("m", "n", "ŋ"):
        return "nasal"
    if symbol in ("l", "ɹ", "j", "w"):
        return "approximant"
    if symbol in ("tʃ", "dʒ"):
        return "affricate"
    return "other"


def read_utterances(model, text, voice, seed=0):
    """Mirror speak_text but keep (symbols, durations) for each model call."""
    limit = model.config.max_tokens - 12
    result = []

    def read(piece):
        try:
            phones = phonemize(piece)
        except NothingToPronounce:
            return
        if len(phones) <= limit:
            try:
                wave, details = _voice(model, phones, voice, 1.0, 0.0, seed + len(result), 1.0)
            except UtteranceTooLong:
                pass
            else:
                durations = details["durations"][0, :len(phones)].tolist()
                result.append({"symbols": list(phones.symbols), "durations": durations, "wave": wave})
                return
        for part in shorter(piece):
            read(part)

    for sentence in sentences(finished(text)):
        read(sentence)
    return result


def spectra_by_class(wave, symbols, durations, min_frames=2):
    mag = C.stft_mag(wave, NFFT, HOP)
    power = mag ** 2
    labels = np.repeat(list(symbols), np.asarray(durations))
    n = min(len(labels), len(power))
    out = {}
    for i in range(n):
        cls = phone_class(labels[i])
        out.setdefault(cls, []).append(power[i])
    return {k: np.mean(v, axis=0) for k, v in out.items() if len(v) >= min_frames}


def main():
    rows = C.manifest()
    ref_ann = {}
    for line in open(ROOT / "refs" / "annotations-v4-validation.jsonl"):
        item = json.loads(line)
        ref_ann[(item["text"], item["speaker"])] = item

    model = load_model(ROOT / "weights" / "tiny-voca.pt", "cpu")
    data = {"F2": [], "M2": []}
    for row in rows:
        key = (row["text"], row["voice_id"])
        if key not in ref_ann:
            print("no annotation", key[0][:30])
            continue
        ann = ref_ann[key]
        ref = C.load(C.DATA / "ref" / row["ref_file"])
        gen = C.load(C.DATA / "gen" / f"{row['id']}.wav")
        ref_cls = spectra_by_class(ref, ann["symbols"].split(), ann["durations"])
        utts = read_utterances(model, row["text"], row["voice_id"])
        gen_cls = {}
        for u in utts:
            for k, v in spectra_by_class(u["wave"].numpy(), u["symbols"], u["durations"]).items():
                gen_cls.setdefault(k, []).append(v)
        gen_cls = {k: np.mean(v, axis=0) for k, v in gen_cls.items()}
        data[row["voice_id"]].append((row["id"], ref_cls, gen_cls))
        print(row["id"], list(ref_cls), list(gen_cls))

    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    classes = ["vowel", "fricative", "stop", "nasal", "approximant", "affricate", "silence"]
    for ax, voice in zip(axes[:2], ("F2", "M2")):
        table = []
        for cls in classes:
            r = [d[1][cls] for d in data[voice] if cls in d[1]]
            g = [d[2][cls] for d in data[voice] if cls in d[2]]
            if not r or not g:
                table.append([np.nan] * (len(EDGES) - 1)); continue
            r = np.mean(r, axis=0); g = np.mean(g, axis=0)
            # remove within-class level: compare spectral shape only
            r = r / (r[(FREQ > 80) & (FREQ < 12000)].sum() + 1e-12)
            g = g / (g[(FREQ > 80) & (FREQ < 12000)].sum() + 1e-12)
            row = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = (FREQ >= lo) & (FREQ < hi)
                row.append(10 * np.log10((g[m].sum() + 1e-12) / (r[m].sum() + 1e-12)))
            table.append(row)
        out[voice] = {"classes": classes, "bands": EDGES, "gen_minus_ref_db": table}
        im = ax.imshow(np.array(table), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xticks(range(len(EDGES) - 1)); ax.set_xticklabels([f"{lo/1000:g}k" for lo in EDGES[:-1]])
        ax.set_title(f"{voice}: spectral shape, gen - ref (dB)")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # consonant-to-vowel level ratio: are consonants too quiet in absolute terms?
    ax = axes[2]
    width = 0.35
    for k, voice in enumerate(("F2", "M2")):
        def ratio(cls, which):
            vals = []
            for d in data[voice]:
                if cls in d[1] and "vowel" in d[1]:
                    if which == 1:
                        vals.append(10 * np.log10((d[1][cls].sum() + 1e-12) / (d[1]["vowel"].sum() + 1e-12)))
                    elif cls in d[2]:
                        vals.append(10 * np.log10((d[2][cls].sum() + 1e-12) / (d[2]["vowel"].sum() + 1e-12)))
            return np.mean(vals)
        cons = [c for c in classes if c not in ("vowel", "silence")]
        for j, cls in enumerate(cons):
            r, g = ratio(cls, 1), ratio(cls, 2)
            ax.bar(j + (k - 0.5) * width, g - r, width, label=voice if j == 0 else None,
                   color=f"C{k}")
        out[voice]["consonant_minus_vowel_ref"] = {c: ratio(c, 1) for c in cons}
        out[voice]["consonant_minus_vowel_gen"] = {c: ratio(c, 2) for c in cons}
    ax.set_xticks(range(len(cons))); ax.set_xticklabels(cons, rotation=20)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("gen - ref (dB)")
    ax.set_title("consonant level relative to vowel")
    ax.grid(alpha=0.25, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "phone_heatmap.png", dpi=140); plt.close(fig)
    json.dump(out, open(OUT / "phones.json", "w"), indent=1)
    pickle.dump(data, open(OUT / "class_spectra.pkl", "wb"))
    for voice in ("F2", "M2"):
        print("\n", voice)
        for cls, row in zip(classes, out[voice]["gen_minus_ref_db"]):
            print(f"  {cls:12s}", " ".join("  nan" if not np.isfinite(v) else f"{v:+5.1f}" for v in row))
        print("  level vs vowel (ref -> gen):")
        for c in cons:
            print(f"    {c:12s} {out[voice]['consonant_minus_vowel_ref'][c]:+5.1f} -> "
                  f"{out[voice]['consonant_minus_vowel_gen'][c]:+5.1f} dB")


if __name__ == "__main__":
    main()
