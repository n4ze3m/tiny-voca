"""Generate tiny-voca audio matched to the held-out reference clips.

Pipeline, all inference only:
  data/ref/manifest.json  ->  data/gen/<id>.wav       (same text, same voice)
  data/gen/steady_*.wav                               (quasi-stationary vowels,
                                                       for vocoder diagnostics)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DIZIONE_DATA", str(Path(__file__).resolve().parent.parent / "weights" / "phonemizer"))
os.environ.setdefault("TINY_VOCA_FRONTEND", "own")

import numpy as np
import soundfile as sf
import torch

from tiny_voca.infer import load_model, speak_text
from tiny_voca.text import SYMBOL_TO_ID

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, SR, save, load  # noqa: E402

GEN = DATA / "gen"
STEADY = DATA / "steady"


def steady_vowel(model, symbol, voice, repeats=40, seed=0):
    """A long run of one vowel token: close to a stationary vocalization, so the
    vocoder's own spectrum (not the text/prosody) dominates the measurement."""
    token = SYMBOL_TO_ID[symbol]
    ids = [SYMBOL_TO_ID["<bos>"]] + [token] * repeats + [SYMBOL_TO_ID["<eos>"]]
    stress = [0] + [1] * repeats + [0]
    tokens = torch.tensor([ids])
    stress_t = torch.tensor([stress])
    speaker = torch.tensor([model.config.speakers.index(voice)])
    with torch.inference_mode():
        audio, lengths, details = model.synthesize(
            tokens, stress_t, torch.tensor([len(ids)]), speaker,
            speed=0.85, pitch_shift=0.0, seed=seed, pitch_range=1.0)
    return audio[0, :int(lengths[0])].cpu().numpy()


def main():
    model = load_model(ROOT / "weights" / "tiny-voca.pt", "cpu")
    GEN.mkdir(parents=True, exist_ok=True)
    STEADY.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(DATA / "ref" / "manifest.json"))
    for row in rows:
        name = row["id"]
        voice = row["voice_id"]
        text = row["text"]
        out = GEN / f"{name}.wav"
        if out.exists():
            print("skip", name)
            continue
        # peak=None: keep the raw model level for honest spectral comparison.
        wave, spoken = speak_text(model, text, voice, speed=1.0, seed=0, peak=None)
        x = wave.numpy().astype(np.float64)
        save(out, x)
        row["gen_file"] = out.name
        row["gen_seconds"] = len(x) / SR
        row["gen_peak"] = float(np.abs(x).max())
        print(f"{name} {voice} ref={row['duration_seconds']:.2f}s gen={row['gen_seconds']:.2f}s peak={row['gen_peak']:.3f}")

    for voice in ("F2", "M2"):
        for symbol in ("ɑː", "iː", "uː", "s"):
            path = STEADY / f"{voice}_{'s' if symbol == 's' else symbol.replace('ː','')}.wav"
            if path.exists():
                continue
            x = steady_vowel(model, symbol, voice)
            save(path, x)
            print("steady", path.name, f"{len(x)/SR:.2f}s", f"peak={np.abs(x).max():.3f}")

    json.dump(rows, open(DATA / "ref" / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
