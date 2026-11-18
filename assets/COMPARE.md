# Comparison assets

All files use the same text, speaker seed 0, and 24 kHz output:

> The quick brown fox jumps over the lazy dog. She sells seashells by the
> seashore, and the old lighthouse keeper counted the ships each evening.

F2 files use 7.95 s; M2 files use 7.50 s. Processed files are 24-bit and
loudness-matched to −18.5 dBFS RMS. `stock` is the raw peak-limited model
output. The full machine-readable measurements are in `metrics.json`.

| file | voice | chain | floor | p10–p90 | HF 4–8k | HF 8–12k | peak |
|---|---|---|---:|---:|---:|---:|---:|
| `tiny_voca_stock.wav` | F2 | raw model output | −71.8 | 18.7 | 0.700 | 0.654 | 0.898 |
| `tiny_voca_filtered.wav` | F2 | `clean` tone + de-hiss | −85.0 | 18.3 | 0.683 | 0.654 | 0.982 |
| `tiny_voca_smooth.wav` | F2 | `clean` + HF contrast 1.5 | −81.3 | 17.4 | 0.532 | 0.496 | 0.967 |
| `tiny_voca_warm.wav` | F2 | `warm`: smooth + body boost | −84.5 | 20.2 | 0.532 | 0.493 | 0.840 |
| `tiny_voca_human.wav` | F2 | `warm` + synthetic room/breaths/dynamics | −58.9 | 21.2 | 0.551 | 0.507 | 0.851 |
| `tiny_voca_hfsynth.wav` | F2 | `warm` + 7–12 kHz harmonic rebuild | −84.5 | 20.2 | 0.534 | 0.378 | 0.840 |
| `tiny_voca_warm_m2.wav` | M2 | `warm` | −84.8 | 19.9 | 0.574 | 0.495 | 0.883 |
| `tiny_voca_hfsynth_m2.wav` | M2 | `warm` + HF harmonic rebuild | −84.8 | 19.9 | 0.577 | 0.382 | 0.885 |
| `tiny_voca_natural.wav` | F2 | full chain, F2 `pitch_shift=0.07` | −58.1 | 21.4 | 0.540 | 0.405 | 0.825 |
| `tiny_voca_natural_soft.wav` | F2 | full chain, conservative settings | −59.4 | 21.1 | 0.535 | 0.411 | 0.844 |
| `tiny_voca_natural_formant.wav` | F2 | full chain, formant 0.96 | −58.2 | 21.6 | 0.556 | 0.372 | 0.810 |
| `tiny_voca_natural_dry.wav` | F2 | full chain, no room/breaths | −85.9 | 21.2 | 0.535 | 0.399 | 0.893 |
| `tiny_voca_natural_nopitch.wav` | F2 | full chain, no pitch calibration | −58.5 | 21.3 | 0.547 | 0.409 | 0.840 |

Human reference (LJSpeech, different speaker/recording): floor −56.7 dBFS,
p10–p90 22.0 dB, HF flatness 0.50/0.36.

## Generate

The canonical generator is:

```bash
.venv/bin/python postproc/make_assets.py          # all 13 files
.venv/bin/python postproc/make_assets.py --list   # names
```

Generate one file:

```bash
.venv/bin/python postproc/make_assets.py --only stock
.venv/bin/python postproc/make_assets.py --only filtered
.venv/bin/python postproc/make_assets.py --only smooth
.venv/bin/python postproc/make_assets.py --only warm
.venv/bin/python postproc/make_assets.py --only human
.venv/bin/python postproc/make_assets.py --only hfsynth
.venv/bin/python postproc/make_assets.py --only warm_m2
.venv/bin/python postproc/make_assets.py --only hfsynth_m2
.venv/bin/python postproc/make_assets.py --only natural
.venv/bin/python postproc/make_assets.py --only natural_soft
.venv/bin/python postproc/make_assets.py --only natural_formant
.venv/bin/python postproc/make_assets.py --only natural_dry
.venv/bin/python postproc/make_assets.py --only natural_nopitch
```

## Underlying commands

These are the equivalent module commands for the main stages. `make_assets.py`
is still the exact generator for the committed files (24-bit, RMS-matched,
seed 0).

```bash
TEXT="The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, and the old lighthouse keeper counted the ships each evening."

# raw source
python -m tiny_voca.infer --checkpoint weights/tiny-voca.pt --voice F2 \
  --text "$TEXT" --output raw.wav
python -m tiny_voca.infer --checkpoint weights/tiny-voca.pt --voice F2 \
  --text "$TEXT" --output raw_pitch.wav --pitch-shift 0.07

# clean / smooth / warm
python -m tiny_voca.postfilter --input raw.wav --output filtered.wav \
  --voice F2 --preset clean --target-rms -18.5 --subtype PCM_24
python -m tiny_voca.postfilter --input raw.wav --output smooth.wav \
  --voice F2 --preset smooth --target-rms -18.5 --subtype PCM_24
python -m tiny_voca.postfilter --input raw.wav --output warm.wav \
  --voice F2 --preset warm --target-rms -18.5 --subtype PCM_24

# room/breaths/dynamics
python -m tiny_voca.humanize --input raw.wav --output human.wav \
  --voice F2 --preset human --target-rms -18.5 --subtype PCM_24

# 7-12 kHz harmonic rebuild
python -m tiny_voca.hf_synth --input raw.wav --output hfsynth.wav \
  --voice F2 --preset hf --base warm --target-rms -18.5 --subtype PCM_24

# full natural chain (natural / soft / formant / dry)
python -m tiny_voca.naturalize --input raw_pitch.wav --output natural.wav \
  --voice F2 --preset natural --subtype PCM_24
python -m tiny_voca.naturalize --input raw_pitch.wav --output natural_soft.wav \
  --voice F2 --preset soft --subtype PCM_24
python -m tiny_voca.naturalize --input raw_pitch.wav --output natural_formant.wav \
  --voice F2 --preset formant --subtype PCM_24
python -m tiny_voca.naturalize --input raw_pitch.wav --output natural_dry.wav \
  --voice F2 --preset dry --subtype PCM_24

# M2 pair
python -m tiny_voca.infer --checkpoint weights/tiny-voca.pt --voice M2 \
  --text "$TEXT" --output raw_m2.wav
python -m tiny_voca.postfilter --input raw_m2.wav --output warm_m2.wav \
  --voice M2 --preset warm --target-rms -18.5 --subtype PCM_24
python -m tiny_voca.hf_synth --input raw_m2.wav --output hfsynth_m2.wav \
  --voice M2 --preset hf --base warm --target-rms -18.5 --subtype PCM_24
```

The branch is `hf-harmonic-formant-synth`; `tiny_voca_hfsynth` and
`tiny_voca_naturalize` are experimental. The mainline `postfilter` and
`humanize` presets are unchanged.
