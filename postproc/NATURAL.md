# `naturalize`: full programmatic naturalness chain

Branch: `hf-harmonic-formant-synth`

This composes everything that survived the experiments into one post-processing
chain. It is still **experimental**; the mainline `postfilter`/`humanize`
defaults are unchanged.

## Chain

1. **F2 pitch calibration at synthesis time**: `pitch_shift=0.07`
   (~1.2 semitones). This is not post-processing — it must happen in
   `speak_text` — but it is the single biggest prosody correction.
2. **`warm` + HF harmonic rebuild** (`hf_synth`, 7–12 kHz, floor 20%). The
   top-octave flatness drops from ~0.49 to ~0.38.
3. **Mild formant warp** (spectral-envelope warp, ratio 0.99). F0 and the
   harmonic comb stay put; the formant envelope moves down ~1%. The stronger
   `formant` preset uses 0.96.
4. **Band-limited shimmer**: 0.8% slow amplitude variation applied only below
   5 kHz, gated to voiced frames. This adds life without re-hazing the HF.
5. **Slow expander** (0.7) to restore p10–p90 dynamic range.
6. **Breaths** in long pauses (0.7) and a **low-passed room bed** (room 0.8,
   tone/wet filtered at 2 kHz, wet 0.18). The low-pass is deliberate: a
   broadband room re-introduces HF flatness.
7. Loudness match to −18.5 dBFS RMS and peak limit at 0.99.

## Results

| file | quiet floor | p10–p90 | HF flat 4–8k | HF flat 8–12k | HNR | CPP | STOI |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock, no pitch shift | −71.8 | 18.7 | 0.700 | 0.654 | 9.25 | 89.1 | — |
| `natural` | **−58.1** | **21.4** | **0.54** | **0.41** | 9.49 | 97.1 | 0.988 |
| `natural_soft` | −59.4 | 21.1 | 0.54 | 0.41 | 9.56 | 97.8 | 0.992 |
| `natural_formant` | −58.2 | 21.6 | 0.56 | **0.37** | 9.61 | 96.5 | 0.985 |
| `natural_dry` | −85.9 | 21.2 | 0.54 | 0.40 | 9.87 | 98.4 | 0.999 |
| human reference | −56.7 | 22.0 | 0.50 | 0.36 | 7.8 | 87 | — |

STOI is against the pitch-shifted `warm` base; the pitch calibration itself is
not included in that number. F0 on this text: default 191/261/333 Hz,
`pitch_shift=0.07` gives 207/279/358 Hz (p10/p50/p90).

The `natural` file now sits close to the human reference on the three
perceptual proxies we can measure: **quiet floor**, **dynamic range**, and
**HF flatness**. `natural_formant` goes further on the voice-character side
(stronger formant warp, 8–12 kHz flatness 0.37).

## Listen

All 24-bit, RMS-matched:

- `assets/tiny_voca_stock.wav` — raw, no pitch shift
- `assets/tiny_voca_natural.wav` — **the main result**
- `assets/tiny_voca_natural_soft.wav` — safer formant/shimmer/room
- `assets/tiny_voca_natural_formant.wav` — stronger formant shift
- `assets/tiny_voca_natural_dry.wav` — chain without room/breaths
- `assets/tiny_voca_natural_nopitch.wav` — same chain without pitch calibration
- `assets/tiny_voca_warm.wav` — previous best, for A/B

## Usage

```bash
python -m tiny_voca.naturalize --input raw.wav --output natural.wav --voice F2
python -m tiny_voca.naturalize --input raw.wav --output soft.wav --voice F2 --preset soft
python -m tiny_voca.naturalize --input raw.wav --output formant.wav --voice F2 --preset formant
```

For F2, generate the raw input with `pitch_shift=0.07`:

```python
wav, phones = speak_text(model, text, "F2", pitch_shift=0.07)
open("raw.wav", "wb").write(...)  # or use the CLI with --pitch-shift 0.07
```

## Caveats

* No formal listening test yet. The formant warp, shimmer and room level are
  taste parameters.
* The room is intentionally low-frequency only (2 kHz); it adds a floor and
  early reflections but no HF air.
* Pitch calibration is synthesis-time, not post-processing. Without it, use
  `natural_nopitch` as the reference.
* The mainline pipeline remains `--postfilter warm` / `humanize`; this branch is
  an experiment.

## What is still missing

Prosody timing (final lengthening, phrase declination, stress accents) and
articulation transients (plosive/affricate attacks) are the next real gains.
Both need a listening test before more DSP is piled on.
