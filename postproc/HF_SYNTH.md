# Experimental: HF harmonic/formant resynthesis

Branch: `hf-harmonic-formant-synth`

This is the next step after the `warm` HF-contrast filter. It is **experimental**
and is not used by the default `--postfilter` pipeline.

## Motivation

`warm` fixes the 4-8 kHz texture by raising spectral contrast, but the top
octave is still noisy:

| | `warm` | human reference |
|---|---:|---:|
| HF flatness 4-8 kHz | 0.53 | 0.50 |
| HF flatness 8-12 kHz | 0.49 | 0.36 |

The target (Qwen3-TTS) itself has ~0.72/0.67 flatness, so the top-octave
sizzle is inherited. The HF band contains real harmonics plus broadband
noise. The idea here is to **rebuild the 7-12 kHz magnitude from a harmonic
model** while keeping a controlled noise floor, instead of just cutting or
contrasting it.

## Method

For each 1024-point / 256-hop STFT frame:

1. Track F0 with normalized autocorrelation (60-500 Hz), smoothed by a 3-frame
   median to avoid octave jumps.
2. For every harmonic k·F0 above 7 kHz, search a small window for the local
   spectral peak (the search window grows with k to tolerate F0 error, but is
   capped so it cannot reach the next harmonic).
3. Fit a parabolic peak to get the amplitude; smooth the amplitudes across
   harmonic number with a 5-point Hann kernel so the formant envelope survives.
4. Rebuild the HF magnitude as those Gaussian peaks (σ=2 FFT bins) plus a
   noise floor at 20% of a 70th-percentile spectral envelope.
5. Scale the rebuilt band to the original HF energy, keep the original phase,
   and cross-fade in from 7 to 8.5 kHz.
6. Leave unvoiced frames and silence untouched.

Presets: `hf` (floor 0.20, σ 2.0), `strong` (0.10, 1.5), `subtle` (0.30, 2.5).

## Results

On the comparison clip (F2) and the same text in M2, loudness-matched to stock:

| file | HF flat 4-8k | HF flat 8-12k | STOI vs `warm` | peak |
|---|---:|---:|---:|---:|
| F2 `warm` | 0.532 | 0.493 | 1.000 | 0.84 |
| F2 `hf` | **0.534** | **0.378** | 1.000 | 0.84 |
| F2 `hf` strong | 0.531 | **0.326** | 1.000 | 0.84 |
| M2 `warm` | 0.574 | 0.495 | 1.000 | 0.88 |
| M2 `hf` | **0.577** | **0.382** | 1.000 | 0.89 |
| human reference | 0.50 | 0.36 | — | — |

8-12 kHz band energy is preserved (F2 0.268% → 0.262%, M2 0.194% → 0.191%),
and the 8-12 kHz inter-harmonic level barely moves (−25.6 → −25.8 dB for F2,
−24.7 → −24.6 dB for M2). The flatness drop comes from **regularising the
harmonic peaks**, not from gating the noise floor: the 20% noise bed keeps the
natural between-harmonic energy.

Artifacts: quiet-frame level/kurtosis and the sample-level second-difference
statistics are unchanged (the stage only touches voiced HF), and STOI against
`warm` is 1.000. This is a conservative magnitude rebuild.

## Listen

- `assets/tiny_voca_warm.wav` — current best (F2)
- `assets/tiny_voca_hfsynth.wav` — F2, `hf` preset
- `assets/tiny_voca_hfsynth_strong.wav` — F2, stronger floor
- `assets/tiny_voca_warm_m2.wav` — current best (M2)
- `assets/tiny_voca_hfsynth_m2.wav` — M2, `hf` preset

All processed files are 24-bit and RMS-matched to the stock clip.

## Usage

```bash
python -m tiny_voca.hf_synth --input stock.wav --output hf.wav --voice F2 --preset hf
python -m tiny_voca.hf_synth --input stock.wav --output hf.wav --voice F2 \
  --base warm --floor 0.2 --sigma 2.0 --start 7000
```

`--base` selects the spectral filter applied first (`clean`, `smooth`, `warm`,
...); `--preserve harmonic|total` chooses the energy target; `--subtype` writes
float, 24-bit or dithered 16-bit.

## Caveats

* 4-8 kHz is intentionally untouched: `warm` already matches human there for
  F2, and rebuilding 4-8 kHz made M2 slightly worse (dense harmonics + F0
  ambiguity).
* M2 4-8 kHz flatness remains ~0.57 versus human 0.50; a future version would
  need better F0/harmonic tracking for low-pitched voices.
* Only the magnitude is rebuilt; the phase is the original. Strong settings
  may sound "phasey" or over-clean and should be checked by ear.
* This is a branch experiment; the default `postfilter`/`humanize` behaviour is
  unchanged.
