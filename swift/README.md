# tiny-voca on Core ML, in Swift

This package runs tiny-voca with Core ML and without Python.
The Encoder and the Decoder run on the Apple Neural Engine (ANE).
The Vocoder runs in float32 on the GPU.
The host steps between the models (durations, frame expansion, pulse train, noise) are Swift ports of `tiny_voca.coreml.vocoder`.

- Library `TinyVoca`: `TinyVoca(modelsDirectory:)` and `synthesize(tokens:stress:voice:speed:pitchShift:pitchRange:seed:)`.
- Executable `tiny-voca-coreml`: speaks an `ids.json` file into a WAV file, times the stages, and prints the compute plan.
- Platforms: macOS 15+ and iOS 18+ (the library). Swift 6.

## Make the models

The models come from the Python package (see the main README).
The runner needs `Encoder.mlmodelc`, `Decoder.mlmodelc` and `Vocoder.mlmodelc` in one directory.
If only an `.mlpackage` is there, the runner compiles it once and keeps the `.mlmodelc`.

```sh
python -m tiny_voca.coreml.export --checkpoint weights/tiny-voca.pt --output models
```

The export gives the ids to the ANE models as float16 inputs, and then all ops are on the ANE.

## Build

```sh
cd swift
swift build -c release
```

The executable is `swift/.build/release/tiny-voca-coreml`.

## Phonemize

The phonemizer is Python only. `phonemize.py` writes the token ids:

```sh
python swift/phonemize.py "Hello there. How are you today?" -o ids.json
```

It splits the text into sentences as `tiny_voca.infer.speak_text` does.
Each sentence is one utterance. `--each` makes each argument one utterance, without splitting.

## Run

```sh
tiny-voca-coreml --models models --ids ids.json --output out.wav --voice F2
```

Options: `--voice F2|M2`, `--speed`, `--pitch-shift`, `--pitch-range`, `--seed`.
The audio of the utterances is joined and scaled down when its peak is above 0.99 (as `limit_peak`).
The output is a 24 kHz mono float32 WAV file.
Utterance k uses the seed `seed + k`, as `speak_text` does.
The Gaussian noise is not numpy's or torch's noise for the same seed, so the audio is not sample-equal to Python.

Other options:

- `--bench N`: loads all functions, then times N runs of each utterance. It prints the median time of each stage and the real-time factor.
- `--units ane|cpu|all|gpu`: compute units of the Encoder and the Decoder (default `ane`, CPU and Neural Engine). Do not use `all`: then Core ML can put long decoder buckets on the GPU, which is slower.
- `--vocoder-units gpu|cpu|all`: compute units of the Vocoder (default `gpu`).
- `--plan`: prints, for each Encoder and Decoder function, the share of ops that Core ML puts on the Neural Engine (MLComputePlan).
- `--load-times`: loads all functions and prints the load time of each.
- `--noise FILE`, `--dump DIR`: for parity checks. `--noise` reads raw float32 noise for all utterances. `--dump` writes the intermediate values of each utterance as raw float32 or int32 files.

Without `--bench`, the runner loads only the functions that the utterances need.

## Measured numbers

Apple M4 Max, macOS 26, release build, the default buckets of `ane.py`.

Parity. The Swift runner and the Python Core ML path (same models, same ids, same noise) give the same values.
The encoder, duration, durations, decoder and f0 outputs are bit-equal.
The pulse train differs by at most 1.2e-7 (one float32 step).
The audio SNR is 165 dB or more, for the 5 test sentences and both voices.

Compute plan (`--plan`, CPU and Neural Engine): 100% of the ops of every Encoder function (n32 to n512) and every Decoder function (t128 to t3072) are on the Neural Engine.
The Vocoder has its 129 ops on the GPU.

Speed. The 5 sentences of `scripts/check_onnx.py`; median of 30 runs; lowest of 3 rounds.
Times are in milliseconds. "pulse+noise" is the pulse train and the noise. xRT is audio time divided by compute time.

Encoder and Decoder on the Neural Engine (`--units ane`):

| voice | tokens | frames | audio s | encoder | host | decoder | pulse+noise | vocoder | total | xRT |
|---|---|---|---|---|---|---|---|---|---|---|
| F2 | 24 | 182 | 1.94 | 0.29 | 0.06 | 0.70 | 0.39 | 1.82 | 3.28 | 591 |
| F2 | 46 | 303 | 3.23 | 0.31 | 0.09 | 1.54 | 0.49 | 2.06 | 4.54 | 711 |
| F2 | 82 | 517 | 5.51 | 0.38 | 0.13 | 2.33 | 0.75 | 2.65 | 6.22 | 886 |
| F2 | 58 | 430 | 4.59 | 0.30 | 0.10 | 1.46 | 0.59 | 2.37 | 4.79 | 958 |
| F2 | 170 | 1029 | 10.98 | 0.56 | 0.23 | 7.03 | 1.11 | 4.02 | 12.97 | 847 |
| M2 | 24 | 177 | 1.89 | 0.29 | 0.07 | 0.74 | 0.45 | 1.83 | 3.39 | 558 |
| M2 | 46 | 272 | 2.90 | 0.31 | 0.09 | 1.63 | 0.53 | 2.07 | 4.68 | 620 |
| M2 | 82 | 458 | 4.89 | 0.39 | 0.14 | 1.68 | 0.73 | 2.63 | 5.61 | 872 |
| M2 | 58 | 420 | 4.48 | 0.32 | 0.13 | 1.69 | 0.70 | 2.53 | 5.39 | 831 |
| M2 | 170 | 971 | 10.36 | 0.62 | 0.28 | 5.07 | 1.28 | 4.20 | 11.46 | 904 |

Encoder and Decoder on the CPU only (`--units cpu`), for comparison:

| voice | tokens | audio s | encoder | decoder | total | xRT | ANE gain (total) |
|---|---|---|---|---|---|---|---|
| F2 | 24 | 1.93 | 0.45 | 4.03 | 8.45 | 228 | 2.6x |
| F2 | 46 | 3.22 | 0.64 | 10.58 | 15.68 | 205 | 3.5x |
| F2 | 82 | 5.50 | 1.12 | 16.94 | 23.34 | 236 | 3.8x |
| F2 | 58 | 4.59 | 0.65 | 10.55 | 16.28 | 282 | 3.4x |
| F2 | 170 | 10.97 | 2.67 | 74.40 | 87.29 | 126 | 6.7x |
| M2 | 24 | 1.88 | 0.43 | 3.76 | 7.84 | 240 | 2.3x |
| M2 | 46 | 2.91 | 0.63 | 10.43 | 15.24 | 191 | 3.3x |
| M2 | 82 | 4.90 | 1.17 | 10.40 | 16.58 | 296 | 3.0x |
| M2 | 58 | 4.47 | 0.55 | 9.22 | 14.29 | 313 | 2.7x |
| M2 | 170 | 10.35 | 2.25 | 22.34 | 31.95 | 324 | 2.8x |

The PyTorch model (CPU, 4 threads) makes the first sentence (1.94 s) in 41 ms (x47). The Swift runner makes it in 3.3 ms.
With the ANE, the decoder is 5 to 10 times faster than on the CPU. The vocoder on the GPU is now the largest stage for short sentences.

Load time (all 14 functions, `--load-times`):

- First load of a new model directory: 65 s when loaded one after the other (Core ML compiles each function for the ANE; t3072 takes 29 s). This happens once. Core ML keeps the result in its cache.
- Later loads: 0.6 to 2.5 s for all functions in parallel (12 s one after the other: each load of a decoder function reads the whole multifunction program, about 1.2 s).
- A normal run loads only the needed functions: about 1.1 s for a short text.
