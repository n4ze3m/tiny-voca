# tiny-voca

tiny-voca is a small English text-to-speech model. It has two voices, 4.5M
parameters, and runs on a CPU or in the browser.

[Demo](https://tiny-voca.n4ze3m.com) · [Weights](https://huggingface.co/n4ze3m/tiny-voca) · [Dataset](https://huggingface.co/datasets/n4ze3m/tiny-voca)

## Features

- **Small**: 4,537,291 parameters, 18 MB, waveform generator included.
- **Fast**: about 70x real time on four CPU threads.
- **Two voices**: F2 (female) and M2 (male), American English, 24 kHz.
- **Runs anywhere**: PyTorch, ONNX Runtime, and ONNX Runtime Web.
- **Open**: code, weights, training recipe and dataset. Trained from scratch.

## Install

```bash
git clone https://github.com/n4ze3m/tiny-voca.git
cd tiny-voca
pip install -e ".[own]"
hf download n4ze3m/tiny-voca --local-dir weights
export DIZIONE_DATA=weights/phonemizer
```

## Usage

```bash
python -m tiny_voca.infer \
  --checkpoint weights/tiny-voca.pt \
  --voice F2 \
  --text "Hello! This is tiny voca." \
  --output hello.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a file instead of `--text` |
| `--voice` | `F2` | `F2` or `M2` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Raise or lower the pitch |
| `--pitch-range` | 1.0 | More or less pitch movement |
| `--postfilter` | `off` | `off`, `neutral`, `smooth`, `clean`, `warm` or `strong` (see below) |

Long text is fine. It is read one sentence at a time and the audio is joined.
A text that ends without a mark gets a full stop, so the last word does not
break off. A sentence with nothing to read, such as one in another script, is
skipped. The output never peaks above 0.99, so it does not clip as a 16-bit
WAV. In Python, `tiny_voca.infer.speak_text` does the same.

## Post-processing

`--postfilter clean` runs an inference-only spectral filter: a gentle
voice-specific tone correction, a de-hiss stage for pauses, a 55 Hz high-pass
and a true-peak limiter. Presets are `neutral` (high-pass + limiter only),
`smooth` (HF de-sizzle), `clean`, `warm` (adds a measured 180-700 Hz body
boost) and `strong` (more noise removal). It can also be applied to an existing 24 kHz file:

```bash
python -m tiny_voca.postfilter --input raw.wav --output clean.wav --voice F2 --preset clean
```

For a less dry, more "recorded" result, `tiny_voca.humanize` adds a quiet room
tone, sparse early reflections, occasional breaths and a gentle slow expander:

```bash
python -m tiny_voca.humanize --input raw.wav --output human.wav --voice F2 --preset human
```

The filter was tuned on the validation targets. `smooth`/`warm` also raise
high-frequency spectral contrast (`--contrast`, default 1.5) to remove the
broadband HF sizzle the model inherits from its target. The filter cannot fix
prosody or timing. For F2, `--pitch-shift 0.07` brings the F0 distribution
closer to the target. The measurements, negative results and limitations are in
[postproc/REPORT.md](postproc/REPORT.md).

On the `hf-harmonic-formant-synth` branch, `tiny-voca-hfsynth` is an
experimental 7-12 kHz harmonic/formant rebuild, and `tiny-voca-naturalize`
composes pitch calibration, HF synthesis, formant warp, shimmer and room into
one naturalness chain. See [postproc/HF_SYNTH.md](postproc/HF_SYNTH.md) and
[postproc/NATURAL.md](postproc/NATURAL.md).

## ONNX

```bash
python -m tiny_voca.export --checkpoint weights/tiny-voca.pt --output tiny-voca.onnx
```

A ready file is in the weights repository as `onnx/model.onnx`. The
[model card](https://huggingface.co/n4ze3m/tiny-voca) shows how to call it.

## Web demo

The [demo](https://tiny-voca.n4ze3m.com) runs the model in the browser. It
plays each sentence as soon as it is made, and keeps the whole clip in the
player. Its source is in `web/`.

```bash
cd web
bash fetch_assets.sh /path/to/dizione
npx wrangler dev
```

## Training

```bash
export TINY_VOCA_FRONTEND=own
python -m tiny_voca.prepare --inputs data/train-*.parquet --split train --output features
python -m tiny_voca.annotate --features features --output annotations
python -m tiny_voca.train --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output runs/NAME --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0
python -m tiny_voca.evaluate --checkpoint runs/NAME/inference.pt --annotations annotations --output eval/NAME
```

The released weights took 7.7 hours on one RTX 4090. More about the model and
the data is in [docs/model.md](docs/model.md).

## Tests

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests
```

## Limits

- English only. Two fixed voices. No voice cloning.
- The F2 voice sounds younger than intended. A more mature voice is planned.
- Foreign names and odd spellings are often read wrong.
- No streaming yet.

## Community

Built by other people on top of tiny-voca.

Languages and voices:

- [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts): German.
- [tiny-voca-it](https://github.com/carbogninalberto/tiny-voca-it): Italian.
- [zerre-tts](https://github.com/erkinemreta1/zerre-tts): Turkish.
- [clarvia-tts-tiny](https://huggingface.co/wite-tech/clarvia-tts-tiny): Turkish, ten voices, with its own [phonemizer](https://huggingface.co/wite-tech/clarvia-phonemizer) for numbers, dates and money.
- [korean-tts](https://github.com/ooodev/korean-tts): Korean, and Korean mixed with English.
- [slyszeglosy](https://github.com/damaon/slyszeglosy): Polish.
- [mini-voz](https://github.com/ElMariones/mini-voz): Spanish, with a Castilian accent.
- [tiny-voca-es](https://github.com/stdevMac/tiny-voca): Spanish from Spain, with a browser demo.
- [tiny-voca-fr](https://github.com/DanielSuncost/tiny-voca-fr): French, with a browser demo.
- [tiny-voca-ptbr](https://github.com/SemFence/tiny-voca-ptbr): Brazilian Portuguese, with its own phonemizer.
- [tiny-voca-ptbr-kokoro](https://github.com/Haple/tiny-voca-ptbr-kokoro): Brazilian Portuguese, with Kokoro's Brazilian voices as the teacher.
- [tiny-voca Arabic](https://github.com/BenkadourKhaled/tiny-voca/tree/arabic-halabi-f2): Modern Standard Arabic, trained on a CPU from the Arabic Speech Corpus.
- [Bengali Voice](https://tts.hansda.in/): Bengali script and Romanized Bengali, with a browser demo.
- [alpa](https://huggingface.co/77ethers/alpa): Sanskrit, fine-tuned from tiny-voca on a CPU.
- [fiato](https://huggingface.co/cfahlgren1/fiato): English, with sighs, exhales and gasps.

Ports and apps:

- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/tts.md#tiny-voca): runs tiny-voca with `llama-tts`, from [GGUF files](https://huggingface.co/ggml-org/tiny-voca-GGUF).
- [voca](https://github.com/sultson/voca): a rewrite in Rust.
- [tiny-voca-esp32s3](https://github.com/ngxson/tiny-voca-esp32s3): runs on an ESP32-S3 microcontroller.
- [tiny-voca-dashboard](https://github.com/kleenpulse/tiny-voca-dashboard): a local dashboard.
- [tinyvoca](https://github.com/spurious-cox/tinyvoca): a Mac app.
- [tiny-voca-esp32s3](https://github.com/ngxson/tiny-voca-esp32s3): ESP32-S3 port.

Same idea, other sounds:

- [tinygroove](https://github.com/totomanov/tinygroove): a drum machine with no samples.

Made something? Open an issue and I will add it.

## Cost

The whole project cost $9.56 of GPU time on one rented RTX 4090. That is the
dataset, the failed tries and the final training run.

![Cost breakdown](docs/cost.png)

## Support

It cost less than a pizza to make. If you like it, you can buy me one:
https://ko-fi.com/n4ze3m

## License

Apache-2.0 for code and weights. CC BY 4.0 for the dataset.

## Citation

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
