# tiny-voca

tiny-voca is a small English text-to-speech model. It has two voices, 4.5M
parameters, and runs on a CPU or in the browser.

[Demo](https://tiny-voca.n4ze3m.com) · [Weights](https://huggingface.co/n4ze3m/tiny-voca) · [Dataset](https://huggingface.co/datasets/n4ze3m/tiny-voca)

## Features

- **Small**: 4,537,291 parameters, 18 MB, waveform generator included.
- **Fast**: about 70x real time on four CPU threads.
- **Two voices**: F2 (female) and M2 (male), American English, 24 kHz.
- **Runs anywhere**: PyTorch, ONNX Runtime, ONNX Runtime Web, and Core ML.
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

Long text is fine. It is read one sentence at a time and the audio is joined.
A text that ends without a mark gets a full stop, so the last word does not
break off. A sentence with nothing to read, such as one in another script, is
skipped. The output never peaks above 0.99, so it does not clip as a 16-bit
WAV. In Python, `tiny_voca.infer.speak_text` does the same.

## ONNX

```bash
python -m tiny_voca.export --checkpoint weights/tiny-voca.pt --output tiny-voca.onnx
```

A ready file is in the weights repository as `onnx/model.onnx`. The
[model card](https://huggingface.co/n4ze3m/tiny-voca) shows how to call it.

## Core ML (Apple Neural Engine)

On a Mac with Apple silicon, the model can run with Core ML.

```bash
pip install -e ".[own,coreml]"
python -m tiny_voca.coreml \
  --checkpoint weights/tiny-voca.pt \
  --voice F2 \
  --text "Hello! This is tiny voca." \
  --output hello.wav
```

It takes the same options as `tiny_voca.infer`. Without `--output`, it plays
the audio.

The first run converts the checkpoint to three Core ML models. This takes
about two minutes. The models are kept in `~/Library/Caches/tiny-voca/coreml`
(or in `TINY_VOCA_CACHE`), and later runs use them. To write them to a folder
of your choice, use `python -m tiny_voca.coreml.export --checkpoint
weights/tiny-voca.pt --output coreml`, and then `--models coreml` in place of
`--checkpoint`.

The encoder and the decoder run on the Neural Engine in 16-bit floats. The
vocoder runs on the GPU. The small steps between them run on the CPU.
`python -m tiny_voca.coreml.plan --checkpoint weights/tiny-voca.pt` shows how
much of each model runs on the Neural Engine: all of it.

The first load of each model is slow, because macOS compiles it for the
Neural Engine. Later loads take a fraction of a second.

On an M4 Max, a 2-second sentence takes about 6 ms and an 11-second sentence
about 20 ms. That is 250 to 500 times real time, and 6 to 15 times faster than
PyTorch on four CPU threads. With the encoder and the decoder on the CPU only,
it is about 1.7 times slower.

The audio is not identical to PyTorch's sample for sample. The noise is drawn
in a different way, and 16-bit floats move the pitch by about one cent.
`scripts/check_coreml.py` compares the two and writes both for listening.

To run without Python, use the Swift package in `swift/`. It loads the same
models. The phonemizer is in Python only, so a small script writes the
phonemes to a file first. See [swift/README.md](swift/README.md).

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

Built by other people on top of tiny-voca. I have not tested them.

Languages and voices:

- [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts): German.
- [tiny-voca-it](https://github.com/carbogninalberto/tiny-voca-it): Italian.
- [zerre-tts](https://github.com/erkinemreta1/zerre-tts): Turkish.
- [korean-tts](https://github.com/ooodev/korean-tts): Korean, and Korean mixed with English.
- [slyszeglosy](https://github.com/damaon/slyszeglosy): Polish.
- [mini-voz](https://github.com/ElMariones/mini-voz): Spanish, with a Castilian accent.
- [tiny-voca-es](https://github.com/stdevMac/tiny-voca): Spanish from Spain, with a browser demo.
- [alpa](https://huggingface.co/77ethers/alpa): Sanskrit, fine-tuned from tiny-voca on a CPU.
- [fiato](https://huggingface.co/cfahlgren1/fiato): English, with sighs, exhales and gasps.

Ports and apps:

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
