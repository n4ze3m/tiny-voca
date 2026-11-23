# tiny-voca

tiny-voca is a small text-to-speech model that runs on a CPU or in the browser.
The upstream English release has two voices; this repository also builds an
**Arabic MSA voice from scratch** on the CC BY 4.0 Halabi corpus.

[English demo](https://tiny-voca.n4ze3m.com) · [Weights](https://huggingface.co/n4ze3m/tiny-voca) · [Dataset](https://huggingface.co/datasets/n4ze3m/tiny-voca)

## Features

- **Small**: 4,537,291 parameters for the English model, 4,532,683 for the
  Arabic one; 18 MB checkpoint, 21 MB ONNX, waveform generator included.
- **Fast**: ~85x real time for the Arabic ONNX model on four CPU threads
  (PyTorch: ~118x).
- **Arabic voice (Phase 5, this repository)**: one studio voice `F2`, Modern
  Standard Arabic, Damascene accent, 24 kHz. Trained from scratch on CPU from
  1,214 utterances (2.85 h); the English checkpoint is incompatible (76 symbols
  and 2 voices instead of 54 symbols and 1 voice).
- **English voices**: F2 (female) and M2 (male), American English, 24 kHz.
- **Runs anywhere**: PyTorch, ONNX Runtime, ONNX Runtime Web.
- **Open**: code, weights, training recipe and dataset.
- **Browser front end without eSpeak**: the Arabic demo ships a pure-Python G2P
  in Pyodide. eSpeak NG (GPL-3.0) is never embedded.
- **Voice answers (optional HTTP service)**: text question -> LLM answer ->
  Arabic tashkeel -> WAV voice message. The chat UI streams one WAV per
  sentence, so the first audio arrives before the LLM has finished. See
  [server/README.md](server/README.md).

One command starts the whole stack (Ollama `qwen2.5:14b`, Apache-2.0, + the API)
with the 9 GB model in a persistent volume:

```bash
docker compose up -d --build
# then open http://127.0.0.1:8000/app
```

Documentation: [Arabic front end](docs/arabic-frontend.md) ·
[Arabic data](docs/arabic-data.md) · [CPU pipeline](docs/arabic-pipeline.md) ·
[Browser demo](docs/arabic-demo.md) · [Voice-answer service](server/README.md).

## Install

```bash
git clone https://github.com/n4ze3m/tiny-voca.git
cd tiny-voca
pip install -e ".[own]"                     # English front end
pip install -e ".[prepare,annotate,arabic-data,arabic]" onnx onnxruntime
```

For Arabic, always set the language and the rules front end **before** importing
the library:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:TINY_VOCA_LANGUAGE = "ar"
$env:TINY_VOCA_FRONTEND = "rules"
```

## Usage

English (unchanged upstream behaviour):

```bash
python -m tiny_voca.infer \
  --checkpoint weights/tiny-voca.pt \
  --voice F2 \
  --text "Hello! This is tiny voca." \
  --output hello.wav
```

Arabic (checkpoint built in this repository):

```powershell
python -m tiny_voca.infer `
  --checkpoint runs\ar-halabi-f2\inference.pt `
  --voice F2 --device cpu `
  --text "مَرْحَبًا بِالْعالَمِ." `
  --output out\arabic.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a file instead of `--text` |
| `--voice` | `F2` | `F2` or `M2` in English, only `F2` in Arabic |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Raise or lower the pitch |
| `--pitch-range` | 1.0 | More or less pitch movement |

A text that ends without a mark gets a full stop, so the last word does not
break off. A sentence with nothing to read is skipped. The output of
`speak_text` never peaks above 0.99, so it does not clip as a 16-bit WAV.

**Arabic texts must already be vocalised** (full tashkeel): the rules front end
has no reliable automatic diacritisation. `scripts/diacritize.py` can prepare
texts offline with CAMeL or Shakkala.

## ONNX

English:

```bash
python -m tiny_voca.export --checkpoint weights/tiny-voca.pt --output tiny-voca.onnx
```

Arabic, from the final checkpoint:

```powershell
python scripts\export_checkpoint.py --checkpoint runs\ar-halabi-f2\last.pt `
  --output runs\ar-halabi-f2\inference.pt
python -m tiny_voca.export --checkpoint runs\ar-halabi-f2\inference.pt `
  --output runs\ar-halabi-f2\tiny-voca-ar.onnx --opset 17
```

The neighbouring `tiny-voca-ar.json` contains `language: "ar"`,
`frontend: "rules"`, `voices: ["F2"]`, `max_tokens: 512` and the CC BY 4.0
corpus attribution. Verification against PyTorch:

```powershell
python -m tiny_voca.export --checkpoint runs\ar-halabi-f2\inference.pt `
  --output runs\ar-halabi-f2\tiny-voca-ar-exact.onnx --external-noise --opset 17
python scripts\check_onnx.py --checkpoint runs\ar-halabi-f2\inference.pt `
  --onnx runs\ar-halabi-f2\tiny-voca-ar-exact.onnx `
  --release runs\ar-halabi-f2\tiny-voca-ar.onnx --output onnx-check --threads 4
```

Result: identical durations and lengths on five vocalised sentences, maximum
audio error 0.013, **RTFX 85.4** (ONNX, 4 threads).

## Web demo

The source is in `web/`. The Arabic build serves `model.onnx`, `config.json` and
the verbatim G2P file from `web/public/assets/`; no model is fetched from a
remote repository and no GPL binary is bundled.

```bash
cd web
bash fetch_assets.sh /path/to/exported-model   # or an HTTPS base URL
npx wrangler dev
```

The UI is RTL and offers a single voice `F2`. It accepts Arabic punctuation
(`،` `؛` `؟`), reads vocalised examples, and explains that texts without
tashkeel cannot be pronounced reliably.

## Training

English (upstream, GPU recipe):

```bash
export TINY_VOCA_FRONTEND=own
python -m tiny_voca.prepare --inputs data/train-*.parquet --split train --output features
python -m tiny_voca.annotate --features features --output annotations
python -m tiny_voca.train --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output runs/NAME --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0
```

Arabic CPU recipe, counts and exact commands: [docs/arabic-pipeline.md](docs/arabic-pipeline.md).
Summary:

| Split | Annotated | Rejected by QC | Hours |
|---|---:|---:|---:|
| train | 1,214 | 507 | 2.85 |
| validation | 61 | 29 | 0.15 |
| test | 99 | 1 | 0.29 |

3,000 steps, seed 20261113, GAN disabled, no recorded-prosody conditioning.
Loss 48.42 → 8.89; validation predicted mel 0.855 → 0.572. At step 3,000:
pitch error 0.98 semitones, voicing accuracy 0.978, pause accuracy 0.936.

Free-running evaluation without ASR (no model download):

```powershell
python -m tiny_voca.evaluate --checkpoint runs\ar-halabi-f2\inference.pt `
  --annotations data\halabi\annotations --output evaluation\ar-f2 --tag ar-f2-step3000 `
  --splits test validation --novel 8 --per-voice 4 --device cpu --no-asr --seed 0
```

| Group | Samples | Mel DTW | Duration ratio | Pitch std (generated / reference) |
|---|---:|---:|---:|---:|
| test | 4 | 0.827 | 0.860 | 1.83 / 3.44 |
| validation | 4 | 0.717 | 0.936 | 1.75 / 2.24 |
| novel | 8 | — | — | 1.92 / — |

0 front-end token mismatches. These numbers support a listening test; they do
not replace one. WAV pairs for each sample are in `evaluation/ar-f2/audio/`.

## Tests

```bash
python -m unittest discover -s tests
```

Last run on the Arabic build: **181 tests, 4 skipped, OK**.

## Limits

- Arabic is experimental: one voice, Damascene accent, limited data and CPU
  budget. Prosody is flatter than the reference (pitch std about half), and
  generated audio is sometimes slightly short.
- Arabic texts must be vocalised; no reliable browser-side tashkeel.
- English remains two fixed voices; no voice cloning.
- Foreign names and odd spellings are often read wrong.
- The voice service streams sentence by sentence; the browser demo still plays
  pre-generated sentences (no live TTS in the browser).

## Attribution

The Arabic model is derived from the **Arabic Speech Corpus (Halabi 2016)**,
licensed **CC BY 4.0**:

> Halabi, Nawar (2016). *Modern standard Arabic phonetics for speech synthesis*.
> PhD thesis, University of Southampton.

The citation is kept in the manifests, checkpoints' documentation, ONNX
metadata and demo docs. Code and model weights are Apache-2.0.

## Community

Built by other people on top of tiny-voca. I have not tested them.

Languages and voices:

- [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts): German.
- [tiny-voca-it](https://github.com/carbogninalberto/tiny-voca-it): Italian.
- [zerre-tts](https://github.com/erkinemreta1/zerre-tts): Turkish.
- [korean-tts](https://github.com/ooodev/korean-tts): Korean, and Korean mixed with English.
- [slyszeglosy](https://github.com/damaon/slyszeglosy): Polish.
- [mini-voz](https://github.com/ElMariones/mini-voz): Spanish, with a Castilian accent.
- [alpa](https://huggingface.co/77ethers/alpa): Sanskrit, fine-tuned from tiny-voca on a CPU.
- [fiato](https://huggingface.co/cfahlgren1/fiato): English, with sighs, exhales and gasps.

Ports and apps:

- [voca](https://github.com/sultson/voca): a rewrite in Rust.
- [tiny-voca-esp32s3](https://github.com/ngxson/tiny-voca-esp32s3): runs on an ESP32-S3 microcontroller.
- [tiny-voca-dashboard](https://github.com/kleenpulse/tiny-voca-dashboard): a local dashboard.
- [tinyvoca](https://github.com/spurious-cox/tinyvoca): a Mac app.

Same idea, other sounds:

- [tinygroove](https://github.com/totomanov/tinygroove): a drum machine with no samples.

Made something? Open an issue and I will add it.

## Cost

The upstream English project cost $9.56 of GPU time on one rented RTX 4090.
The Arabic model in this repository was trained on a local CPU: about 4.5 hours
for 3,000 steps, plus ~4 hours of CPU alignment and feature preparation.

![Cost breakdown](docs/cost.png)

## Support

It cost less than a pizza to make. If you like it, you can buy me one:
https://ko-fi.com/n4ze3m

## License

Apache-2.0 for code and model weights. CC BY 4.0 for the Halabi dataset and its
derived artefacts; see [runs/ar-halabi-f2/ATTRIBUTION.md](runs/ar-halabi-f2/ATTRIBUTION.md).

## Citation

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
