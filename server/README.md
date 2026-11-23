# Arabic voice-answer service

A small HTTP service that turns a **text question** into an **Arabic voice
message**:

```text
client question (text)
    -> LLM answer (short, Modern Standard Arabic, no diacritics)
    -> CAMeL Tools tashkeel
    -> Arabic G2P + ONNX model (F2, 24 kHz)
    -> WAV reply + the answer text in a response header
```

An optional streaming path (`POST /api/voice/answer/stream`) starts the same
pipeline **per sentence** as soon as the LLM finishes that sentence, so the
first audio arrives while the model is still writing. The model is unchanged
(`qwen2.5:14b`, Apache-2.0).

The service reuses the exact ONNX model and the exact G2P file of the browser
demo (`web/public/assets/`). Nothing is downloaded at inference time. eSpeak NG
(GPL-3.0) is **not** used and **not** shipped.

## 1. Install

From the repository root, with Python 3.11-3.14:

```powershell
python -m pip install -e ".[voice-service]"
camel_data -i defaults          # CALIMA MSA database, once, at image build time
```

`camel_data -i defaults` writes the morphology database to the user data
directory. Run it once when building the image, not at every start.

On a fresh clone the 21 MB ONNX is **not** present (`*.onnx` is gitignored).
Restore the pinned release artefact with a SHA-256 check:

```powershell
python scripts\fetch_model.py
```

`config.json` and `arabic_text.py` are versioned. The Docker build and GitHub
Actions run the same script, so no local file is ever required. To use your own
locally trained model instead:

```powershell
Copy-Item runs\ar-halabi-f2\tiny-voca-ar.onnx server\assets\model.onnx -Force
```

## 2. Production stack (Docker + Ollama) — recommended

`docker-compose.yml` starts Ollama and the API together. The 9 GB model lives in
a named volume, so it survives rebuilds and there is nothing to install on the
host except Docker itself.

```powershell
docker compose up -d --build
docker compose logs -f api          # wait for "Application startup complete"
```

Open <http://127.0.0.1:8000/app> and ask a question in Arabic. Health check:
<http://127.0.0.1:8000/health>.

```powershell
docker compose ps
docker compose down                 # keeps ollama_data: no 9 GB re-download
```

### Why `qwen2.5:14b`

| Model | Arabic | Licence | Verdict |
|---|---|---|---|
| `command-r7b-arabic` | best specialised Arabic benchmarks | CC BY-NC-4.0 | **rejected**: no commercial reuse |
| `qwen2.5:14b` | strong MSA, multilingual, 9.0 GB | **Apache-2.0** | **selected** |

Measured on this machine (i7-12700, 32 GB RAM, Intel UHD 770, no discrete GPU):

| Call | Latency |
|---|---|
| Native Ollama, 8 tokens, cold (model load) | 26.9 s |
| Native Ollama, 8 tokens, warm | 2.1 s |
| Full `/api/voice/answer`, short answer, cold | ~32 s |
| Full `/api/voice/answer`, ~30-word answer, warm | ~19 s |

CPU inference of a 14B model is the bottleneck; the Arabic ONNX synthesis itself
costs only ~0.3 s of CPU for 5 s of speech. Hence `TINY_VOCA_LLM_KEEP_ALIVE=-1`
(keep the model resident) and `TINY_VOCA_LLM_TIMEOUT=120` in Compose. We kept
`qwen2.5:14b` for answer quality: streaming hides most of the wait without
downgrading the model. (On an 8 GB host, `qwen2.5:7b` in both
`TINY_VOCA_LLM_MODEL` and the `model-init` pull is mandatory.)

### Why streaming instead of a smaller model

The chat UI now reads `POST /api/voice/answer/stream`. Ollama emits an SSE
stream of token fragments; `server/answer.py` groups them into sentences and
the service diacritizes + synthesizes each one immediately. The browser queues
the audio parts so they play in order, and shows
`جارٍ إعداد الجواب الصوتي... (1/2)` while the rest is being made. Stop and
regenerate buttons are available. The original blocking endpoint is preserved
for API clients and as a browser fallback.

| Call | Latency (measured, warm model) |
|---|---|
| Full `/api/voice/answer`, ~30-word answer | ~19 s before any sound |
| First audio part, short one-sentence answer | ~9 s (9.3 s for the whole answer) |
| First audio part, longer two-sentence answer | ~9.5 s instead of ~24.5 s for the whole answer |
| Later parts | emitted as each sentence completes; ~0.3 s synthesis each |

These figures are CPU-only (i7-12700). Time to the first sound is dominated by
how quickly the 14B model writes the **first sentence**; streaming removes the
wait for the rest of the answer. The first ever request still pays the ~27 s
model load; `keep_alive=-1` pays it only once.

The API image is CPU-only (no PyTorch), runs as a non-root user, and downloads
the CALIMA database once at **build** time. `server/assets/model.onnx` is copied
into the image; `.dockerignore` keeps `runs/`, `data/` and `*.pt` out of the
build context.

## 3. Configure the LLM (without Docker)

Any OpenAI-compatible `/v1/chat/completions` endpoint works: OpenAI, Gemini's
OpenAI compatibility layer, Ollama, vLLM, llama.cpp...

| Variable | Default | Meaning |
|---|---|---|
| `TINY_VOCA_LLM_URL` | `https://api.openai.com/v1/chat/completions` | Chat completions endpoint |
| `TINY_VOCA_LLM_KEY` | *(required)* | Bearer token. For a local server with no auth, set any non-empty value |
| `TINY_VOCA_LLM_MODEL` | `gpt-4o-mini` | Model name sent in the request |
| `TINY_VOCA_LLM_TIMEOUT` | `60` | HTTP timeout in seconds (a cold 14B load takes ~27 s) |
| `TINY_VOCA_LLM_KEEP_ALIVE` | *(unset)* | Ollama only: `-1` keeps the model in RAM between questions (`-1m`, `5m` also accepted). Leave unset for hosted APIs |
| `TINY_VOCA_ASSETS` | `server/assets` | Directory with `model.onnx`, `config.json`, `arabic_text.py` |

The system prompt asks for one or two short sentences in MSA, **without**
diacritics: tashkeel is added by the service. The API key stays on the server;
it is never returned to the client.

Example with a local Ollama server:

```powershell
$env:TINY_VOCA_LLM_URL        = "http://127.0.0.1:11434/v1/chat/completions"
$env:TINY_VOCA_LLM_KEY        = "local"
$env:TINY_VOCA_LLM_MODEL      = "qwen2.5:14b"
$env:TINY_VOCA_LLM_KEEP_ALIVE = "-1"
$env:TINY_VOCA_LLM_TIMEOUT    = "120"
```

A timeout returns `504 Gateway Timeout`; a missing key returns `503`.

`TINY_VOCA_LANGUAGE=ar` and `TINY_VOCA_FRONTEND=rules` are set by the module
itself before importing the library.

## 4. Start

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
python -m uvicorn server.tts_service:app --host 0.0.0.0 --port 8000
```

Interactive documentation: <http://127.0.0.1:8000/docs>.  
**Ready-to-use chat application: <http://127.0.0.1:8000/app>** (`/demo` is an
alias) — type a question in Arabic, press « أرسل » and hear the voice answer.
The answer text is **hidden in the UI: voice only**. The implementation is kept:
the text still arrives in `X-Answer-Text` and stays in the conversation history
sent to the LLM, so follow-up questions keep their context. Everything is served
from the same origin, so no CORS setup is needed.

The ONNX session is created once at the first request and protected by a lock;
it is never reloaded per request.

### Test without any LLM or API key

A development-only fake LLM is included. In two terminals:

```powershell
# Terminal 1 - fake LLM (stdlib only, no install)
python server\mock_llm.py 8011

# Terminal 2 - the voice service
$env:TINY_VOCA_LLM_URL   = "http://127.0.0.1:8011/v1/chat/completions"
$env:TINY_VOCA_LLM_KEY   = "local"
$env:TINY_VOCA_LLM_MODEL = "mock-ar"
$env:PYTHONIOENCODING    = "utf-8"
python -m uvicorn server.tts_service:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/demo>, type an Arabic question and press
« أرسل ». The answer is always the same short sentence: this tests the pipeline,
not the quality of the LLM. Stop both with `Ctrl+C`.

## 5. Endpoints

| Method | Path | Body | Reply |
|---|---|---|---|
| `GET` | `/health` | — | language, frontend, voices, sample rate, step, attribution |
| `POST` | `/api/tts` | `{"text": "مَرْحَبًا بِالْعالَمِ."}` | `audio/wav` |
| `POST` | `/api/voice/answer` | `{"question": "كيف حالك؟"}` | `audio/wav` + `X-Answer-Text` header (percent-encoded) |
| `POST` | `/api/voice/answer/stream` | `{"question": "كيف حالك؟"}` | `application/x-ndjson`: one audio part per line, then a `done` event |

`/api/tts` fields: `text`, `voice` (`F2`), `speed` (0.95), `pitch_range` (1.35),
`already_vocalized` (`true`). The rules G2P can also read unvocalised Arabic,
but vocalised text is more accurate; set `already_vocalized` to `false` to run
CAMeL tashkeel on the text first. Text with no readable Arabic word returns 400.

`/api/voice/answer` fields: `question`, `history` (list of
`{"role": "user"|"assistant", "content": str}`, optional), `voice`, `speed`,
`pitch_range`, `max_chars` (300). Only the last six history turns reach the
LLM. Repeated identical questions are served from an in-memory cache keyed by
model, question, history, length, voice, speed and pitch range.

`/api/voice/answer/stream` accepts the same fields plus `chunk_chars` (220):
the maximum length of one synthesized sentence chunk. Each line is one JSON
object; `audio` is a base64-encoded 24 kHz 16-bit PCM WAV:

```json
{"part": 1, "text": "عاصمة مصر هي القاهرة.", "audio": "UklGR..."}
{"part": 2, "text": "تقع على ضفاف النيل.", "audio": "UklGR..."}
{"done": true, "text": "عاصمة مصر هي القاهرة. تقع على ضفاف النيل.", "parts": 2}
```

On a late failure the stream emits `{"error": "...", "status": 503|504|502}`
instead of a broken HTTP status (headers are already sent). Identical questions
replay the cached audio parts without calling the LLM again.

To watch the stream arrive in real time (`-N` disables curl buffering; each
line is printed when the corresponding sentence has been synthesized):

```powershell
curl.exe -N -X POST "http://127.0.0.1:8000/api/voice/answer/stream" `
  -H "Content-Type: application/json" `
  -d '{"question":"ما عاصمة مصر؟"}'
```

In Python, use `httpx.stream(...)` and `response.iter_lines()` (a plain
`httpx.post` waits for the whole answer and hides the latency gain).

### Try it

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/tts" `
  -H "Content-Type: application/json" `
  -d '{"text":"مَرْحَبًا بِالْعالَمِ."}' --output reponse.wav

curl.exe -X POST "http://127.0.0.1:8000/api/voice/answer" `
  -H "Content-Type: application/json" `
  -d '{"question":"كيف حالك اليوم؟"}' --output message-vocal.wav
```

Browser client: the endpoint is **POST-only** (a plain `<audio src="...">` will
not work). `POST` with `fetch`, read the `X-Answer-Text` header with
`decodeURIComponent`, and pass the response blob to an `<audio>` element —
`server/app.html` is a complete working example.

## 6. Tests

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m unittest discover -s tests -p "test_answer_stream.py" -v   # SSE + chunking, no model
python -m unittest discover -s tests -p "test_voice_service.py" -v   # real ONNX + endpoints
python -m unittest discover -s tests         # full suite
```

The tests exercise the real ONNX model and CAMeL Tools but replace the LLM
with a stand-in: they need neither network nor API key (apart from the one-time
model download above).

`.github/workflows/ci.yml` runs this exact suite on every push and pull request,
then builds the API image from a clean checkout — including the verified model
download. A green CI means a fresh clone is deployable.

## 7. Production notes

- **Format**: 24 kHz mono **16-bit PCM WAV** (tag 1, ~48 KB/s), playable by every
  browser and voice-messaging tool. Float WAV (tag 3) made some players report a
  wrong or truncated duration. For WhatsApp/Telegram, convert to Opus/OGG
  in an image that has a **licensed** ffmpeg build:
  `ffmpeg -i reponse.wav -c:a libopus -b:a 32k reponse.ogg`. ffmpeg is not
  installed on the development machine.
- **Performance**: the Arabic ONNX model runs at ~85x real time on 4 CPU
  threads; about 0.3 s of CPU for 5 s of speech. LLM latency usually dominates.
- **Concurrency**: the ONNX session is locked; for more throughput run several
  worker processes behind a load balancer. Put a disk cache and a rate limit in
  front of the LLM call.
- **Limits**: `max_tokens=512`, `max_chars<=600`. CAMeL tashkeel can misread
  proper names and the pronunciation follows that error.
- **Quality**: single voice `F2`, Damascene accent, flatter prosody than the
  reference and occasionally short audio. Sentence-level streaming, no voice
  cloning. Slight pauses can be heard between audio parts; raise `chunk_chars`
  to 300-400 to get fewer, longer parts at the cost of a later first sound.
- **Attribution**: all model/data derivatives come from the Arabic Speech
  Corpus (Halabi 2016), CC BY 4.0. Keep the citation in the service metadata
  and in any user-facing page: Halabi, Nawar (2016). *Modern standard Arabic
  phonetics for speech synthesis*. PhD thesis, University of Southampton.
