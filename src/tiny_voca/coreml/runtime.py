"""tiny-voca on Core ML: three models and the host steps between them.

    Encoder (fp16, ANE)  tokens -> encoder states, raw durations         one function per token bucket
    host                 durations, frame expansion (gather + position features), vowel mask
    Decoder (fp16, ANE)  frames -> log F0, harmonic and noise parameters  one function per frame bucket
    host                 F0, band-limited pulse train (float64), Gaussian noise
    Vocoder (fp32, GPU)  parameters, pulses, noise -> audio               flexible length

``CoreMLVoca`` loads a directory written by ``export_models`` (or
``python -m tiny_voca.coreml.export``). The models are loaded from compiled
``.mlmodelc`` bundles kept next to the packages: an ``.mlpackage`` is compiled
again on every load (3-11 s per decoder function), a kept ``.mlmodelc`` loads in
0.05-0.2 s per function and Core ML caches its Neural Engine compilation across processes.
Functions are loaded on first use, so a short sentence never waits for the long
buckets (``preload`` loads them all).

``speak_text`` reads text of any length like ``tiny_voca.infer.speak_text``.
The noise is numpy's (see ``vocoder``), so the audio is not sample-identical to
PyTorch for the same seed; with PyTorch's durations and noise passed in it is
(``check_coreml.py``).
"""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from ..config import Config
from ..infer import ENGLISH_ONLY, PEAK, finished, limit_peak, sentences, shorter
from ..model import UtteranceTooLong
from ..text import NothingToPronounce, frontend_fingerprint, phonemize
from . import ane, vocoder as V

METADATA = "tiny-voca-coreml.json"
# Raise when the converted models change, so cached exports of the same checkpoint are made again.
FORMAT_VERSION = 1
CACHE_VARIABLE = "TINY_VOCA_CACHE"
NAMES = {"encoder": "Encoder", "decoder": "Decoder", "vocoder": "Vocoder"}
# Measured on an M4 Max: with ALL the planner moves the long decoder buckets to the GPU, and they get slower.
COMPUTE_UNITS = {"encoder": "CPU_AND_NE", "decoder": "CPU_AND_NE", "vocoder": V.COMPUTE_UNITS}
STAGES = ("encoder", "expand", "decoder", "source", "vocoder")


def compile_package(package, force=False):
    """The ``.mlmodelc`` next to ``package``, compiled once (``force``: again)."""
    import coremltools as ct
    package = Path(package)
    target = package.with_suffix(".mlmodelc")
    if force or not target.exists():
        partial = target.with_name(target.stem + ".partial.mlmodelc")
        shutil.rmtree(partial, ignore_errors=True)
        ct.utils.compile_model(str(package), str(partial))
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
    return target


def _io(names, shapes, dtypes):
    return [{"name": n, "shape": list(s), "dtype": d} for n, s, d in zip(names, shapes, dtypes)]


def export_models(model, checkpoint, directory, encoder_buckets=ane.ENCODER_BUCKETS, decoder_buckets=ane.DECODER_BUCKETS,
                  query_chunk=ane.QUERY_CHUNK, max_frames=None, log=print):
    """Encoder, Decoder and Vocoder packages, their compiled ``.mlmodelc`` and the metadata,
    written to ``directory`` from a loaded ``TinyVoca``. Ids enter the ANE models as float16:
    the ANE cannot read int32, and float16 ids keep every op of both models on the NE."""
    from ..export import metadata
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = model.float().eval()
    c = model.config
    max_frames = min(max_frames or c.max_frames, max(decoder_buckets))
    max_tokens = min(c.max_tokens, max(encoder_buckets))
    steps = (("encoder", lambda path: ane.convert_encoder(model, path, encoder_buckets, query_chunk, id_dtype=np.float16)),
             ("decoder", lambda path: ane.convert_decoder(model, path, decoder_buckets, query_chunk, id_dtype=np.float16)),
             ("vocoder", lambda path: V.convert(model, path, max_frames=max_frames)))
    for kind, convert in steps:
        start = time.perf_counter()
        package = directory / f"{NAMES[kind]}.mlpackage"
        shutil.rmtree(package, ignore_errors=True)
        convert(package)
        compile_package(package, force=True)
        log(f"{package.name}: converted and compiled in {time.perf_counter() - start:.0f} s")
    harmonic, noise = c.cep_causal + c.cep_anti + c.hop // c.synth_hop, c.noise_cep + c.hop // c.noise_hop
    f16, ch = "float16", c.channels
    info = {k: v for k, v in metadata(model, checkpoint).items() if k not in ("inputs", "outputs")}
    info.update({
        "format": "tiny-voca-coreml", "format_version": FORMAT_VERSION, "max_tokens": max_tokens, "max_frames": max_frames,
        "config": c.to_dict(), "query_chunk": query_chunk,
        "phone_tokens": [int(i) for i in np.flatnonzero(V.PHONE_TOKENS)], "vowel_tokens": [int(i) for i in np.flatnonzero(V.VOWEL_TOKENS)],
        "models": {
            "encoder": {"package": "Encoder.mlpackage", "compiled": "Encoder.mlmodelc", "compute_units": COMPUTE_UNITS["encoder"],
                        "precision": f16, "buckets": sorted(encoder_buckets), "functions": [f"n{n}" for n in sorted(encoder_buckets)],
                        "inputs": _io(ane.ENCODER_INPUTS, ([1, "N"], [1, "N"], [1], [1, "N"]), (f16,) * 4),
                        "outputs": _io(ane.ENCODER_OUTPUTS, ([1, ch, 1, "N"], [1, 2, "N"]), (f16,) * 2)},
            "decoder": {"package": "Decoder.mlpackage", "compiled": "Decoder.mlmodelc", "compute_units": COMPUTE_UNITS["decoder"],
                        "precision": f16, "buckets": sorted(decoder_buckets), "functions": [f"t{t}" for t in sorted(decoder_buckets)],
                        "inputs": _io(ane.DECODER_INPUTS, ([1, ch, 1, "T"], [1, 3, 1, "T"], [1], [1, "T"], [1, "T"], [1]), (f16,) * 6),
                        "outputs": _io(ane.DECODER_OUTPUTS, ([1, "T"], [1, harmonic, "T"], [1, noise, "T"]), (f16,) * 3)},
            "vocoder": {"package": "Vocoder.mlpackage", "compiled": "Vocoder.mlmodelc", "compute_units": COMPUTE_UNITS["vocoder"],
                        "precision": "float32", "frames": [1, max_frames],
                        "inputs": _io(V.INPUTS, ([1, harmonic, "T"], [1, noise, "T"], [1, "T*hop"], [1, "T*hop"]), ("float32",) * 4),
                        "outputs": _io(V.OUTPUTS, ([1, "T*hop"],), ("float32",))}}})
    (directory / METADATA).write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return info


def cache_directory():
    """Where ``cached_models`` keeps its exports: ``TINY_VOCA_CACHE``, else ~/Library/Caches/tiny-voca/coreml."""
    return Path(os.environ.get(CACHE_VARIABLE) or Path.home() / "Library" / "Caches" / "tiny-voca" / "coreml")


def _load_checkpoint(checkpoint):
    from ..infer import load_model, match_frontend
    match_frontend(checkpoint)
    return load_model(checkpoint, "cpu")


def cached_models(checkpoint, cache=None, log=print, **options):
    """The export directory for ``checkpoint``, converted on the first call and kept in
    ``cache`` (default ``cache_directory()``). The key is the checkpoint's SHA-256 and
    ``FORMAT_VERSION``, so a new checkpoint or converter gets its own export. The export
    is written beside its final place and renamed when complete: a stopped first run
    leaves nothing that a later run could take for a finished export."""
    digest = hashlib.sha256()
    with open(checkpoint, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    cache = Path(cache or cache_directory())
    target = cache / f"{digest.hexdigest()[:16]}-v{FORMAT_VERSION}"
    if (target / METADATA).exists():
        return target
    log(f"Converting {Path(checkpoint).name} for the Neural Engine. This is done once and takes about two minutes; "
        f"the models are kept in {target}")
    partial = cache / f"{target.name}.partial-{os.getpid()}"
    shutil.rmtree(partial, ignore_errors=True)
    try:
        export_models(_load_checkpoint(checkpoint), checkpoint, partial, log=log, **options)
        if (target / METADATA).exists():      # another process finished first
            return target
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return target


def _compute_units(name):
    import coremltools as ct
    return getattr(ct.ComputeUnit, name)


class CoreMLVoca:
    """The three Core ML models of an export directory. ``compute_units`` overrides the
    defaults per model, e.g. ``{"encoder": "CPU_ONLY", "decoder": "CPU_ONLY"}``."""

    def __init__(self, directory, compute_units=None, preload=False):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / METADATA).read_text(encoding="utf-8"))
        self.config = Config.from_dict(self.metadata["config"])
        self.voices = tuple(self.metadata["voices"])
        self.max_tokens, self.max_frames = self.metadata["max_tokens"], self.metadata["max_frames"]
        models = self.metadata["models"]
        self.encoder_buckets, self.decoder_buckets = tuple(models["encoder"]["buckets"]), tuple(models["decoder"]["buckets"])
        self.compute_units = {**COMPUTE_UNITS, **(compute_units or {})}
        self.phone_tokens = np.zeros(self.config.vocab_size, bool)
        self.phone_tokens[self.metadata["phone_tokens"]] = True
        self.vowel_tokens = np.zeros(self.config.vocab_size, bool)
        self.vowel_tokens[self.metadata["vowel_tokens"]] = True
        self._models = {}
        if preload:
            self.preload()

    def _model(self, kind, function=None):
        key = (kind, function)
        if key not in self._models:
            import coremltools as ct
            entry = self.metadata["models"][kind]
            path = self.directory / entry["compiled"]
            if not path.exists():
                path = compile_package(self.directory / entry["package"])
            self._models[key] = ct.models.CompiledMLModel(str(path), _compute_units(self.compute_units[kind]), function_name=function)
        return self._models[key]

    def preload(self):
        """Load every function now (else each one is loaded on first use, 0.05-0.2 s)."""
        for n in self.encoder_buckets:
            self._model("encoder", f"n{n}")
        for t in self.decoder_buckets:
            self._model("decoder", f"t{t}")
        self._model("vocoder")
        return self

    def voice_index(self, voice):
        return voice if isinstance(voice, (int, np.integer)) else self.voices.index(voice)

    def synthesize(self, tokens, stress, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, durations=None, noise=None):
        """One utterance: token ids and stress [N] -> (audio float32 [frames * hop], details).

        As ``TinyVoca.synthesize`` for one row (vowels kept voiced, no pitch jitter).
        ``durations`` [N] replace the predicted ones and ``noise`` [frames * hop] the seeded
        numpy noise (for comparisons with PyTorch). ``details`` has the durations, the log F0
        (after the shift), F0 in Hz, the vocoder parameters, and the time of each stage in seconds."""
        if not math.isfinite(speed) or not 0.25 <= speed <= 4 or not math.isfinite(pitch_shift) or not 0 <= pitch_range <= 3:
            raise ValueError("speed must be between 0.25 and 4, pitch_shift finite and pitch_range between 0 and 3")
        c = self.config
        tokens, stress = np.asarray(tokens, np.int64).reshape(-1), np.asarray(stress, np.int64).reshape(-1)
        n = len(tokens)
        if not 1 <= n <= self.max_tokens or len(stress) != n:
            raise ValueError("invalid token length")
        voice = self.voice_index(voice)
        times = {}
        clock = time.perf_counter()

        def lap(stage):
            nonlocal clock
            now = time.perf_counter()
            times[stage] = now - clock
            clock = now

        size = ane.bucket(n, self.encoder_buckets)
        out = ane.predict(self._model("encoder", f"n{size}"), ane.encoder_inputs(tokens, stress, voice, size, np.float16, np.float16))
        hidden, raw = out["hidden"][0, :, 0, :n], out["duration"][0, :, :n]
        lap("encoder")
        if durations is None:
            durations = V.durations(raw, tokens, speed, self.phone_tokens)
        durations = np.asarray(durations, np.int64).reshape(-1)
        if len(durations) != n:
            raise ValueError("one duration per token is needed")
        frames, position, index = V.expand(hidden, durations, self.max_frames)
        total = frames.shape[1]
        size = ane.bucket(total, self.decoder_buckets)
        inputs = {"frames": np.zeros((1, c.channels, 1, size), np.float16), "position": np.zeros((1, 3, 1, size), np.float16),
                  "voice": np.array([voice], np.float16), "frame_mask": np.zeros((1, size), np.float16),
                  "vowel_mask": np.zeros((1, size), np.float16), "pitch_range": np.array([pitch_range], np.float16)}
        inputs["frames"][0, :, 0, :total] = frames
        inputs["position"][0, :, 0, :total] = position
        inputs["frame_mask"][0, :total] = 1
        inputs["vowel_mask"][0, :total] = V.vowel_mask(tokens, index, self.vowel_tokens)
        lap("expand")
        out = ane.predict(self._model("decoder", f"t{size}"), inputs)
        pitch = out["pitch"][0, :total].astype(np.float32)
        harmonic, noise_params = out["harmonic"][0, :, :total], out["noise_params"][0, :, :total]
        lap("decoder")
        f0 = V.f0_from_pitch(pitch, pitch_shift, c)
        inputs = V.vocoder_inputs(harmonic, noise_params, f0, noise, seed, c)
        lap("source")
        audio = ane.predict(self._model("vocoder"), inputs)["audio"][0]
        lap("vocoder")
        if not np.isfinite(audio).all():
            raise RuntimeError("model produced non-finite audio")
        return audio, {"durations": durations, "pitch": pitch + np.float32(pitch_shift), "f0": f0, "harmonic": harmonic,
                       "noise_params": noise_params, "times": times}

    def speak_text(self, text, voice, speed=1.0, pitch_shift=0.0, seed=0, pitch_range=1.0, peak=PEAK):
        """Text of any length -> (float32 waveform, phonemized utterances), read as
        ``tiny_voca.infer.speak_text`` reads it: sentence by sentence, a full stop added
        when the text ends without a mark, too long sentences cut after commas, then
        between words, unreadable sentences skipped. ``peak`` scales the joined audio
        down when it is louder; None keeps it as it is."""
        limit = self.max_tokens - 12
        waves, spoken = [], []

        def read(piece):
            try:
                phones = phonemize(piece)
            except NothingToPronounce:
                return
            if len(phones) <= limit:
                try:
                    wave, _ = self.synthesize(phones.ids, phones.stress, voice, speed, pitch_shift, seed + len(waves), pitch_range)
                except UtteranceTooLong:
                    pass
                else:
                    waves.append(wave)
                    spoken.append(phones)
                    return
            pieces = shorter(piece)
            if not pieces:
                raise ValueError(f"one word is too long to speak: {piece[:40]!r}")
            for part in pieces:
                read(part)

        for sentence in sentences(finished(text)):
            read(sentence)
        if not waves:
            raise NothingToPronounce(f"nothing in the text could be read: {ENGLISH_ONLY}")
        wave = np.concatenate(waves)
        return (wave if peak is None else limit_peak(torch.from_numpy(wave), peak).numpy()), spoken

    def frontend_matches(self):
        return frontend_fingerprint() == self.metadata["frontend_fingerprint"]
