"""Speech with the Core ML models (Apple Neural Engine). Same options as ``tiny_voca.infer``.

    python -m tiny_voca.coreml --checkpoint tiny-voca.pt --voice F2 --text "Hello!" --output hello.wav
    python -m tiny_voca.coreml --checkpoint tiny-voca.pt --voice F2 --text "Hello!"     # plays the audio

The checkpoint is converted on its first use and the models are kept in a cache
(``TINY_VOCA_CACHE``, else ~/Library/Caches/tiny-voca/coreml). ``--models``
takes a directory written by ``python -m tiny_voca.coreml.export`` instead;
``python -m tiny_voca.coreml.plan`` shows where each op runs.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ..text import FRONTEND_VARIABLE, FRONTENDS, NothingToPronounce, frontend_fingerprint, set_frontend

UNITS = ("CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU", "ALL")


def _match_frontend(fingerprint):
    """Select the front end whose fingerprint the export recorded (``infer.match_frontend`` for a checkpoint)."""
    for name in FRONTENDS:
        set_frontend(name)
        try:
            if frontend_fingerprint() == fingerprint:
                return name
        except ImportError:
            continue
    set_frontend(FRONTENDS[0])
    return None


def models_directory(args):
    """``--models``, or the cached export of ``--checkpoint`` (made now on the first use)."""
    from .runtime import cached_models
    return args.models or cached_models(args.checkpoint, log=lambda line: print(line, file=sys.stderr, flush=True))


def speak(args, parser):
    import soundfile as sf
    import torch
    from ..infer import ENGLISH_ONLY, PEAK, limit_peak
    from .runtime import CoreMLVoca
    units = {"encoder": args.compute_units, "decoder": args.compute_units} if args.compute_units else None
    voca = CoreMLVoca(models_directory(args), units)
    if args.frontend:
        set_frontend(args.frontend)
    elif FRONTEND_VARIABLE not in os.environ:
        _match_frontend(voca.metadata["frontend_fingerprint"])
    if not args.allow_frontend_mismatch and not voca.frontend_matches():
        parser.exit(1, "the models were exported from a checkpoint trained with a different phoneme inventory/front end\n")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    try:
        raw, spoken = voca.speak_text(text, args.voice, args.speed, args.pitch_shift, pitch_range=args.pitch_range, peak=None)
    except NothingToPronounce:
        parser.exit(1, f"{ENGLISH_ONLY}. Nothing in the text could be read, so no file was written.\n")
    raw = torch.from_numpy(raw)
    wave = limit_peak(raw)
    for phones in spoken:
        print(" ".join(phones.symbols))
    loudest = float(raw.abs().max())
    scaled = f", scaled down to {PEAK}" if wave is not raw else ""
    summary = f"{len(wave)/voca.config.sample_rate:.2f}s in {len(spoken)} utterance{'s' if len(spoken) != 1 else ''}; peak={loudest:.4f}{scaled}"
    if args.output is None:
        print(f"Playing {summary}", flush=True)
        play(wave.numpy(), voca.config.sample_rate)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Float samples, never above PEAK: a later conversion to 16-bit does not clip.
    sf.write(args.output, wave.numpy(), voca.config.sample_rate, subtype="FLOAT")
    print(f"Saved {summary}")


def play(wave, sample_rate):
    """Play ``wave`` with macOS's ``afplay`` (Core ML is macOS only, so it is always there)
    from a temporary WAV file, and wait until it ends. Ctrl-C stops the sound."""
    import soundfile as sf
    with tempfile.TemporaryDirectory(prefix="tiny-voca-") as folder:
        path = Path(folder) / "speech.wav"
        sf.write(path, wave, sample_rate, subtype="FLOAT")
        player = subprocess.Popen(["afplay", str(path)])
        try:
            player.wait()
        except KeyboardInterrupt:
            player.terminate()
            player.wait()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tiny_voca.coreml",
                                     description="Generate speech with tiny-voca on Core ML (Apple Neural Engine)")
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--checkpoint", type=Path, help="Inference checkpoint; converted on first use and kept in a cache")
    models.add_argument("--models", type=Path, help="Directory written by python -m tiny_voca.coreml.export")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text of any length; it is read sentence by sentence")
    source.add_argument("--text-file", type=Path, help="Read the text from a file (UTF-8)")
    parser.add_argument("--voice", choices=("F2", "M2"), default="F2")
    parser.add_argument("--output", type=Path, help="WAV file to write; without it the audio is played")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--pitch-shift", type=float, default=0, help="Added to log F0 (0.058 is about one semitone)")
    parser.add_argument("--pitch-range", type=float, default=1, help="Scale pitch movement around the utterance mean (1 = as predicted)")
    parser.add_argument("--compute-units", choices=UNITS, help="For the encoder and decoder (default CPU_AND_NE)")
    parser.add_argument("--frontend", choices=FRONTENDS, help="Phonemizer (default: TINY_VOCA_FRONTEND, else the one the models were exported with)")
    parser.add_argument("--allow-frontend-mismatch", action="store_true",
                        help="Speak although the models were trained on the other front end's tokens")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.exit(1, "Core ML runs on macOS only\n")
    speak(args, parser)


if __name__ == "__main__":
    main()
