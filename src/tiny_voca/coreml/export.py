"""Core ML export of a checkpoint to a directory of your choice.

    python -m tiny_voca.coreml.export --checkpoint tiny-voca.pt --output coreml

Writes Encoder, Decoder and Vocoder packages, their compiled ``.mlmodelc`` and
``tiny-voca-coreml.json``. ``python -m tiny_voca.coreml --models coreml`` speaks
with them. Without this step, ``python -m tiny_voca.coreml --checkpoint`` makes
the same export once in a cache.
"""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    from .ane import DECODER_BUCKETS, ENCODER_BUCKETS
    parser = argparse.ArgumentParser(prog="python -m tiny_voca.coreml.export",
                                     description="Convert a tiny-voca checkpoint to Encoder, Decoder and Vocoder Core ML packages")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the packages and tiny-voca-coreml.json")
    parser.add_argument("--encoder-buckets", type=int, nargs="+", default=list(ENCODER_BUCKETS), help="Token lengths, one function each")
    parser.add_argument("--decoder-buckets", type=int, nargs="+", default=list(DECODER_BUCKETS), help="Frame lengths, one function each")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.exit(1, "Core ML runs on macOS only\n")
    from ..infer import load_model, match_frontend
    from .runtime import export_models
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, "cpu")
    export_models(model, args.checkpoint, args.output, tuple(args.encoder_buckets), tuple(args.decoder_buckets))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
