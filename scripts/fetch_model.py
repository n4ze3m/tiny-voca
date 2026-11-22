#!/usr/bin/env python3
"""Fetch the Arabic ONNX voice from the public release and verify its hash.

A fresh clone cannot contain the 21 MB model: ``*.onnx`` is gitignored. This
script is the single source of truth used by the Dockerfile, CI and developers,
so a build never depends on a file that only exists on one workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = (
    "https://github.com/BenkadourKhaled/tiny-voca/releases/download/"
    "v0.1-ar-demo-kit/tiny-voca-ar-demo-kit.zip"
)
MODEL_SHA256 = "121d285ce938dab1b3720f77ed02121d80f83ae2ec23d33ec13e095f321390d4"
MODEL_MEMBER = "model/model.onnx"
DEFAULT_OUTPUT = Path("server/assets/model.onnx")


def sha256(data: bytes) -> str:
    """Hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def download(url: str, timeout: int = 300) -> bytes:
    """Download the public release bundle; redirects are followed by urllib."""
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-voca-build"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_model(archive: bytes, member: str = MODEL_MEMBER) -> bytes:
    """Read the ONNX file out of the demo-kit ZIP.

    The kit was zipped on Windows: member names use backslashes there, but a
    ZIP built elsewhere uses forward slashes. Accept both so the Docker build
    and CI work on any host.
    """
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = {name.replace("\\", "/"): name for name in bundle.namelist()}
        wanted = member.replace("\\", "/")
        if wanted not in names:
            raise KeyError(f"{wanted} not in archive; found {list(names)[:5]}")
        return bundle.read(names[wanted])


def verify(data: bytes, expected: str) -> None:
    """Fail loudly if the model is not exactly the pinned artefact."""
    actual = sha256(data)
    if actual != expected.lower():
        raise ValueError(f"model checksum mismatch: expected {expected}, got {actual}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("TINY_VOCA_MODEL_URL", RELEASE_URL))
    parser.add_argument("--sha256", default=os.getenv("TINY_VOCA_MODEL_SHA256", MODEL_SHA256))
    parser.add_argument("--member", default=MODEL_MEMBER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    archive = download(args.url)
    model = extract_model(archive, args.member)
    verify(model, args.sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_bytes(model)
    temporary.replace(args.out)                      # atomic: no half-written model
    print(f"model ready: {args.out} ({len(model)} bytes, sha256 {sha256(model)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
