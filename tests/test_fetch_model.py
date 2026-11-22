"""The gitignored ONNX model can be restored on any fresh clone."""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_model import (  # noqa: E402
    MODEL_MEMBER,
    extract_model,
    main,
    sha256,
    verify,
)


def bundle(model: bytes) -> bytes:
    """A ZIP with the same layout as the public demo kit."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(MODEL_MEMBER, model)
        archive.writestr("model/config.json", "{}")
    return buffer.getvalue()


class FetchModelTests(unittest.TestCase):
    def test_extracts_the_model_from_the_release_layout(self):
        model = b"a small fake onnx"
        self.assertEqual(extract_model(bundle(model)), model)

    def test_accepts_windows_backslash_members(self):
        model = b"a small fake onnx"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(MODEL_MEMBER.replace("/", "\\"), model)
        self.assertEqual(extract_model(buffer.getvalue()), model)

    def test_checksum_guard_rejects_a_corrupted_model(self):
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify(b"corrupted", "0" * 64)

    def test_main_writes_an_atomic_verified_file_without_network(self):
        model = b"a verified fake onnx"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "assets" / "model.onnx"
            with patch("scripts.fetch_model.download", return_value=bundle(model)):
                result = main(["--url", "unused", "--sha256", sha256(model),
                               "--out", str(destination)])
            self.assertEqual(result, 0)
            self.assertEqual(destination.read_bytes(), model)
            self.assertFalse(destination.with_suffix(".onnx.tmp").exists())


if __name__ == "__main__":
    unittest.main()
