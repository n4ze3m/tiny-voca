"""Text -> ids.json for tiny-voca-coreml. The phonemizer is Python only.

The text is split into sentences as ``tiny_voca.infer.speak_text`` does
(``sentences(finished(text))``, long pieces cut with ``shorter``); each
sentence is one utterance: ``[{"text": ..., "ids": [...], "stress": [...]}, ...]``.
With ``--each`` every argument is one utterance as it is (no splitting).

    python swift/phonemize.py "Hello there. How are you?" -o ids.json
"""

import argparse
import json
import sys
from pathlib import Path

from tiny_voca.config import Config
from tiny_voca.infer import finished, sentences, shorter
from tiny_voca.text import NothingToPronounce, phonemize

# The Swift runner has these token tables built in (Config.phoneTokens, Config.vowelTokens in Host.swift).
PHONE_TOKENS = [4] + list(range(17, 76))
VOWEL_TOKENS = list(range(43, 76))


def check_tables():
    from tiny_voca.coreml.vocoder import PHONE_TOKENS as phones, VOWEL_TOKENS as vowels
    if Config.vocab_size != 76 or list(phones.nonzero()[0]) != PHONE_TOKENS or list(vowels.nonzero()[0]) != VOWEL_TOKENS:
        sys.exit("the phoneme inventory changed: update Config in swift/Sources/TinyVoca/Host.swift and this script")


def utterances(text, limit=Config.max_tokens - 12):
    """``speak_text``'s pieces of ``text`` as (text, phonemized). Pieces above ``limit`` tokens are cut;
    a piece above max_frames frames is reported by the Swift runner (cut the text there)."""
    found = []

    def read(piece):
        try:
            phones = phonemize(piece)
        except NothingToPronounce:
            return
        if len(phones) <= limit:
            found.append((piece, phones))
            return
        pieces = shorter(piece)
        if not pieces:
            raise ValueError(f"one word is too long to speak: {piece[:40]!r}")
        for part in pieces:
            read(part)

    for sentence in sentences(finished(text)):
        read(sentence)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("text", nargs="*", help="text (default: standard input)")
    parser.add_argument("-o", "--output", type=Path, default=Path("ids.json"))
    parser.add_argument("--each", action="store_true", help="one utterance per argument, without sentence splitting")
    args = parser.parse_args()
    check_tables()
    texts = args.text or [sys.stdin.read()]
    if args.each:
        found = [(text, phonemize(text)) for text in texts]
    else:
        found = utterances(" ".join(texts))
    if not found:
        sys.exit("nothing in the text could be read")
    items = [{"text": text, "ids": list(phones.ids), "stress": list(phones.stress)} for text, phones in found]
    args.output.write_text(json.dumps(items))
    print(f"{args.output}: {len(items)} utterance(s), {sum(len(i['ids']) for i in items)} tokens")


if __name__ == "__main__":
    main()
