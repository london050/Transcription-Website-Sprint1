#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transcribe_pipeline import process_file


def main():
    parser = argparse.ArgumentParser(
        description="Local audio transcriber + (optional) translator."
    )
    parser.add_argument("input_file", help="Path to .mp3/.mp4/.wav file")
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model: tiny, base, small, medium, large",
    )
    parser.add_argument(
        "--whisper-lang",
        default=None,
        help="Force language code (e.g. en, fr). Default: auto-detect.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate transcript using MarianMT.",
    )
    parser.add_argument(
        "--marian-model",
        default=None,
        help="Marian model name, e.g. Helsinki-NLP/opus-mt-en-fr",
    )

    args = parser.parse_args()

    paths = process_file(
        Path(args.input_file),
        whisper_model=args.whisper_model,
        whisper_lang=args.whisper_lang,
        translate=args.translate,
        marian_model=args.marian_model,
    )

    # Print JSON so other tools can consume it if needed
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
