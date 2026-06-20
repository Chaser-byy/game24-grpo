#!/usr/bin/env python3
"""Download Qwen from ModelScope when Hugging Face is unavailable."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model from ModelScope")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from modelscope import snapshot_download
    except ModuleNotFoundError as error:
        raise SystemExit('ModelScope is required; run: pip install -e ".[download]"') from error

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model_dir = snapshot_download(args.model_id, local_dir=str(output_dir))
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
