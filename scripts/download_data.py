#!/usr/bin/env python3
"""Download Game of 24 training data to a local JSONL file."""

import argparse
import json
import os
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a dataset to local JSONL")
    parser.add_argument("--provider", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--dataset", help="Dataset ID; a provider-specific default is used")
    parser.add_argument("--subset", default="default", help="ModelScope subset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/raw/game24_train.jsonl")
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint or mirror; use https://huggingface.co for the official Hub",
    )
    parser.add_argument("--limit", type=int, help="Optional number of records to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    if args.provider == "modelscope":
        dataset_id = args.dataset or "cqupthzr/game24"
        records = _load_modelscope(dataset_id, args.subset, args.split)
    else:
        dataset_id = args.dataset or "nlile/24-game"
        records = _load_huggingface(dataset_id, args.split, args.endpoint)

    if args.limit is not None:
        records = islice(records, args.limit)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as file:
        for item in records:
            record = dict(item)
            if dataset_id == "cqupthzr/game24":
                record["solvable"] = True
            record["source"] = dataset_id
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1

    print(f"Saved {count} records from {dataset_id} to {output}")


def _load_modelscope(dataset_id: str, subset: str, split: str) -> Iterable[dict[str, Any]]:
    print(f"Downloading {dataset_id} ({subset}/{split}) from ModelScope")
    try:
        owner, name = dataset_id.split("/", 1)
    except ValueError as error:
        raise SystemExit("ModelScope dataset ID must use owner/name format") from error

    page = 1
    page_size = 100
    while True:
        query = urlencode(
            {
                "Owner": owner,
                "Name": name,
                "Revision": "master",
                "Subset": subset,
                "Split": split,
                "PageSize": page_size,
                "PageNumber": page,
            }
        )
        url = f"https://modelscope.cn/api/v1/datasets/preview?{query}"
        with urlopen(url, timeout=60) as response:
            payload = json.load(response)

        if payload.get("Code") != 200:
            raise RuntimeError(payload.get("Message", "ModelScope download failed"))
        rows = payload.get("Data") or []
        for row in rows:
            yield json.loads(row["Content"])

        total = int(payload.get("TotalCount") or 0)
        if not rows or page * page_size >= total:
            break
        page += 1


def _load_huggingface(dataset_id: str, split: str, endpoint: str) -> Iterable[dict[str, Any]]:
    # huggingface_hub reads this setting when datasets is imported.
    os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise SystemExit('datasets is required; run: pip install -e ".[data]"') from error

    print(f"Downloading {dataset_id} ({split}) from {endpoint}")
    return load_dataset(dataset_id, split=split)


if __name__ == "__main__":
    main()
