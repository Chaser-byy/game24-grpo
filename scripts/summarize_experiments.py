#!/usr/bin/env python3
"""Combine ID, hard, unsolvable, and OOD summaries into one report table."""

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "accuracy_at_1",
    "strict_format_rate",
    "syntax_rate",
    "legal_number_rate",
    "correct_abstention_rate",
    "false_claim_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation summaries")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for every evaluation population",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for value in args.summary:
        if "=" not in value:
            raise SystemExit("--summary must use LABEL=PATH")
        label, path = value.split("=", 1)
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(
            {
                "label": label,
                "model": summary.get("model"),
                "data": summary.get("data"),
                "fingerprint": summary.get("dataset_fingerprint"),
                "total": summary.get("total"),
                **{field: summary.get(field) for field in FIELDS},
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} experiment rows to {output}")


if __name__ == "__main__":
    main()
