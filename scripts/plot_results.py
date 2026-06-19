#!/usr/bin/env python3
"""Create report-ready plots from one Game of 24 experiment."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GRPO training and evaluation results")
    parser.add_argument("--train-metrics", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--grpo-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def series(records: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    points = [
        (float(record.get("step", index)), float(record[key]))
        for index, record in enumerate(records)
        if isinstance(record.get(key), (int, float)) and not isinstance(record[key], bool)
    ]
    return [point[0] for point in points], [point[1] for point in points]


def metric_label(key: str) -> str:
    name = key.removeprefix("rewards/").removesuffix("/mean")
    return name.replace("_", " ").title()


def main() -> None:
    args = parse_args()

    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise SystemExit('matplotlib is required; run: pip install -e ".[analysis]"') from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.train_metrics)

    reward_keys = []
    for candidate in ("reward", "total_reward", "rewards/total/mean"):
        if any(isinstance(record.get(candidate), (int, float)) for record in records):
            reward_keys.append(candidate)
            break
    reward_keys.extend(
        sorted(
            {
                key
                for record in records
                for key, value in record.items()
                if key.startswith("rewards/")
                and "std" not in key
                and isinstance(value, (int, float))
                and key not in reward_keys
            }
        )
    )

    if reward_keys:
        plt.figure(figsize=(8, 5))
        for key in reward_keys:
            steps, values = series(records, key)
            if values:
                label = "Total Reward" if key in {"reward", "total_reward"} else metric_label(key)
                plt.plot(steps, values, marker="o", markersize=3, label=label)
        plt.title("GRPO Training Rewards")
        plt.xlabel("Training Step")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_rewards.png", dpi=200)
        plt.close()
    else:
        print("Skipping reward plot: no reward metrics found")

    loss_steps, loss_values = series(records, "loss")
    if loss_values:
        plt.figure(figsize=(8, 5))
        plt.plot(loss_steps, loss_values, marker="o", markersize=3, color="tab:red")
        plt.title("GRPO Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss.png", dpi=200)
        plt.close()
    else:
        print("Skipping loss plot: no loss metric found")

    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    grpo = json.loads(Path(args.grpo_summary).read_text(encoding="utf-8"))
    comparison_fields = [
        ("accuracy", "Accuracy"),
        ("extraction_rate", "Extraction Rate"),
        ("legal_rate", "Legal Number Usage"),
        ("average_reward", "Average Reward"),
    ]
    comparison = [
        (label, float(baseline[key]), float(grpo[key]))
        for key, label in comparison_fields
        if isinstance(baseline.get(key), (int, float))
        and isinstance(grpo.get(key), (int, float))
    ]
    missing = [
        key for key, _ in comparison_fields if key not in baseline or key not in grpo
    ]
    if missing:
        print(f"Skipping missing comparison metrics: {', '.join(missing)}")

    csv_path = output_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "baseline", "grpo"])
        writer.writerows(comparison)

    if comparison:
        labels = [item[0] for item in comparison]
        baseline_values = [item[1] for item in comparison]
        grpo_values = [item[2] for item in comparison]
        positions = list(range(len(labels)))
        width = 0.36

        plt.figure(figsize=(9, 5))
        left = plt.bar(
            [position - width / 2 for position in positions],
            baseline_values,
            width,
            label="Baseline",
        )
        right = plt.bar(
            [position + width / 2 for position in positions],
            grpo_values,
            width,
            label="After GRPO",
        )
        plt.bar_label(left, fmt="%.3f", padding=3)
        plt.bar_label(right, fmt="%.3f", padding=3)
        plt.xticks(positions, labels)
        plt.ylabel("Value")
        plt.title("Baseline vs. GRPO Evaluation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "before_after_comparison.png", dpi=200)
        plt.close()
    else:
        print("Skipping comparison plot: no shared metrics found")

    print(f"Plots and summary saved to {output_dir}")


if __name__ == "__main__":
    main()
