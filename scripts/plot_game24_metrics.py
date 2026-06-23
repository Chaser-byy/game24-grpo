import argparse
import json
import os
import re

import pandas as pd


CONSOLE_STEP_RE = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")
PAIR_RE = re.compile(r"(?P<key>[A-Za-z0-9_./-]+):(?P<value>-?\d+(?:\.\d+)?)")


def read_records(path):
    if path.endswith(".csv"):
        return pd.read_csv(path).to_dict("records")
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = CONSOLE_STEP_RE.search(line)
            if not match:
                continue
            record = {"step": int(match.group("step"))}
            for pair in PAIR_RE.finditer(match.group("body")):
                record[pair.group("key")] = float(pair.group("value"))
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="W&B CSV/JSONL export or a console log produced by tee")
    parser.add_argument("--output_dir", default="outputs/game24_plots")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    frame = pd.DataFrame(read_records(args.input))
    if frame.empty:
        raise SystemExit(f"No metric records found in {args.input}")
    if "step" not in frame.columns:
        frame["step"] = range(len(frame))

    groups = {
        "reward": ["critic/score/mean", "critic/rewards/mean", "game24/solved_rate", "game24/format_rate", "val/test_score/game24"],
        "kl": ["critic/kl", "actor/ppo_kl"],
        "length": ["response_length/mean", "response_length/max", "prompt_length/mean"],
        "optimization": ["actor/lr", "actor/grad_norm", "actor/pg_loss"],
        "grpo_group": [
            "game24/group_reward_std_mean",
            "game24/group_reward_zero_std_rate",
            "critic/advantages/mean",
            "critic/advantages/max",
            "critic/advantages/min",
        ],
    }

    for name, columns in groups.items():
        available = [column for column in columns if column in frame.columns]
        if not available:
            continue
        plt.figure(figsize=(8, 4.5))
        for column in available:
            plt.plot(frame["step"], frame[column], label=column)
        plt.xlabel("step")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(args.output_dir, f"{name}.png")
        plt.savefig(path, dpi=180)
        print(f"Saved {path}")
        plt.close()


if __name__ == "__main__":
    main()
