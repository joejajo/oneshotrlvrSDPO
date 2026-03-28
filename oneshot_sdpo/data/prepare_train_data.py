"""
Prepare one-shot RLVR training data.

Creates train.parquet (128 rows) and val.parquet (16 rows), each containing
identical copies of π₁ (the single training problem).

Schema matches what SDPO's NaiveRewardManager expects:
  data_source  : str
  prompt       : list[dict]  — chat-format messages
  ability      : str
  reward_model : dict        — must contain "ground_truth" and "style"
  extra_info   : dict

Usage:
    python prepare_train_data.py --output_dir data/datasets/train
"""

import argparse
import json
import os

import pandas as pd


def build_row(problem: str, answer: str, split: str) -> dict:
    return {
        "data_source": "lighteval/MATH",
        "prompt": [{"role": "user", "content": problem}],
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"split": split},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where train.parquet and val.parquet will be written",
    )
    parser.add_argument(
        "--pi1_path",
        default=os.path.join(os.path.dirname(__file__), "pi1_example.json"),
        help="Path to pi1_example.json (default: same directory as this script)",
    )
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=16)
    args = parser.parse_args()

    with open(args.pi1_path) as f:
        pi1 = json.load(f)

    problem = pi1["problem"]
    answer = pi1["answer"]

    os.makedirs(args.output_dir, exist_ok=True)

    train_rows = [build_row(problem, answer, "train") for _ in range(args.n_train)]
    val_rows = [build_row(problem, answer, "val") for _ in range(args.n_val)]

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Columns : {train_df.columns.tolist()}")
    print(f"train.parquet: {len(train_df)} rows  →  {train_path}")
    print(f"val.parquet  : {len(val_df)} rows  →  {val_path}")
    print(f"Example row  :\n{train_df.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
