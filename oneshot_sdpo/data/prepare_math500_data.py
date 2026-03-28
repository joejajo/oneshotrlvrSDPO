"""
Prepare MATH-500 evaluation data.

Downloads HuggingFaceH4/MATH-500 and writes eval.parquet (500 rows) using
the same schema as train.parquet so the same data loader and reward function work.

Source dataset: HuggingFaceH4/MATH-500
  — this dataset ID is referenced directly in SDPO's reward router
    (verl/utils/reward_score/__init__.py) confirming it exists and is correct.

data_source is set to "lighteval/MATH" because:
  — that string is in SDPO's default reward router for math
  — our custom reward function handles it regardless of the value

Usage:
    python prepare_math500_data.py --output_dir data/datasets/math500
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


PROMPT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def build_row(problem: str, answer: str) -> dict:
    return {
        "data_source": "lighteval/MATH",
        "prompt": [{"role": "user", "content": problem + PROMPT_SUFFIX}],
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"split": "test"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where eval.parquet will be written",
    )
    parser.add_argument(
        "--hf_dataset",
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset id (default: HuggingFaceH4/MATH-500)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (default: test)",
    )
    args = parser.parse_args()

    print(f"Loading {args.hf_dataset} (split={args.split}) …")
    ds = load_dataset(args.hf_dataset, split=args.split, trust_remote_code=True)
    print(f"Downloaded {len(ds)} examples.  Columns: {ds.column_names}")

    # HuggingFaceH4/MATH-500 uses 'problem' and 'answer' column names.
    # Confirm they exist before proceeding.
    assert "problem" in ds.column_names, (
        f"Expected 'problem' column, got: {ds.column_names}. "
        "Adjust the field names below if the dataset schema differs."
    )
    assert "answer" in ds.column_names, (
        f"Expected 'answer' column, got: {ds.column_names}. "
        "Adjust the field names below if the dataset schema differs."
    )

    rows = [build_row(ex["problem"], str(ex["answer"])) for ex in ds]
    df = pd.DataFrame(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Columns : {df.columns.tolist()}")
    print(f"eval.parquet: {len(df)} rows  →  {out_path}")
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
    print("Row count check passed (500).")
    print(f"Example row:\n{df.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
