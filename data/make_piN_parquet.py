"""
Create a multi-example training parquet for One-Shot-RLVR + SDPO.

Generates data/pi8_r16.parquet: 8 diverse math problems × 16 copies each = 128 rows
(same total size as pi1_r128.parquet so batch_size=128 requires no changes).

Each problem is chosen to cover a different MATH topic, at the right difficulty
level for Qwen2.5-Math-1.5B (~30-70% accuracy — variance-worthy).

Usage:
    python data/make_piN_parquet.py
    python data/make_piN_parquet.py --n_copies 32 --output data/pi8_r32.parquet

Optionally, if you have the One-Shot-RLVR DSR-sub data on HuggingFace:
    python data/make_piN_parquet.py --from_hf --hf_dataset ypwang61/One-Shot-RLVR
"""

import argparse
import json
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Hardcoded problem set — 8 diverse problems with verified exact answers
#
# Selection criteria:
#   - Different MATH topics (algebra, number theory, geometry, combinatorics)
#   - Multi-step reasoning required (not trivially one-step)
#   - Clean exact numerical answer (integer, simple fraction, or decimal)
#   - Difficulty calibrated for Qwen2.5-Math-1.5B (~MATH level 2-4)
#
# Answers verified manually. Each answer can be checked with grade_answer_sympy.
# ---------------------------------------------------------------------------
PROBLEMS = [
    {
        # P1 — Joint/combined variation (original π₁, One-Shot-RLVR)
        # Topic: Algebra — proportional reasoning
        "problem": (
            r"The pressure \( P \) exerted by wind on a sail varies jointly as the area "
            r"\( A \) of the sail and the cube of the wind's velocity \( V \). When the "
            r"velocity is \( 8 \) miles per hour, the pressure on a sail of \( 2 \) square "
            r"feet is \( 4 \) pounds. Find the wind velocity when the pressure on \( 4 \) "
            r"square feet of sail is \( 32 \) pounds. "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "12.8",
    },
    {
        # P2 — Vieta's formulas
        # Topic: Algebra — quadratic roots
        # Derivation: r+s=5, rs=3; r²+s²=(r+s)²-2rs = 25-6 = 19
        "problem": (
            r"Let \( r \) and \( s \) be the two roots of the equation \( x^2 - 5x + 3 = 0 \). "
            r"What is the value of \( r^2 + s^2 \)? "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "19",
    },
    {
        # P3 — Counting divisors using prime factorisation
        # Topic: Number theory — multiplicative functions
        # Derivation: 360=2³·3²·5¹; σ(360)=(1+2+4+8)(1+3+9)(1+5)=15·13·6=1170
        "problem": (
            r"Find the sum of all positive divisors of 360. "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "1170",
    },
    {
        # P4 — Heron's formula
        # Topic: Geometry — triangle area
        # Derivation: s=(13+14+15)/2=21; area=√(21·8·7·6)=√7056=84
        "problem": (
            r"A triangle has side lengths 13, 14, and 15. "
            r"What is the area of the triangle? "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "84",
    },
    {
        # P5 — Complementary counting
        # Topic: Combinatorics — restricted arrangements
        # Derivation: total=4!=24; adjacent pairs: treat {A,B} as 1 → 3!·2!=12;
        #             non-adjacent = 24-12 = 12
        "problem": (
            r"In how many ways can 4 people be arranged in a line if two specific "
            r"people, Alice and Bob, must not stand next to each other? "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "12",
    },
    {
        # P6 — System of equations (word problem)
        # Topic: Algebra — linear systems
        # Derivation: N+P=10, 3N+1.5P=21 → N=4
        "problem": (
            r"A shop sells notebooks for \$3 each and pens for \$1.50 each. "
            r"A customer buys exactly 10 items and pays exactly \$21. "
            r"How many notebooks did the customer buy? "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "4",
    },
    {
        # P7 — Modular arithmetic
        # Topic: Number theory — patterns of remainders
        # Derivation: 7≡-1 (mod 8), so 7^100≡(-1)^100=1 (mod 8)
        "problem": (
            r"What is the remainder when \( 7^{100} \) is divided by 8? "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "1",
    },
    {
        # P8 — Geometric series
        # Topic: Sequences & series
        # Derivation: S=2·(3^6-1)/(3-1)=2·728/2=728
        "problem": (
            r"The first term of a geometric sequence is 2 and the common ratio is 3. "
            r"What is the sum of the first 6 terms? "
            r"Let's think step by step and output the final answer within \boxed{}."
        ),
        "answer": "728",
    },
]


def make_prompt(problem_text: str) -> list:
    """Return the chat-message list expected by verl's data loader."""
    return [{"role": "user", "content": problem_text}]


def make_parquet(
    problems: list[dict],
    n_copies: int,
    output_path: str,
    data_source: str = "deepscaler",
) -> None:
    rows = []
    row_idx = 0
    for p in problems:
        for _ in range(n_copies):
            rows.append(
                {
                    "data_source": data_source,
                    "prompt": make_prompt(p["problem"]),
                    "ability": "math",
                    # Match pi1_r128.parquet schema: reward_model has ground_truth + style
                    "reward_model": {"ground_truth": p["answer"], "style": "rule"},
                    # Match pi1_r128.parquet schema: extra_info has index + split
                    "extra_info": {"index": row_idx, "split": "train"},
                }
            )
            row_idx += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Written {len(df)} rows ({len(problems)} problems × {n_copies} copies) → {output_path}")
    for i, p in enumerate(problems):
        print(f"  P{i+1}: answer={p['answer']:>6}  {p['problem'][:60]}…")


def make_from_hf(hf_dataset: str, n_examples: int, n_copies: int, output_path: str) -> bool:
    """
    Attempt to load top-N variance examples from a HuggingFace dataset.
    Returns True if successful, False if HuggingFace is not available.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
        print(f"Loading {hf_dataset} from HuggingFace …")
        ds = load_dataset(hf_dataset, split="train")
        df = ds.to_pandas()

        # One-Shot-RLVR schema: columns include 'prompt', 'answer', optionally 'variance_score'
        if "variance_score" in df.columns:
            df = df.sort_values("variance_score", ascending=False)
            print(f"Sorted by variance_score; top {n_examples} selected.")
        else:
            print(f"No variance_score column; using first {n_examples} rows.")

        df = df.head(n_examples).reset_index(drop=True)

        rows = []
        row_idx = 0
        for _, row in df.iterrows():
            problem_text = row.get("problem", row.get("prompt", ""))
            if isinstance(problem_text, list):
                problem_text = problem_text[0].get("content", str(problem_text))
            answer = str(row.get("answer", row.get("ground_truth", "")))
            for _ in range(n_copies):
                rows.append(
                    {
                        "data_source": "deepscaler",
                        "prompt": make_prompt(problem_text),
                        "ability": "math",
                        "reward_model": {"ground_truth": answer, "style": "rule"},
                        "extra_info": {"index": row_idx, "split": "train"},
                    }
                )
                row_idx += 1

        out_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_df.to_parquet(output_path, index=False)
        print(f"Written {len(out_df)} rows from HuggingFace → {output_path}")
        return True
    except Exception as e:
        print(f"HuggingFace load failed ({e}); falling back to hardcoded problems.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create multi-example SDPO training parquet")
    parser.add_argument(
        "--n_copies", type=int, default=16,
        help="Number of copies of each problem (default: 16 → 8×16=128 rows total)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output parquet path (default: data/pi{N}_{src}_r{copies}.parquet)",
    )
    parser.add_argument(
        "--from_hf", action="store_true",
        help="Try to load high-variance examples from HuggingFace first",
    )
    parser.add_argument(
        "--hf_dataset", default="ypwang61/One-Shot-RLVR",
        help="HuggingFace dataset name (used with --from_hf)",
    )
    parser.add_argument(
        "--hf_n_examples", type=int, default=8,
        help="Number of examples to pull from HuggingFace (used with --from_hf)",
    )
    args = parser.parse_args()

    n_problems = len(PROBLEMS)
    n_copies = args.n_copies
    total = n_problems * n_copies

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(script_dir, f"pi{n_problems}_r{n_copies}.parquet")

    if args.from_hf:
        success = make_from_hf(
            hf_dataset=args.hf_dataset,
            n_examples=args.hf_n_examples,
            n_copies=n_copies,
            output_path=output_path,
        )
        if success:
            return

    # Fall back to (or use directly) hardcoded problems
    print(f"Building {n_problems} hardcoded problems × {n_copies} copies = {total} rows")
    make_parquet(PROBLEMS, n_copies=n_copies, output_path=output_path)


if __name__ == "__main__":
    main()
