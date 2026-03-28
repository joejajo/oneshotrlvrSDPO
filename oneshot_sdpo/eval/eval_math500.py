"""
Standalone MATH-500 evaluation for One-Shot-RLVR + SDPO checkpoints.

Loads a trained checkpoint, runs greedy vLLM decoding on eval.parquet (500 rows),
grades each answer with the same reward logic as training, and writes an audit JSONL.

Usage:
    python eval_math500.py \\
        --checkpoint output/checkpoints/global_step_500 \\
        --eval_data   data/datasets/math500/eval.parquet \\
        --step        500 \\
        --output_dir  output/eval_results

Checkpoint convention:
    SDPO saves checkpoints as:
        <trainer.default_local_dir>/global_step_{N}/actor/
    Pass either the "global_step_{N}" directory or the "actor" subdirectory;
    this script appends "/actor" if that subdirectory exists.

Output:
    <output_dir>/eval_step_<N>.jsonl
    One JSON object per line, fields:
        step, prompt, response, extracted_answer, ground_truth, reward, is_correct
    Final line is a summary record:
        {"summary": true, "step": N, "accuracy": ..., "correct": ..., "total": 500}
"""

import argparse
import json
import os
import sys

import pandas as pd

# Reward utilities re-used from math_reward.py
# Locate math_reward.py relative to this script (../reward/math_reward.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_REWARD_MODULE = os.path.join(_HERE, "..", "reward", "math_reward.py")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("math_reward", _REWARD_MODULE)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
extract_answer = _mod.extract_answer
grade_answer_mathd = _mod.grade_answer_mathd
grade_answer_sympy = _mod.grade_answer_sympy


def resolve_checkpoint(ckpt_path: str) -> str:
    """Return the actor model directory given a checkpoint root or actor subdir."""
    actor_sub = os.path.join(ckpt_path, "actor")
    if os.path.isdir(actor_sub):
        return actor_sub
    if os.path.isdir(ckpt_path):
        return ckpt_path
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}\n"
        "Expected either 'global_step_N/' or 'global_step_N/actor/'"
    )


def grade(model_answer: str, ground_truth: str) -> bool:
    return grade_answer_mathd(model_answer, ground_truth) or \
           grade_answer_sympy(model_answer, ground_truth)


def build_prompts(df: pd.DataFrame) -> list[str]:
    """Extract the user content string from each prompt cell."""
    prompts = []
    for cell in df["prompt"]:
        # cell is list[dict] with {"role": "user", "content": "..."}
        if isinstance(cell, list):
            text = cell[0]["content"]
        else:
            text = str(cell)
        prompts.append(text)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="MATH-500 eval for SDPO checkpoints")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint root (global_step_N/) or actor subdir",
    )
    parser.add_argument(
        "--eval_data", required=True,
        help="Path to eval.parquet",
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Training step number (used in output filename and JSON; inferred from checkpoint path if omitted)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(
            "/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output", "eval_results"
        ),
        help="Directory where eval_step_N.jsonl is written",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=3072,
        help="Maximum tokens to generate per sample (default: 3072)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="vLLM GPU memory fraction (default: 0.9)",
    )
    args = parser.parse_args()

    # ── Resolve step number ───────────────────────────────────────────────────
    step = args.step
    if step is None:
        # Try to infer from path: "global_step_500" → 500
        basename = os.path.basename(args.checkpoint.rstrip("/"))
        if basename.startswith("global_step_"):
            try:
                step = int(basename.split("global_step_")[1])
            except ValueError:
                step = 0
        else:
            step = 0

    # ── Load eval data ────────────────────────────────────────────────────────
    print(f"Loading eval data from {args.eval_data} …")
    df = pd.read_parquet(args.eval_data)
    assert len(df) == 500, f"Expected 500 eval rows, got {len(df)}"
    ground_truths = [
        str(row["reward_model"]["ground_truth"]) for _, row in df.iterrows()
    ]
    prompts = build_prompts(df)
    print(f"Loaded {len(df)} eval examples.")

    # ── Load model via vLLM ───────────────────────────────────────────────────
    actor_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading model from {actor_path} …")

    from vllm import LLM, SamplingParams  # noqa: PLC0415  (late import — vLLM may not be available at import time)
    llm = LLM(
        model=actor_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    print("Running greedy decoding …")
    outputs = llm.generate(prompts, sampling_params)

    # ── Grade + collect records ───────────────────────────────────────────────
    records = []
    n_correct = 0
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        gt = ground_truths[i]
        model_answer = extract_answer(response)
        if model_answer is not None:
            is_correct = grade(model_answer, gt)
        else:
            is_correct = False
        reward = 1.0 if is_correct else 0.0
        if is_correct:
            n_correct += 1
        records.append({
            "step": step,
            "prompt": prompts[i],
            "response": response,
            "extracted_answer": model_answer,
            "ground_truth": gt,
            "reward": reward,
            "is_correct": is_correct,
        })

    accuracy = n_correct / len(records)
    summary = {
        "summary": True,
        "step": step,
        "accuracy": accuracy,
        "correct": n_correct,
        "total": len(records),
    }

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"eval_step_{step}.jsonl")
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps(summary) + "\n")

    print(f"\nMATH-500 accuracy at step {step}: {accuracy:.4f}  ({n_correct}/{len(records)})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
