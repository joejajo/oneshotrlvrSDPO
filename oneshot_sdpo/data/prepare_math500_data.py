"""
Prepare MATH-500 evaluation data.

Primary path: downloads math500.parquet from the official One-Shot-RLVR
repository (pinned commit) and writes eval.parquet (500 rows).

This is the exact file One-Shot-RLVR uses as data.val_files in its training
script, so it is guaranteed to have the correct verl/SDPO parquet schema.

Fallback path: if the download fails or validation fails, reconstructs
eval.parquet from HuggingFaceH4/MATH-500 via load_dataset.

Source file:
  https://raw.githubusercontent.com/ypwang61/One-Shot-RLVR/
    9a92ed1b92701bf2660256ec93b279347f759c69/
    data/test/math500.parquet
  Commit: 9a92ed1b92701bf2660256ec93b279347f759c69  ("fix data source", 2025-05-01)
  Size:   218 851 bytes
  SHA:    b99729ca68260afd09906277e203ff6b781abf1b (blob)

Usage:
    python prepare_math500_data.py --output_dir data/datasets/math500
"""

import argparse
import io
import os
import urllib.request

import pandas as pd


# ── Pinned source URL ──────────────────────────────────────────────────────────
_COMMIT = "9a92ed1b92701bf2660256ec93b279347f759c69"
_EVAL_URL = (
    f"https://raw.githubusercontent.com/ypwang61/One-Shot-RLVR/"
    f"{_COMMIT}/data/test/math500.parquet"
)
_EXPECTED_ROWS = 500
_MIN_PARQUET_BYTES = 100_000  # real file is 218 851 B; reject anything suspiciously small
_PARQUET_MAGIC = b"PAR1"
_REQUIRED_COLS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def _is_valid_parquet_bytes(data: bytes) -> tuple[bool, str]:
    """Return (ok, reason) — check magic bytes, size, and that it is not LFS/HTML."""
    if len(data) < _MIN_PARQUET_BYTES:
        return False, f"too small ({len(data)} bytes, expected ≥ {_MIN_PARQUET_BYTES})"
    if data[:4] != _PARQUET_MAGIC:
        if data[:7] == b"version":
            return False, "response is a Git LFS pointer, not the actual file"
        if data[:15].lower().lstrip().startswith(b"<!doctype") or b"<html" in data[:200].lower():
            return False, "response is an HTML page"
        return False, f"missing parquet magic (got {data[:4]!r})"
    return True, "ok"


def _download_parquet(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 200:
            raise IOError(f"HTTP {resp.status} for {url}")
        return resp.read()


def _try_download(url: str) -> tuple[pd.DataFrame | None, str]:
    """
    Try to download a parquet from url.
    Returns (DataFrame, "ok") on success, (None, reason) on failure.
    """
    try:
        data = _download_parquet(url)
    except Exception as exc:
        return None, f"download failed: {exc}"

    ok, reason = _is_valid_parquet_bytes(data)
    if not ok:
        return None, reason

    try:
        df = pd.read_parquet(io.BytesIO(data))
    except Exception as exc:
        return None, f"pandas could not read parquet: {exc}"

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        return None, f"missing required columns: {missing}"

    return df, "ok"


# ── Fallback: reconstruct from HuggingFaceH4/MATH-500 ────────────────────────

_PROMPT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


def _build_row(problem: str, answer: str) -> dict:
    return {
        "data_source": "lighteval/MATH",
        "prompt": [{"role": "user", "content": problem + _PROMPT_SUFFIX}],
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"split": "test"},
    }


def _fallback_huggingface(hf_dataset: str, split: str) -> pd.DataFrame:
    from datasets import load_dataset  # noqa: PLC0415
    print(f"  Loading {hf_dataset} (split={split}) from HuggingFace …")
    ds = load_dataset(hf_dataset, split=split, trust_remote_code=True)
    assert "problem" in ds.column_names and "answer" in ds.column_names, (
        f"Unexpected columns: {ds.column_names}"
    )
    rows = [_build_row(ex["problem"], str(ex["answer"])) for ex in ds]
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hf_dataset", default="HuggingFaceH4/MATH-500",
                        help="HuggingFace fallback dataset id")
    parser.add_argument("--hf_split", default="test",
                        help="HuggingFace fallback split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval.parquet")

    # ── Primary: download repo-shipped math500.parquet ─────────────────────────
    print(f"Downloading math500.parquet from pinned commit {_COMMIT[:12]}…")
    print(f"  URL: {_EVAL_URL}")

    df, reason = _try_download(_EVAL_URL)

    if df is not None and len(df) != _EXPECTED_ROWS:
        print(f"  WARNING: expected {_EXPECTED_ROWS} rows, got {len(df)} — falling back")
        df = None

    if df is not None:
        print(f"  Downloaded OK: {len(df)} rows, columns={df.columns.tolist()}")
        source = "repo-shipped (One-Shot-RLVR)"
    else:
        print(f"  Download/validation failed: {reason}")
        print(f"  Falling back to HuggingFace reconstruction ({args.hf_dataset}) …")
        df = _fallback_huggingface(args.hf_dataset, args.hf_split)
        source = f"reconstructed from {args.hf_dataset}"

    df.to_parquet(out_path, index=False)

    # ── Final assertions ───────────────────────────────────────────────────────
    assert len(df) == _EXPECTED_ROWS, f"Expected {_EXPECTED_ROWS} rows, got {len(df)}"
    assert _REQUIRED_COLS.issubset(set(df.columns)), (
        f"Missing columns: {_REQUIRED_COLS - set(df.columns)}"
    )

    print(f"\nSource  : {source}")
    print(f"Columns : {df.columns.tolist()}")
    print(f"eval.parquet: {len(df)} rows  →  {out_path}")
    print(f"Example row:\n{df.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
