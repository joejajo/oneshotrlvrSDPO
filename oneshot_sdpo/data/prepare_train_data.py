"""
Prepare one-shot RLVR training data.

Primary path: downloads pi1_r128.parquet from the official One-Shot-RLVR
repository (pinned commit) and writes:
  train.parquet  — all 128 rows as-is
  val.parquet    — first 16 rows (no repo-shipped val split exists)

Fallback path: if the download fails or validation fails, reconstructs
train.parquet and val.parquet from pi1_example.json (original generate approach).

Source file:
  https://raw.githubusercontent.com/ypwang61/One-Shot-RLVR/
    fa0677487c8aec63f7a87a9568de4bf2c47205b4/
    data/train/one_shot_rlvr/pi1_r128.parquet
  Commit: fa0677487c8aec63f7a87a9568de4bf2c47205b4  ("init", 2025-04-30)
  Size:   6 732 bytes
  SHA:    94e10d72fb6d4c9f01941b0e9cb3e83d6bb5fe69 (blob)

Usage:
    python prepare_train_data.py --output_dir data/datasets/train
"""

import argparse
import io
import json
import os
import urllib.request

import pandas as pd


# ── Pinned source URL ──────────────────────────────────────────────────────────
_COMMIT = "fa0677487c8aec63f7a87a9568de4bf2c47205b4"
_TRAIN_URL = (
    f"https://raw.githubusercontent.com/ypwang61/One-Shot-RLVR/"
    f"{_COMMIT}/data/train/one_shot_rlvr/pi1_r128.parquet"
)
_EXPECTED_ROWS_TRAIN = 128
_EXPECTED_ROWS_VAL = 16
_MIN_PARQUET_BYTES = 1_000   # real parquet is ≥ 6 732 B; reject anything smaller
_PARQUET_MAGIC = b"PAR1"     # parquet files start and end with these 4 bytes
_REQUIRED_COLS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def _is_valid_parquet_bytes(data: bytes) -> tuple[bool, str]:
    """Return (ok, reason) — check magic bytes, size, and that it is not LFS/HTML."""
    if len(data) < _MIN_PARQUET_BYTES:
        return False, f"too small ({len(data)} bytes, expected ≥ {_MIN_PARQUET_BYTES})"
    if data[:4] != _PARQUET_MAGIC:
        # Git LFS pointer starts with b"version https://git-lfs"
        if data[:7] == b"version":
            return False, "response is a Git LFS pointer, not the actual file"
        # HTML page
        if data[:15].lower().lstrip().startswith(b"<!doctype") or b"<html" in data[:200].lower():
            return False, "response is an HTML page"
        return False, f"missing parquet magic (got {data[:4]!r})"
    return True, "ok"


def _download_parquet(url: str) -> bytes:
    """Download url and return raw bytes. Raises on HTTP error."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
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


# ── Fallback: reconstruct from pi1_example.json ───────────────────────────────

def _build_row(problem: str, answer: str, split: str) -> dict:
    return {
        "data_source": "lighteval/MATH",
        "prompt": [{"role": "user", "content": problem}],
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"split": split},
    }


def _fallback_generate(pi1_path: str, n_train: int, n_val: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(pi1_path) as f:
        pi1 = json.load(f)
    problem = pi1["problem"]
    answer = pi1["answer"]
    train_df = pd.DataFrame([_build_row(problem, answer, "train") for _ in range(n_train)])
    val_df = pd.DataFrame([_build_row(problem, answer, "val") for _ in range(n_val)])
    return train_df, val_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pi1_path",
        default=os.path.join(os.path.dirname(__file__), "pi1_example.json"),
        help="Fallback: path to pi1_example.json",
    )
    parser.add_argument("--n_train", type=int, default=_EXPECTED_ROWS_TRAIN)
    parser.add_argument("--n_val", type=int, default=_EXPECTED_ROWS_VAL)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")

    # ── Primary: download repo-shipped pi1_r128.parquet ───────────────────────
    print(f"Downloading pi1_r128.parquet from pinned commit {_COMMIT[:12]}…")
    print(f"  URL: {_TRAIN_URL}")

    df, reason = _try_download(_TRAIN_URL)

    if df is not None:
        # Validate row count
        if len(df) != args.n_train:
            print(f"  WARNING: expected {args.n_train} rows, got {len(df)} — falling back")
            df = None
        else:
            print(f"  Downloaded OK: {len(df)} rows, columns={df.columns.tolist()}")

    if df is not None:
        train_df = df
        val_df = df.iloc[: args.n_val].copy().reset_index(drop=True)
        print(f"  train.parquet: {len(train_df)} rows (repo-shipped)")
        print(f"  val.parquet  : {len(val_df)} rows (first {args.n_val} rows of same file)")
    else:
        print(f"  Download/validation failed: {reason}")
        print(f"  Falling back to reconstruction from {args.pi1_path} …")
        train_df, val_df = _fallback_generate(args.pi1_path, args.n_train, args.n_val)
        print(f"  train.parquet: {len(train_df)} rows (reconstructed from pi1_example.json)")
        print(f"  val.parquet  : {len(val_df)} rows (reconstructed from pi1_example.json)")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    # ── Final assertions ───────────────────────────────────────────────────────
    assert len(train_df) == args.n_train, f"Expected {args.n_train} train rows, got {len(train_df)}"
    assert len(val_df) == args.n_val, f"Expected {args.n_val} val rows, got {len(val_df)}"
    assert _REQUIRED_COLS.issubset(set(train_df.columns)), (
        f"Missing columns: {_REQUIRED_COLS - set(train_df.columns)}"
    )

    print(f"\nColumns : {train_df.columns.tolist()}")
    print(f"train.parquet: {len(train_df)} rows  →  {train_path}")
    print(f"val.parquet  : {len(val_df)} rows  →  {val_path}")
    print(f"Example row  :\n{train_df.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
