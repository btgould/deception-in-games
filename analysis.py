"""Concatenates one experiment's per-sweep-point tables into its canonical result."""

import argparse
import json
from pathlib import Path

import pandas as pd

SORT_KEYS = ["method", "size", "dv", "deception_budget", "game_idx"]


def collect(inputs: str) -> pd.DataFrame:
    """Reads every per-sweep-point table under `inputs` into one sorted frame."""
    parts = sorted(Path(inputs).glob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"No result tables found under {inputs}")

    df = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True, sort=False)
    return df.sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)


def _truncated_by_size(group: pd.DataFrame) -> dict[str, int]:
    """Counts time-limit-truncated solves per square game size, omitting clean sizes."""
    sized = group.dropna(subset=["size"])
    counts = sized.groupby("size")["truncated"].sum()
    return {str(int(size)): int(n) for size, n in counts.items() if n}


def summarize(df: pd.DataFrame) -> dict:
    """Per-method timing and improvement summary, reported as DVC metrics."""
    return {
        "num_results": len(df),
        "num_truncated": int(df["truncated"].sum()),
        "methods": {
            str(method): {
                "num_results": len(group),
                "computation_time_mean": float(group["computation_time"].mean()),
                "computation_time_std": float(group["computation_time"].std()),
                "computation_time_max": float(group["computation_time"].max()),
                "improvement_mean": float(group["improvement"].mean()),
                "improvement_std": float(group["improvement"].std()),
                "num_truncated": int(group["truncated"].sum()),
                "truncated_by_size": _truncated_by_size(group),
            }
            for method, group in df.groupby("method")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_metrics", required=True)
    args = parser.parse_args()

    df = collect(args.inputs)

    for path in (args.out_csv, args.out_metrics):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    with open(args.out_metrics, "w") as f:
        json.dump(summarize(df), f, indent=2)


if __name__ == "__main__":
    main()
