"""Solves the deception problem over one batch of games and writes the raw results."""

import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from methods import DV, DeceptionMethod, solver_for, verify_deception_result
from sampling import load_games
from zs_game import ZSGame

FArray = npt.NDArray[np.float64]

COLUMNS = [
    "label",
    "method",
    "method_label",
    "game_idx",
    "num_rows",
    "num_cols",
    "size",
    "deception_budget",
    "dv",
    "time_limit",
    "honest_value",
    "obtained_value",
    "improvement",
    "computation_time",
    "truncated",
]

ARRAY_KEYS = ["games", "deception", "deceiver_strategy", "victim_strategy"]


def run_batch(
    label: str,
    games: FArray,
    method: DeceptionMethod,
    deception_budgets: Iterable[float],
    dv: float,
    time_limit: float,
    sanity_check: bool = False,
    **solver_options,
) -> tuple[pd.DataFrame, dict[str, FArray]]:
    """
    Solves the deception problem for every (budget, game) pair in a batch.

    Args:
        label: Name associated with the results in the `label` column; does not affect
            the computation.
        games: Payoff matrices of shape (num_games, num_rows, num_cols).
        method: Which deception solver to apply.
        deception_budgets: Budgets to sweep, outer loop.
        dv: Binary search tolerance (ignored by the nonconvex solver).
        time_limit: Gurobi time limit in seconds (ignored by the binary search solver).
        sanity_check: Assert verify_deception_result on every result.
        solver_options: Extra keyword arguments forwarded to solver_for.

    Returns:
        A tidy DataFrame with the COLUMNS schema, and a dict of stacked arrays whose
        rows align positionally with the DataFrame rows.
    """
    solve = solver_for(method, dv=dv, time_limit=time_limit, **solver_options)
    _, num_rows, num_cols = games.shape

    rows = []
    arrays: dict[str, list[FArray]] = {key: [] for key in ARRAY_KEYS}

    for deception_budget in deception_budgets:
        for game_idx, outcomes in enumerate(games):
            game = ZSGame(outcomes)

            t0 = time.perf_counter()
            dr = solve(game, deception_budget)
            computation_time = time.perf_counter() - t0

            if sanity_check:
                assert verify_deception_result(dr), f"Sanity check failed for:\n{dr}"

            rows.append(
                {
                    "label": label,
                    "method": str(method),
                    "method_label": method.label,
                    "game_idx": game_idx,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "size": num_rows if num_rows == num_cols else np.nan,
                    "deception_budget": deception_budget,
                    "dv": dv,
                    "time_limit": time_limit,
                    "honest_value": dr.honest_value,
                    "obtained_value": dr.obtained_value,
                    "improvement": dr.honest_value - dr.obtained_value,
                    "computation_time": computation_time,
                    "truncated": dr.truncated,
                }
            )

            arrays["games"].append(outcomes)
            arrays["deception"].append(dr.deception)
            arrays["deceiver_strategy"].append(dr.deceiver_strategy)
            arrays["victim_strategy"].append(dr.victim_strategy)

    df = pd.DataFrame(rows, columns=COLUMNS)
    stacked = {key: np.stack(values) for key, values in arrays.items()}
    return df, stacked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--method", required=True, type=DeceptionMethod)
    parser.add_argument("--games", required=True)
    parser.add_argument("--budgets", required=True, type=float, nargs="+")
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--dv", type=float, default=DV)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--mip_gap", type=float, default=0.01)
    parser.add_argument("--gurobi_seed", type=int, default=0)
    parser.add_argument("--gurobi_threads", type=int, default=0)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_npz", required=True)
    args = parser.parse_args()

    df, arrays = run_batch(
        label=args.label,
        games=load_games(args.games),
        method=args.method,
        deception_budgets=args.budgets,
        dv=args.dv,
        time_limit=args.time_limit,
        sanity_check=args.sanity_check,
        mip_gap=args.mip_gap,
        gurobi_seed=args.gurobi_seed,
        gurobi_threads=args.gurobi_threads,
    )

    for path in (args.out_csv, args.out_npz):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    np.savez_compressed(args.out_npz, **arrays)


if __name__ == "__main__":
    main()
