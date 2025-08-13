from zs_game import ZSGame
from deception import verify_deception_result

from deceiver_nonconvex import deception_nonconvex
from deceiver_binary_search import deception_binary_search

import numpy as np
from dataclasses import asdict
import time
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})


NUM_GAMES = 5
NUM_ROWS = 5
NUM_COLS = 5
SANITY_CHECK = False


def sample_deceptions(
    num_games: int,
    num_rows: int,
    num_cols: int,
    deception_budgets: Iterable[float],
    method: Callable,
    rng=np.random.default_rng(42),
):
    games = rng.uniform(low=-5, high=5, size=(num_games, num_rows, num_cols))
    results = []

    for deception_budget in deception_budgets:
        for i, game in enumerate(games):
            t0 = time.time()
            dr = method(ZSGame(game), deception_budget)
            tf = time.time()
            result = asdict(dr)
            result["deception_budget"] = deception_budget
            result["game_idx"] = i
            result["computation_time"] = tf - t0
            results.append(result)

            if SANITY_CHECK:
                assert verify_deception_result(dr), f"Sanity check failed for:\n{dr}"

    return pd.DataFrame(results)
    # df.to_csv("deception_results.csv", index=False)


def compare_deceptive_strategies():
    deception_budgets = np.linspace(1, 6, 5)
    df_bin_search = sample_deceptions(
        NUM_GAMES, NUM_ROWS, NUM_COLS, deception_budgets, deception_binary_search
    )
    df_nonconvex = sample_deceptions(
        NUM_GAMES, NUM_ROWS, NUM_COLS, deception_budgets, deception_nonconvex
    )

    # Calculate summary statistics for binary search
    improvements_bin = df_bin_search["honest_value"] - df_bin_search["obtained_value"]
    improvement_mean_by_budget_bin = improvements_bin.groupby(
        df_bin_search["deception_budget"]
    ).mean()
    improvement_std_by_budget_bin = improvements_bin.groupby(
        df_bin_search["deception_budget"]
    ).std()

    # Calculate summary statistics for nonconvex
    improvements_nonconvex = (
        df_nonconvex["honest_value"] - df_nonconvex["obtained_value"]
    )
    improvement_mean_by_budget_nonconvex = improvements_nonconvex.groupby(
        df_nonconvex["deception_budget"]
    ).mean()
    improvement_std_by_budget_nonconvex = improvements_nonconvex.groupby(
        df_nonconvex["deception_budget"]
    ).std()

    # Report statistics
    print("=" * 60)
    print("Comparing deception methods over a range of budgets...")
    print("=" * 60)
    print("Binary Search - Computation time statistics:")
    print(df_bin_search["computation_time"].describe())
    print("\nNonconvex - Computation time statistics:")
    print(df_nonconvex["computation_time"].describe())

    # Plotting
    plt.figure()
    plt.plot(
        deception_budgets,
        improvement_mean_by_budget_bin,
        marker="o",
        label="Binary Search",
        color="blue",
    )
    plt.fill_between(
        deception_budgets,
        [
            m - s
            for m, s in zip(
                improvement_mean_by_budget_bin, improvement_std_by_budget_bin
            )
        ],
        [
            m + s
            for m, s in zip(
                improvement_mean_by_budget_bin, improvement_std_by_budget_bin
            )
        ],
        color="blue",
        alpha=0.2,
    )

    plt.plot(
        deception_budgets,
        improvement_mean_by_budget_nonconvex,
        marker="s",
        label="Nonconvex",
        color="red",
    )
    plt.fill_between(
        deception_budgets,
        [
            m - s
            for m, s in zip(
                improvement_mean_by_budget_nonconvex,
                improvement_std_by_budget_nonconvex,
            )
        ],
        [
            m + s
            for m, s in zip(
                improvement_mean_by_budget_nonconvex,
                improvement_std_by_budget_nonconvex,
            )
        ],
        color="red",
        alpha=0.2,
    )

    plt.xlabel("Deception budget")
    plt.ylabel("Improvement")
    plt.title("Improvement vs Deception Budget")
    plt.legend()
    plt.show()


compare_deceptive_strategies()
