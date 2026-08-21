import time
from collections.abc import Callable, Iterable
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deceiver_binary_search import deception_binary_search
from deceiver_nonconvex import deception_nonconvex
from deception import verify_deception_result
from zs_game import ZSGame

plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})


NUM_GAMES = 5
NUM_ROWS = 5
NUM_COLS = 5
SANITY_CHECK = False

DEFAULT_RNG = np.random.default_rng(42)


def sample_deceptions(
    num_games: int,
    num_rows: int,
    num_cols: int,
    deception_budgets: Iterable[float],
    method: Callable,
    games=None,
    rng=DEFAULT_RNG,
):
    if games is None:
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
    rng = np.random.default_rng(42)
    games = rng.uniform(low=-5, high=5, size=(NUM_GAMES, NUM_ROWS, NUM_COLS))

    df_nonconvex = sample_deceptions(
        NUM_GAMES,
        NUM_ROWS,
        NUM_COLS,
        deception_budgets,
        deception_nonconvex,
        games=games,
    )
    df_bin_search = sample_deceptions(
        NUM_GAMES,
        NUM_ROWS,
        NUM_COLS,
        deception_budgets,
        deception_binary_search,
        games=games,
    )

    # New method: binary search with robustify=False
    def deception_binary_search_no_robust(game, budget):
        return deception_binary_search(game, budget, robustify=False)

    df_bin_search_no_robust = sample_deceptions(
        NUM_GAMES,
        NUM_ROWS,
        NUM_COLS,
        deception_budgets,
        deception_binary_search_no_robust,
        games=games,
    )

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

    # Calculate summary statistics for binary search
    improvements_bin = df_bin_search["honest_value"] - df_bin_search["obtained_value"]
    improvement_mean_by_budget_bin = improvements_bin.groupby(
        df_bin_search["deception_budget"]
    ).mean()
    improvement_std_by_budget_bin = improvements_bin.groupby(
        df_bin_search["deception_budget"]
    ).std()

    # Calculate summary statistics for binary search (robustify=False)
    improvements_bin_no_robust = (
        df_bin_search_no_robust["honest_value"]
        - df_bin_search_no_robust["obtained_value"]
    )
    improvement_mean_by_budget_bin_no_robust = improvements_bin_no_robust.groupby(
        df_bin_search_no_robust["deception_budget"]
    ).mean()
    improvement_std_by_budget_bin_no_robust = improvements_bin_no_robust.groupby(
        df_bin_search_no_robust["deception_budget"]
    ).std()

    # Report statistics
    print("=" * 60)
    print("Comparing deception methods over a range of budgets...")
    print("=" * 60)
    print("\nNonconvex - Computation time statistics:")
    print(df_nonconvex["computation_time"].describe())
    print("Binary Search - Computation time statistics:")
    print(df_bin_search["computation_time"].describe())
    print("\nBinary Search (robustify=False) - Computation time statistics:")
    print(df_bin_search_no_robust["computation_time"].describe())

    # Plotting improvements
    plt.figure(figsize=(8, 6))

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

    plt.plot(
        deception_budgets,
        improvement_mean_by_budget_bin,
        marker="o",
        label="Binary Search (Robust)",
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
        improvement_mean_by_budget_bin_no_robust,
        marker="^",
        label="Binary Search (Non-Robust)",
        color="green",
    )
    plt.fill_between(
        deception_budgets,
        [
            m - s
            for m, s in zip(
                improvement_mean_by_budget_bin_no_robust,
                improvement_std_by_budget_bin_no_robust,
            )
        ],
        [
            m + s
            for m, s in zip(
                improvement_mean_by_budget_bin_no_robust,
                improvement_std_by_budget_bin_no_robust,
            )
        ],
        color="green",
        alpha=0.2,
    )

    plt.xlabel("Deception budget")
    plt.ylabel("Improvement")
    plt.title("Improvement vs Deception Budget")
    plt.legend()
    plt.show()


def computation_time_by_size():
    sizes = list(range(2, 16))  # 2x2 through 15x15
    num_games = 10  # Number of games per size for averaging
    deception_budget = 3.0  # Use a fixed budget for all
    rng = np.random.default_rng(123)

    times_binsearch = []
    times_nonconvex = []

    for n in sizes:
        print(f"Running games of size {n}x{n}...")
        games = rng.uniform(low=-5, high=5, size=(num_games, n, n))
        # Binsearch (robustified)
        t_bin = []
        for game in games:
            t0 = time.time()
            deception_binary_search(ZSGame(game), deception_budget)
            t1 = time.time()
            t_bin.append(t1 - t0)
        times_binsearch.append(np.mean(t_bin))

        # Nonconvex
        if n < 8:  # Too slow for large games, skip for demonstration purposes
            t_non = []
            for game in games:
                t0 = time.time()
                deception_nonconvex(
                    ZSGame(game), deception_budget, time_limit=300
                )  # cutoff after 5 min
                t1 = time.time()
                t_non.append(t1 - t0)
            times_nonconvex.append(np.mean(t_non))
        else:
            times_nonconvex.append(np.nan)

    plt.figure(figsize=(8, 6))
    plt.plot(
        sizes,
        times_binsearch,
        marker="o",
        label="Binary Search (Robust)",
        color="blue",
    )
    plt.plot(sizes, times_nonconvex, marker="s", label="Nonconvex", color="red")
    plt.yscale("log")
    plt.xlabel("Game size ($n \\times n$)")
    plt.ylabel("Mean Computation Time (s, log scale)")
    plt.title("Computation Time vs Game Size")
    plt.legend()
    plt.tight_layout()
    plt.show()


def binsearch_tol_comparison():
    num_games = 100
    num_rows = 5
    num_cols = 5
    deception_budget = 3.0
    rng = np.random.default_rng(42)

    # Logarithmically spaced tolerances
    tolerances = np.logspace(-1, -6, 10)
    improvements_robust = []
    improvements_nonrobust = []

    # Sample games once for all tolerances
    games = rng.uniform(low=-5, high=5, size=(num_games, num_rows, num_cols))

    for tol in tolerances:
        # Robustified
        imp_robust = []
        imp_nonrobust = []
        for game in games:
            zs = ZSGame(game)
            dr_robust = deception_binary_search(
                zs, deception_budget, dv=tol, robustify=True
            )
            dr_nonrobust = deception_binary_search(
                zs, deception_budget, dv=tol, robustify=False
            )
            imp_robust.append(dr_robust.honest_value - dr_robust.obtained_value)
            imp_nonrobust.append(
                dr_nonrobust.honest_value - dr_nonrobust.obtained_value
            )
        improvements_robust.append(np.mean(imp_robust))
        improvements_nonrobust.append(np.mean(imp_nonrobust))

    plt.figure(figsize=(8, 6))
    plt.plot(
        tolerances, improvements_robust, marker="o", label="Robustified", color="blue"
    )
    plt.plot(
        tolerances,
        improvements_nonrobust,
        marker="s",
        label="Non-robustified",
        color="red",
    )
    plt.xscale("log")
    plt.xlabel("Binary Search Tolerance")
    plt.ylabel("Mean Improvement")
    plt.title("Improvement vs Binary Search Tolerance")
    plt.legend()
    plt.tight_layout()
    plt.show()


compare_deceptive_strategies()
computation_time_by_size()
binsearch_tol_comparison()
