from zs_game import ZSGame
from deception import verify_deception_result

# from deceiver_nonconvex import deception_nonconvex as compute_deception
from deceiver_binary_search import deception_binary_search as compute_deception

import numpy as np
from dataclasses import asdict
import time

import matplotlib.pyplot as plt
import pandas as pd


NUM_GAMES = 5
NUM_ROWS = 5
NUM_COLS = 5
SANITY_CHECK = False

rng = np.random.default_rng(42)
games = rng.uniform(low=-5, high=5, size=(NUM_GAMES, NUM_ROWS, NUM_COLS))

deception_budgets = range(1, 6)
results = []

for deception_budget in deception_budgets:
    for i, game in enumerate(games):
        t0 = time.time()
        dr = compute_deception(ZSGame(game), deception_budget)
        tf = time.time()
        result = asdict(dr)
        result["deception_budget"] = deception_budget
        result["game_idx"] = i
        result["computation_time"] = tf - t0
        results.append(result)

        if SANITY_CHECK:
            assert verify_deception_result(dr), f"Sanity check failed for:\n{dr}"

df = pd.DataFrame(results)
df.to_csv("deception_results.csv", index=False)

# Plotting
improvements = df["honest_value"] - df["obtained_value"]
improvements_by_budget = [
    improvements[df["deception_budget"] == db] for db in deception_budgets
]
improvement_mean_by_budget = improvements.groupby(df["deception_budget"]).mean()
improvement_std_by_budget = improvements.groupby(df["deception_budget"]).std()
improvement_mean_by_game = improvements.groupby(df["game_idx"]).mean()
improvement_std_by_game = improvements.groupby(df["game_idx"]).std()

plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})
plt.figure()
plt.plot(
    deception_budgets, improvement_mean_by_budget, marker="o", label="Mean improvement"
)
plt.fill_between(
    deception_budgets,
    [m - s for m, s in zip(improvement_mean_by_budget, improvement_std_by_budget)],
    [m + s for m, s in zip(improvement_mean_by_budget, improvement_std_by_budget)],
    color="b",
    alpha=0.2,
    label="Std deviation",
)
plt.xlabel("Deception budget")
plt.ylabel("Improvement")
plt.title("Improvement vs Deception Budget")
plt.legend()
plt.show()

# Report statistics
print("Computation time statistics:")
print(df["computation_time"].describe())
print("Computation time by game index:")
print(df.groupby("game_idx")["computation_time"].describe())
print("Computation time by deception budget:")
print(df.groupby("deception_budget")["computation_time"].describe())
