from zs_game import ZSGame
from deceiver_binary_search import deception_binary_search
from deceiver_nonconvex import deception_nonconvex

import numpy as np
import matplotlib.pyplot as plt

rps = ZSGame(
    np.array(
        [
            [0, 1, -1],
            [-1, 0, 1],
            [1, -1, 0],
        ]
    )
)

optimal_improvements = []
feasible_improvements = []

for deception_budget in np.linspace(0.1, 3, 10):
    print("Deception budget:", deception_budget)
    print("============================================================")
    print("Computing optimal deception...")
    dr_optimal = deception_nonconvex(rps, deception_budget)
    optimal_improvement = dr_optimal.honest_value - dr_optimal.obtained_value
    optimal_improvements.append(optimal_improvement)
    print(f"Optimal deception:\n{dr_optimal.deception}")
    print(f"Optimal improvement: {optimal_improvement}")

    print("\nComputing feasible deception...")
    dr_feasible = deception_binary_search(rps, deception_budget)
    feasible_improvement = dr_feasible.honest_value - dr_feasible.obtained_value
    feasible_improvements.append(feasible_improvement)
    print(f"Feasible deception:\n{dr_feasible.deception}")
    print(f"Feasible improvement: {feasible_improvement}")
    print()

optimal_improvements = np.array(optimal_improvements)
feasible_improvements = np.array(feasible_improvements)

plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})
plt.figure()
plt.plot(optimal_improvements, label="Optimal Deception")
plt.plot(feasible_improvements, label="Feasible Deception")
plt.xlabel("Deception Budget")
plt.ylabel("Improvement")
plt.title("Deception Improvement vs. Budget for RPS")
plt.legend()
plt.show()
