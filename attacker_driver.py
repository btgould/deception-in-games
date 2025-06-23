from zs_game import ZSGame
from attacker_binary_search import attacker_binary_search
from attacker_mccormick_env import attacker_mccormick_env

import numpy as np
import matplotlib.pyplot as plt
import pickle


NUM_GAMES = 100
NUM_ROWS = 5
NUM_COLS = 5

rng = np.random.default_rng(42)
games = rng.uniform(low=-5, high=5, size=(NUM_GAMES, NUM_ROWS, NUM_COLS))

deception_budgets = range(1, 4)
all_improvements = []

for deception_budget in deception_budgets:
    improvements = np.empty(NUM_GAMES, dtype=float)
    for i, game in enumerate(games):
        dv = 0.01
        # dr = attacker_mccormick_env(ZSGame(game), deception_budget, dv)
        dr = attacker_binary_search(ZSGame(game), deception_budget, dv)
        improvements[i] = dr.honest_value - dr.obtained_value
        assert dr.suboptimality_gap <= dv or np.allclose(dr.suboptimality_gap, dv), (
            f"Suboptimality gap is too large. Expected <={dv}, got {dr.suboptimality_gap}"
        )
    assert np.all(improvements >= 0), (
        "Attacker became worse off by using deception in some games"
    )
    all_improvements.append(improvements)

means = [imps.mean() for imps in all_improvements]
stds = [imps.std() for imps in all_improvements]

with open("all_improvements.pkl", "wb") as f:
    pickle.dump(all_improvements, f)

plt.figure()
plt.plot(deception_budgets, means, marker="o", label="Mean improvement")
plt.fill_between(
    deception_budgets,
    [m - s for m, s in zip(means, stds)],
    [m + s for m, s in zip(means, stds)],
    color="b",
    alpha=0.2,
    label="Std deviation",
)
plt.xlabel("Deception budget")
plt.ylabel("Improvement")
plt.title("Improvement vs Deception Budget")
plt.legend()

all_improvements = np.array(all_improvements)

plt.figure()
plt.plot(deception_budgets, all_improvements, marker="o")
plt.xlabel("Deception budget")
plt.ylabel("Improvement")
plt.title("Improvement vs Deception Budget (per game)")
plt.legend()

plt.show()
