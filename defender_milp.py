from zs_game import ZSGame

import numpy as np
import numpy.typing as npt
from gurobipy import Model, GRB 

from typing import Tuple

FArray = npt.NDArray[np.float64]

np.random.seed(5)

num_rows = 3
num_cols = 3
deception_budget = 0.4
game = ZSGame.random(num_rows, num_cols)
game = ZSGame(np.array([[0.0, 1, -1], [-1, 0, 1], [1, -1, 0]]))
dgame = ZSGame.random_disturbance(game, deception_budget)

chosen_deception = dgame.outcomes - game.outcomes
assert np.linalg.norm(chosen_deception, ord=1) <= deception_budget


def robust_defender_strategy(
    dgame: ZSGame, deception_budget: float
) -> Tuple[float, FArray, FArray]:
    """
    Computes the robust defender strategy for a given game with a deception budget.
    Args:
        dgame: The game with disturbances.
        deception_budget: The budget for deception.
    Returns:
        A tuple of (value, attacker_strategy, defender_strategy).
    """
    num_rows, num_cols = dgame.outcomes.shape

    # Create Gurobi model
    m = Model()
    m.setParam('OutputFlag', 0)  # silent

    # Variables: x (num_rows), alpha, beta
    x = m.addVars(num_rows, lb=0, name="x")
    alpha = m.addVar(lb=-GRB.INFINITY, name="alpha")
    beta = m.addVar(lb=0, name="beta")

    # Objective: minimize alpha + deception_budget * beta
    m.setObjective(alpha + deception_budget * beta, GRB.MINIMIZE)

    # Constraints
    # sum(x) == 1
    m.addConstr(sum(x[i] for i in range(num_rows)) == 1, "prob_sum")

    # For each column j: sum_i x[i] * dgame.outcomes[i, j] <= alpha
    for j in range(num_cols):
        m.addConstr(
            sum(x[i] * dgame.outcomes[i, j] for i in range(num_rows)) <= alpha,
            f"alpha_col_{j}"
        )

    # For each row i: x[i] <= beta
    for i in range(num_rows):
        m.addConstr(x[i] <= beta, f"beta_row_{i}")

    m.optimize()

    # Extract solution
    x_sol = np.array([x[i].X for i in range(num_rows)], dtype=float)
    alpha_sol = alpha.X
    beta_sol = beta.X

    # For dual variables (defender strategy), use shadow prices of alpha constraints
    robust_y = np.array([-m.getConstrByName(f"alpha_col_{j}").Pi for j in range(num_cols)])

    return m.ObjVal, np.hstack((x_sol, [alpha_sol, beta_sol])), robust_y


# Compare to non-robust behavior

robust_val, robust_x, robust_y = robust_defender_strategy(dgame, deception_budget)
alpha, beta = robust_x[-2:]
robust_x = robust_x[:-2]  # remove alpha, beta from x


# ============================================================
# Is there a canonical worst deception?
# ============================================================

outer = np.outer(robust_x, robust_y)
dnorm = np.linalg.norm(outer, ord=np.inf)
worst_row = np.argmax(
    np.sum(outer, axis=1)
)  # don't need abs here, as x and y are non-negative
worst_D = np.zeros_like(
    game.outcomes
)  
worst_D[worst_row, :] = (
    deception_budget  # don't need sign check here, as x and y are non-negative
)
worst_game = dgame.from_disturbance(-worst_D)

gp_comps = robust_x @ dgame.outcomes
max_gp_comp = np.max(gp_comps)
valid_strats = np.isclose(gp_comps, max_gp_comp, atol=1e-9).astype(float)
unif_y = valid_strats / np.sum(valid_strats)
single_y = np.zeros(num_cols)
single_y[np.argmax(gp_comps)] = 1

worst_val, worst_x, worst_y = worst_game.solve()
# print(f"{deceptive_val=}\n{worst_val=}")
# print(f"{worst_x=}\n{x=}")
# print(f"{worst_y=}\n{y=}")

outcome_val = game.outcome(robust_x, robust_y)
full_knowledge_val = game.solve()[0]


# print(f"{worst_game.outcomes=}")
# print(f"{dgame.outcomes=}")
# print(f"{worst_val=}")
# print(f"{robust_val=}")
# print(f"{full_knowledge_val=}")

# assert robust_val <= full_knowledge_val

# print(np.linalg.norm(dgame.outcomes - game.outcomes, ord=1))

# print(f"{deceptive_val=}\n{outcome_val=}\n{full_knowledge_val=}")


# ============================================================
# Display results
# ============================================================

print(f"The defender viewed the (disturbed) game\n{dgame.outcomes}")
print(
    f"The calculated worst-case deception is\n{worst_D},\nwhich implicitly induces the believed game\n{worst_game.outcomes}\n"
)

print(
    f"The defender chose strategy\n{robust_y} (dual)\n{unif_y} (uniform)\n{single_y} (single)"
)
print(f"The attacker chose strategy\n{robust_x}\n")

print(f"The LP solver predicts a value of {robust_val:.3f} for the defender.")
print(
    f"The outcome of the believed game is\n{worst_game.outcome(robust_x, robust_y):.3f} (dual)\n{worst_game.outcome(robust_x, unif_y):.3f} (uniform)\n{worst_game.outcome(robust_x, single_y):.3f} (single)"
)

print(f"{alpha=}, {beta=} => {alpha + deception_budget * beta=}")
print(f"Manually calculated: {robust_x @ dgame.outcomes=}")
print(f"Manually calculated: {dgame.outcome(robust_x, robust_y)=}")
print(f"Manually calculated: {robust_x @ worst_D @ unif_y=}")
