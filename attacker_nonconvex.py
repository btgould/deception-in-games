from zs_game import ZSGame

import numpy as np
import numpy.typing as npt
import pandas as pd

import gurobipy as gp
from gurobipy import Model, GRB

import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict

FArray = npt.NDArray[np.float64]


@dataclass
class DeceptionResult:
    game: ZSGame
    deceiver_strategy: FArray
    deception: FArray
    victim_strategy: FArray
    honest_value: float
    obtained_value: float


def attacker_nonconvex(game: ZSGame, deception_budget: float) -> DeceptionResult:
    """
    Computes the optimal deception strategy for the attacker by solving a non-convex program.
    Args:
        game: The zero-sum game to analyze.
        deception_budget: The budget for deception.
    Returns:
        A DeceptionResult containing the best deception strategy and victim's response.
    """

    num_rows, num_cols = game.outcomes.shape

    x_sol = np.empty(num_rows, dtype=float)
    D_sol = np.empty_like(game.outcomes)
    y_sol = np.empty(num_cols, dtype=float)
    best_dr = DeceptionResult(
        game=game,
        deceiver_strategy=x_sol,
        deception=D_sol,
        victim_strategy=y_sol,
        honest_value=np.inf,
        obtained_value=np.inf,
    )

    m = Model()
    m.setParam("OutputFlag", 0)  # silent
    m.setParam("MIPGap", 0.01)
    m.setParam("TimeLimit", 60)
    m.setParam("NonConvex", 2)  # Enable non-convex QCP

    # Decision variables
    D = m.addVars(
        num_rows, num_cols, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="deception"
    )
    D_abs = m.addVars(num_rows, num_cols, lb=0, name="deception_norm_aux")
    x = m.addVars(num_rows, lb=0, ub=1, name="deceiver_strat")
    y = m.addVars(num_cols, lb=0, ub=1, name="victim_strat")
    v = m.addVar(lb=-GRB.INFINITY, name="primal_val")
    gamma = m.addVars(num_rows, lb=0, ub=1, name="dual_strat")
    u = m.addVar(lb=-GRB.INFINITY, name="dual_val")

    # Deception budget constraints
    m.addConstrs(
        D_abs[i, j] >= D[i, j] for i in range(num_rows) for j in range(num_cols)
    )
    m.addConstrs(
        D_abs[i, j] >= -D[i, j] for i in range(num_rows) for j in range(num_cols)
    )
    m.addConstrs(
        (
            gp.quicksum(D_abs[i, j] for i in range(num_rows)) <= deception_budget
            for j in range(num_cols)
        ),
        name="col1norm",
    )

    # Simplex constraints
    m.addConstr(x.sum() == 1, name="simplex_x")
    m.addConstr(y.sum() == 1, name="simplex_y")
    m.addConstr(gamma.sum() == 1, name="simplex_gamma")

    # Bilinear products via QConstr for McCormick envelopes
    w1 = m.addVars(num_rows, num_cols, name="w1")
    w2 = m.addVars(num_rows, num_cols, name="w2")
    m.addConstrs(
        w1[i, j] == D[i, j] * y[j] for i in range(num_rows) for j in range(num_cols)
    )
    m.addConstrs(
        w2[i, j] == gamma[i] * D[i, j] for i in range(num_rows) for j in range(num_cols)
    )

    # Primal feasibility
    m.addConstrs(
        (
            gp.quicksum(game.outcomes[i][j] * y[j] for j in range(num_cols))
            + gp.quicksum(w1[i, j] for j in range(num_cols))
            >= v
            for i in range(num_rows)
        ),
        name="primal_row",
    )

    # Dual feasibility
    m.addConstrs(
        (
            gp.quicksum(game.outcomes[i][j] * gamma[i] for i in range(num_rows))
            + gp.quicksum(w2[i, j] for i in range(num_rows))
            <= u
            for j in range(num_cols)
        ),
        name="dual_col",
    )

    # Strong duality (could technically be removed, but is handled in presolve anyways)
    m.addConstr(v == u, name="strong_duality")

    # Objective
    m.setObjective(
        gp.quicksum(
            gp.quicksum(x[i] * game.outcomes[i][j] * y[j] for j in range(num_cols))
            for i in range(num_rows)
        ),
        GRB.MINIMIZE,
    )

    m.optimize()

    x_sol = np.array([x[i].X for i in range(num_rows)], dtype=float)
    D_sol = np.array([[D[i, j].X for j in range(num_cols)] for i in range(num_rows)])
    y_sol = np.array([y[j].X for j in range(num_cols)])

    honest_value = game.solve()[0]
    achieved_value = m.ObjVal

    best_dr = DeceptionResult(
        game=game,
        deceiver_strategy=x_sol,
        deception=D_sol,
        victim_strategy=y_sol,
        honest_value=honest_value,
        obtained_value=achieved_value,
    )

    return best_dr


def _is_victim_rational(dr: DeceptionResult) -> bool:
    dgame = dr.game.from_disturbance(dr.deception)
    rational_sv = dgame.solve()[0]
    computed_sv = np.min(dgame.outcomes @ dr.victim_strategy)

    return np.allclose(rational_sv, computed_sv, atol=1e-4)


def _does_deceiver_benefit(dr: DeceptionResult):
    return dr.honest_value > dr.obtained_value or np.allclose(
        dr.honest_value, dr.obtained_value
    )


def verify_deception_result(dr: DeceptionResult) -> bool:
    """
    Performs various sanity checks on the computed deception.

    Args:
        dr: The DeceptionResult to verify.
    Returns:
        True if the deception result is valid, False otherwise.
    """
    verified: bool = True
    verified &= _is_victim_rational(dr)
    verified &= _does_deceiver_benefit(dr)
    return verified


if __name__ == "__main__":
    NUM_GAMES = 5
    NUM_ROWS = 5
    NUM_COLS = 5
    SANITY_CHECK = False

    rng = np.random.default_rng(42)
    games = rng.uniform(low=-5, high=5, size=(NUM_GAMES, NUM_ROWS, NUM_COLS))

    deception_budgets = [1, 2, 3]
    results = []

    for deception_budget in deception_budgets:
        for i, game in enumerate(games):
            dr = attacker_nonconvex(ZSGame(game), deception_budget)
            result = asdict(dr)
            result["deception_budget"] = deception_budget
            result["game_idx"] = i
            results.append(result)

            if SANITY_CHECK:
                assert verify_deception_result(dr), f"Sanity check failed for:\n{dr}"

    df = pd.DataFrame(results)
    improvements = df["honest_value"] - df["obtained_value"]
    improvements_by_budget = [
        improvements[df["deception_budget"] == db] for db in deception_budgets
    ]
    means = [imp.mean() for imp in improvements_by_budget]
    stds = [imp.std() for imp in improvements_by_budget]

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
    plt.show()
