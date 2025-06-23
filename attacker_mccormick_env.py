from zs_game import ZSGame

import numpy as np
import numpy.typing as npt
import gurobipy as gp
from gurobipy import Model, GRB

import matplotlib.pyplot as plt

from dataclasses import dataclass

FArray = npt.NDArray[np.float64]


@dataclass
class DeceptionResult:
    game: ZSGame
    chosen_deception: FArray
    defender_strategy: FArray
    honest_value: float
    suboptimality_gap: float
    obtained_value: float


def attacker_mccormick_env(
    game: ZSGame, deception_budget: float, dv: float = 0.01
) -> DeceptionResult:
    """
    Computes the optimal deception strategy for the attacker using binary search.
    Args:
        game: The zero-sum game to analyze.
        deception_budget: The budget for deception.
    Returns:
        A DeceptionResult containing the best deception strategy and defender's response.
    """

    num_rows, num_cols = game.outcomes.shape

    min_val = np.min(game.outcomes) - deception_budget
    max_val = np.max(game.outcomes) + deception_budget

    D_sol = np.empty_like(game.outcomes)
    y_sol = np.empty(num_cols, dtype=float)
    best_dr = DeceptionResult(
        game=game,
        chosen_deception=D_sol,
        defender_strategy=y_sol,
        honest_value=np.inf,
        suboptimality_gap=np.inf,
        obtained_value=np.inf,
    )

    for row in range(num_rows):
        m = Model()
        m.setParam("OutputFlag", 0)  # silent
        m.setParam("MIPGap", 0.01)
        m.setParam("TimeLimit", 60)
        m.setParam("NonConvex", 2)  # Enable non-convex QCP

        # Decision variables
        D = m.addVars(num_rows, num_cols, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="D")
        # Auxiliary for |D_ij|
        U = m.addVars(num_rows, num_cols, lb=0, name="U")

        y = m.addVars(num_cols, lb=0, ub=1, name="y")
        v = m.addVar(lb=-GRB.INFINITY, name="v")
        gamma = m.addVars(num_rows, lb=0, ub=1, name="alpha")
        u = m.addVar(lb=-GRB.INFINITY, name="lambda")

        # Enforce operator-1 norm: max_j ∑_i |D_ij| ≤ Δ
        # Linearize |D_ij| with U_ij ≥  D_ij, U_ij ≥ -D_ij
        for i in range(num_rows):
            for j in range(num_cols):
                m.addConstr(U[i, j] >= D[i, j])
                m.addConstr(U[i, j] >= -D[i, j])
        for j in range(num_cols):
            m.addConstr(
                gp.quicksum(U[i, j] for i in range(num_rows)) <= deception_budget,
                name=f"col1norm_{j}",
            )

        # Simplex constraints
        m.addConstr(y.sum() == 1, name="simplex_y")
        m.addConstr(gamma.sum() == 1, name="simplex_gamma")

        # Bilinear products via QConstr for McCormick envelopes
        w1 = m.addVars(num_rows, num_cols, name="w1")
        w2 = m.addVars(num_rows, num_cols, name="w2")
        for i in range(num_rows):
            for j in range(num_cols):
                m.addQConstr(w1[i, j] == D[i, j] * y[j])
                m.addQConstr(w2[i, j] == D[i, j] * gamma[i])

        # Primal feasibility: (G + D) y ≥ v·1
        for i in range(num_rows):
            m.addConstr(
                gp.quicksum(game.outcomes[i][j] * y[j] for j in range(num_cols))
                + gp.quicksum(w1[i, j] for j in range(num_cols))
                >= v,
                name=f"primal_row_{i}",
            )

        # Dual feasibility: Gᵀ gamma + Dᵀ gamma ≤ u·1
        for j in range(num_cols):
            m.addConstr(
                gp.quicksum(game.outcomes[i][j] * gamma[i] for i in range(num_rows))
                + gp.quicksum(w2[i, j] for i in range(num_rows))
                <= u,
                name=f"dual_col_{j}",
            )

        # Strong duality
        m.addConstr(v == u, name="strong_duality")

        # Objective
        m.setObjective(
            gp.quicksum(game.outcomes[row][j] * y[j] for j in range(num_cols)),
            GRB.MINIMIZE,
        )

        m.optimize()

        D_sol = np.array(
            [[D[i, j].X for j in range(num_cols)] for i in range(num_rows)]
        )
        y_sol = np.array([y[j].X for j in range(num_cols)])
        gamma_sol = np.array([gamma[i].X for i in range(num_rows)])

        honest_value, y_lp, gamma_lp = game.solve()
        suboptimality_gap = honest_value - v.X
        achieved_value = np.min(game.outcomes @ y_sol)

        # print(f"Chose deception:\n{D_sol}")
        # print(f"Induced defender strategy:\n{y_sol} (QP),\n{y_lp} (LP)")
        # print(f"Defender perceived dual variables:\n{gamma_sol} (QP),\n{gamma_lp} (LP)")

        row_dr = DeceptionResult(
            game=game,
            chosen_deception=D_sol,
            defender_strategy=y_sol,
            honest_value=honest_value,
            suboptimality_gap=suboptimality_gap,
            obtained_value=achieved_value,
        )

        if row_dr and (row_dr.obtained_value <= best_dr.obtained_value):
            best_dr = row_dr

    return best_dr


# TODO: robust LP bound objective by dv parameter variation

if __name__ == "__main__":
    NUM_GAMES = 1
    NUM_ROWS = 5
    NUM_COLS = 5

    rng = np.random.default_rng(42)
    games = rng.uniform(low=-5, high=5, size=(NUM_GAMES, NUM_ROWS, NUM_COLS))

    deception_budgets = [1, 2, 3]
    all_improvements = []

    for deception_budget in deception_budgets:
        improvements = np.empty(NUM_GAMES, dtype=float)
        for i, game in enumerate(games):
            dv = 0.01
            dr = attacker_mccormick_env(ZSGame(game), deception_budget, dv)
            improvements[i] = dr.honest_value - dr.obtained_value
            assert dr.suboptimality_gap <= dv or np.allclose(
                dr.suboptimality_gap, dv
            ), (
                f"Suboptimality gap is too large. Expected <={dv}, got {dr.suboptimality_gap}"
            )
        assert np.all(improvements >= 0), (
            "Attacker became worse off by using deception in some games"
        )
        all_improvements.append(improvements)

    means = [imps.mean() for imps in all_improvements]
    stds = [imps.std() for imps in all_improvements]

    # plt.figure()
    # plt.plot(deception_budgets, means, marker="o", label="Mean improvement")
    # plt.fill_between(
    #     deception_budgets,
    #     [m - s for m, s in zip(means, stds)],
    #     [m + s for m, s in zip(means, stds)],
    #     color="b",
    #     alpha=0.2,
    #     label="Std deviation",
    # )
    # plt.xlabel("Deception budget")
    # plt.ylabel("Improvement")
    # plt.title("Improvement vs Deception Budget")
    # plt.legend()
    # plt.show()
