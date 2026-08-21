import gurobipy as gp
import numpy as np
import numpy.typing as npt
from gurobipy import GRB, Model

from deception import DeceptionResult
from zs_game import ZSGame

FArray = npt.NDArray[np.float64]


def deception_nonconvex(
    game: ZSGame, deception_budget: float, time_limit: float = 60
) -> DeceptionResult:
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
    m.setParam("TimeLimit", time_limit)
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
