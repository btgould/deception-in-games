from zs_game import ZSGame
from deception import DeceptionResult

import numpy as np
import numpy.typing as npt

import gurobipy as gp
from gurobipy import Model, GRB

FArray = npt.NDArray[np.float64]

DV: float = 1e-6


def _tolerance_bound(
    dr: DeceptionResult, subrationaltiy_level: float
) -> DeceptionResult:
    """
    Accounts for the possible error induced by numerical tolerances in the binary search procedure on the victim's rationality level.

    Args:
        dr: A DeceptionResult output by the binary search procedure
        subrationaltiy_level: victim's subrationality level

    Returns:
        A DeceptionResult with the deceiver worst-case victim strategy realized (subject to rationality constraints)
    """

    num_rows, num_cols = dr.game.outcomes.shape

    m = Model()
    m.setParam("OutputFlag", 0)  # silent

    y = m.addVars(num_cols, name="y")

    m.addConstr(gp.quicksum(y[i] for i in range(num_cols)) == 1)
    m.addConstrs(
        gp.quicksum(
            (dr.game.outcomes + dr.deception)[i, j] * y[j] for j in range(num_cols)
        )
        >= subrationaltiy_level
        for i in range(num_rows)
    )

    c = dr.deceiver_strategy @ dr.game.outcomes
    m.setObjective(
        gp.quicksum(c[j] * y[j] for j in range(num_cols)),
        GRB.MAXIMIZE,
    )

    m.optimize()

    dr.obtained_value = m.ObjVal
    dr.victim_strategy = np.array([y[j].X for j in range(num_cols)], dtype=float)

    return dr


def deception_binary_search(
    game: ZSGame, deception_budget: float, dv: float = DV
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
    best_v_step = min_val
    v_sol = min_val

    D_sol = np.empty_like(game.outcomes)
    x_sol = np.empty(num_rows, dtype=float)
    y_sol = np.empty(num_cols, dtype=float)
    best_dr = DeceptionResult(
        game=game,
        deceiver_strategy=x_sol,
        deception=D_sol,
        victim_strategy=y_sol,
        honest_value=np.inf,
        obtained_value=np.inf,
    )

    for row in range(num_rows):
        low = min_val
        high = max_val
        row_dr = None

        while high - low > dv:
            v_step = (low + high) / 2
            m = Model()
            m.setParam(GRB.Param.OutputFlag, 0)  # silent
            m.setParam(GRB.Param.FeasibilityTol, 1e-9)

            y = m.addVars(num_cols, name="y")
            d = m.addVars(num_rows, lb=-GRB.INFINITY, name="d")
            t = m.addVars(num_rows, name="t")

            m.setObjective(
                gp.quicksum(game.outcomes[row, j] * y[j] for j in range(num_cols)),
                GRB.MINIMIZE,
            )

            m.addConstrs(
                (
                    gp.quicksum(game.outcomes[i, j] * y[j] for j in range(num_cols))
                    + d[i]
                    >= v_step
                    for i in range(num_rows)
                ),
                "rationality",
            )
            m.addConstrs(d[i] <= t[i] for i in range(num_rows))
            m.addConstrs(-d[i] <= t[i] for i in range(num_rows))

            m.addConstr(gp.quicksum(y[i] for i in range(num_cols)) == 1)
            m.addConstr(gp.quicksum(t[i] for i in range(num_rows)) <= deception_budget)

            m.optimize()

            if m.status == GRB.OPTIMAL:
                low = v_step

                y_sol = np.array([y[j].X for j in range(num_cols)], dtype=float)
                d_sol = np.array([d[i].X for i in range(num_rows)], dtype=float)
                D_sol = np.repeat(d_sol[:, np.newaxis], num_cols, axis=1)
                v_sol = v_step
            else:
                high = v_step

        honest_value = game.solve()[0]
        achieved_value = np.min(game.outcomes @ y_sol)
        decever_strategy = np.zeros(num_rows, dtype=float)
        decever_strategy[row] = 1.0

        row_dr = DeceptionResult(
            game=game,
            deceiver_strategy=decever_strategy,
            deception=D_sol,
            victim_strategy=y_sol,
            honest_value=honest_value,
            obtained_value=achieved_value,
        )

        if row_dr and (row_dr.obtained_value <= best_dr.obtained_value):
            best_dr = row_dr
            best_v_step = v_sol

    best_dr = _tolerance_bound(best_dr, best_v_step)
    return best_dr


# NOTE: this does not robust LP bound objective by dv parameter variation
