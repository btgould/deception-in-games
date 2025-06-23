from zs_game import ZSGame

import numpy as np
import numpy.typing as npt
from gurobipy import Model, GRB

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


def attacker_binary_search(
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
        low = min_val
        high = max_val
        row_dr = None
        v_step = (low + high) / 2

        while high - low > dv:
            m = Model()
            m.setParam("OutputFlag", 0)  # silent

            y = m.addVars(num_cols, name="y")
            d = m.addVars(num_rows, lb=-GRB.INFINITY, name="d")
            t = m.addVars(num_rows, name="t")

            m.setObjective(
                sum(game.outcomes[row, j] * y[j] for j in range(num_cols)), GRB.MINIMIZE
            )

            for i in range(num_rows):
                m.addConstr(
                    sum(game.outcomes[i, j] * y[j] for j in range(num_cols)) + d[i]
                    >= v_step
                )
                m.addConstr(d[i] <= t[i])
                m.addConstr(-d[i] <= t[i])

            m.addConstr(sum(y[i] for i in range(num_cols)) == 1)
            m.addConstr(sum(t[i] for i in range(num_rows)) <= deception_budget)

            m.optimize()

            if m.status == GRB.OPTIMAL:
                low = v_step

                y_sol = np.array([y[j].X for j in range(num_cols)], dtype=float)
                d_sol = np.array([d[i].X for i in range(num_rows)], dtype=float)
                D_sol = np.repeat(d_sol[:, np.newaxis], num_cols, axis=1)
            else:
                high = v_step

            v_step = (low + high) / 2

        honest_value = game.solve()[0]
        suboptimality_gap = honest_value - v_step
        achieved_value = np.min(game.outcomes @ y_sol)

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
