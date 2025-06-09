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
    num = int((max_val - min_val) / dv) + 1

    D_sol = np.empty_like(game.outcomes)
    y_sol = np.empty(num_cols, dtype=float)
    best_dr = DeceptionResult(
        game=game,
        chosen_deception=D_sol,
        defender_strategy=y_sol,
        suboptimality_gap=np.inf,
        obtained_value=np.inf,
    )

    for row in range(num_rows):
        low = min_val
        high = max_val
        best_row_dr = None

        while high - low > dv:
            v_step = (low + high) / 2
            # print(f"Computing for {row=} with {v_step=}")

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

                if high - low <= dv:
                    y_sol = np.array([y[j].X for j in range(num_cols)], dtype=float)
                    d_sol = np.array([d[i].X for i in range(num_rows)], dtype=float)
                    D_sol = np.repeat(d_sol[:, np.newaxis], num_cols, axis=1)

                    dgame = game.from_disturbance(D_sol)
                    suboptimality_gap = dgame.solve()[0] - v_step
                    achieved_value = np.min(game.outcomes @ y_sol)

                    best_row_dr = DeceptionResult(
                        game=game,
                        chosen_deception=D_sol,
                        defender_strategy=y_sol,
                        suboptimality_gap=suboptimality_gap,
                        obtained_value=achieved_value,
                    )
            else:
                high = v_step

        # print(best_row_dr)

        if best_row_dr and (best_row_dr.obtained_value <= best_dr.obtained_value):
            best_dr = best_row_dr

    return best_dr


# TODO: robust LP bound objective by dv parameter variation

if __name__ == "__main__":
    game = ZSGame(np.array([[0.0, 1, -1], [-1, 0, 1], [1, -1, 0]]))
    deception_budget = 2.0
    dv = 0.01
    dr = attacker_binary_search(game, deception_budget, dv)
    assert dr.suboptimality_gap < dv, "Suboptimality gap is too large"
