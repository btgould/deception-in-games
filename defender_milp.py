from zs_game import ZSGame

import numpy as np
import numpy.typing as npt
from gurobipy import Model, GRB

from dataclasses import dataclass

from typing import Tuple

FArray = npt.NDArray[np.float64]
M = 2  # big-M constant (needs to select against probability weights, anything >= 1 is sufficiently large)


@dataclass
class DeceptiveGame:
    game: ZSGame
    dgame: ZSGame
    worst_case_game: ZSGame

    robust_x: FArray
    robust_y: FArray

    security_val: float
    outcome_val: float
    full_knowledge_val: float

    def __str__(self):
        return (
            f"The defender viewed the (disturbed) game\n{self.dgame.outcomes}\n"
            f"The induced security game is\n{self.worst_case_game.outcomes}\n\n"
            f"The defender chose strategy\n{self.robust_y}\n"
            f"The attacker chose strategy\n{self.robust_x}\n\n"
            f"The defender's security value is {self.security_val:.3f}\n"
            f"The outcome of the believed game is {self.worst_case_game.outcome(self.robust_x, self.robust_y):.3f}\n"
            f"The outcome of the original game is {self.outcome_val:.3f}\n"
            f"If the defender had full knowledge, the outcome would be {self.full_knowledge_val:.3f}"
        )


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
    m.setParam("OutputFlag", 0)  # silent

    # Variables: x (num_rows), alpha, beta
    x = m.addVars(num_rows, lb=0, name="x")
    alpha = m.addVar(lb=-GRB.INFINITY, name="alpha")
    beta = m.addVar(name="beta")
    selector = m.addVars(num_rows, vtype=GRB.BINARY, name="selector")

    # Objective: minimize alpha + deception_budget * beta
    m.setObjective(alpha - deception_budget * beta, GRB.MINIMIZE)

    # Constraints
    # Enforce attacker probability distribution: sum_i x[i] = 1
    m.addConstr(sum(x[i] for i in range(num_rows)) == 1, "prob_sum")

    # For each column j: sum_i x[i] * dgame.outcomes[i, j] <= alpha
    for j in range(num_cols):
        m.addConstr(
            sum(x[i] * dgame.outcomes[i, j] for i in range(num_rows)) <= alpha,
            f"alpha_col_{j}",
        )

    # For each row i: x[i] <= beta
    for i in range(num_rows):
        m.addConstr(x[i] + M * (1 - selector[i]) >= beta, f"beta_row_{i}")

    # Only one row can be selected
    m.addConstr(sum(selector[i] for i in range(num_rows)) == 1, "selector_sum")

    m.optimize()

    # Extract solution
    x_sol = np.array([x[i].X for i in range(num_rows)], dtype=float)

    gp_comps = x_sol @ dgame.outcomes
    max_gp_comp = np.max(gp_comps)
    valid_strats = np.isclose(gp_comps, max_gp_comp, atol=1e-9).astype(float)
    y_sol = (
        valid_strats / np.sum(valid_strats)
    )  # TODO: choosing a uniform distribution here is arbitrary, and may affect the outcome of the true game (but will not affect the security value)

    # For dual variables (defender strategy), use shadow prices of alpha constraints
    # robust_y = np.array(
    #     [-m.getConstrByName(f"alpha_col_{j}").Pi for j in range(num_cols)]
    # )

    return m.ObjVal, x_sol, y_sol


def deception_robust_solve(game: ZSGame, deception_budget: float) -> DeceptiveGame:
    """
    Solves the robust defender strategy problem for a given game and deception budget.
    Args:
        game: The original game.
        deception_budget: The budget for deception.
    Returns:
        A DeceptiveGame object containing the results.
    """
    dgame = ZSGame.random_disturbance(game, deception_budget)
    robust_val, robust_x, robust_y = robust_defender_strategy(dgame, deception_budget)

    # Compute the worst-case game
    outer = np.outer(robust_x, robust_y)
    worst_row = np.argmax(
        np.sum(outer, axis=1)
    )  # don't need abs here, as x and y are non-negative
    worst_D = np.zeros_like(game.outcomes)
    worst_D[worst_row, :] = (
        deception_budget  # don't need sign check here, as x and y are non-negative
    )
    worst_case_game = dgame.from_disturbance(-worst_D)

    outcome_val = game.outcome(robust_x, robust_y)
    full_knowledge_val = game.solve()[0]

    return DeceptiveGame(
        game=game,
        dgame=dgame,
        worst_case_game=worst_case_game,
        robust_x=robust_x,
        robust_y=robust_y,
        security_val=robust_val,
        outcome_val=outcome_val.item(),
        full_knowledge_val=full_knowledge_val,
    )


if __name__ == "__main__":
    np.random.seed(5)
    deception_budget = 2
    game = ZSGame(np.array([[0.0, 1, -1], [-1, 0, 1], [1, -1, 0]]))

    deceptive_game = deception_robust_solve(game, deception_budget)
    print(deceptive_game)
