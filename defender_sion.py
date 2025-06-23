from zs_game import ZSGame

import numpy as np
import numpy.typing as npt
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

from dataclasses import dataclass

from typing import Tuple, List

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
    attacker_robust_val: float
    attacker_nash_val: float
    attacker_br_val: float
    full_knowledge_val: float

    def __str__(self):
        return (
            f"The defender viewed the (disturbed) game\n{self.dgame.outcomes}\n"
            f"The induced security game is\n{self.worst_case_game.outcomes}\n\n"
            f"The defender chose strategy\n{self.robust_y}\n"
            f"The attacker chose strategy\n{self.robust_x}\n\n"
            f"The defender's security value is {self.security_val:.3f}\n"
            f"The outcome of the believed game is {self.worst_case_game.outcome(self.robust_x, self.robust_y):.3f}\n"
            f"The attacker, without knowledge of the disturbance, would cause an outcome of {self.attacker_robust_val:.3f}\n"
            f"The attacker, playing the Nash equilibrium of the original game, would cause an outcome of {self.attacker_nash_val:.3f}\n"
            f"The attacker, playing a best response to the defender's robust strategy, would cause an outcome of {self.attacker_br_val:.3f}\n"
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

    x = m.addVars(num_rows, lb=0, name="x")
    selector = m.addVars(num_rows, vtype=GRB.BINARY, name="selector")
    alpha = m.addVar(lb=-GRB.INFINITY, name="alpha")
    beta = m.addVar(lb=-GRB.INFINITY, name="beta")

    # Objective: minimize alpha + deception_budget * beta
    m.setObjective(alpha - deception_budget * beta, GRB.MINIMIZE)

    # Constraints
    # Enforce attacker probability distribution: sum_i x[i] = 1
    m.addConstr(quicksum(x[i] for i in range(num_rows)) == 1, "prob_sum")

    # For each column j: sum_i x[i] * dgame.outcomes[i, j] <= alpha
    m.addConstrs(
        (
            quicksum(x[i] * dgame.outcomes[i, j] for i in range(num_rows)) <= alpha
            for j in range(num_cols)
        ),
        "alpha_col",
    )

    # Select best row for attacker
    m.addConstr(quicksum(selector[i] for i in range(num_rows)) == 1, "select_one_row")
    m.addConstrs(
        (beta <= x[i] + M * (1 - selector[i]) for i in range(num_rows)), "row_selection"
    )

    m.optimize()

    # lp_model, duals = fix_integers_and_get_duals(m)

    # Extract solution
    x_sol = np.array([x[i].X for i in range(num_rows)], dtype=float)
    y_ind = np.argmax(x_sol @ dgame.outcomes)
    y_sol = np.zeros(num_cols, dtype=float)
    y_sol[y_ind] = 1.0  # Defender plays a pure strategy against the worst-case game
    # TODO: test with best response to LP

    return m.ObjVal, x_sol, y_sol


def deception_robust_solve(
    game: ZSGame, deception_budget: float, num_deceptions: int = 10
) -> List[DeceptiveGame]:
    """
    Solves the robust defender strategy problem for a given game and deception budget,
    sampling num_deceptions disturbed games.
    Args:
        game: The original game.
        deception_budget: The budget for deception.
        num_deceptions: Number of disturbed games to sample.
    Returns:
        A list of DeceptiveGame objects containing the results.
    """
    results = []
    for _ in range(num_deceptions):
        dgame = ZSGame.random_disturbance(game, deception_budget)
        robust_val, robust_x, robust_y = robust_defender_strategy(
            dgame, deception_budget
        )

        # Compute the worst-case game
        outer = np.outer(robust_x, robust_y)
        worst_row = np.argmax(np.sum(outer, axis=1))
        worst_D = np.zeros_like(game.outcomes)
        worst_D[worst_row, :] = deception_budget
        worst_case_game = dgame.from_disturbance(-worst_D)

        full_knowledge_val, full_knowledge_x, full_knowledge_y = game.solve()
        attacker_robust_val = game.outcome(
            robust_x, robust_y
        ).item()  # best attacker can do without knowledge of G
        attacker_nash_val = game.outcome(
            full_knowledge_x, robust_y
        ).item()  # Assume attacker just plays Nash for original game
        attacker_br_val = np.min(
            game.outcomes @ robust_y
        )  # Assume attacker perfectly predicts robust defender strategy

        support = robust_y != 0
        chosen_game = ZSGame(game.outcomes[:, support])
        # assert robust_val <= chosen_game.solve()[0], (
        #     f"Robust value {robust_val} exceeds chosen game value {chosen_game.solve()[0]}"
        # )

        if robust_val > chosen_game.solve()[0]:
            print(
                f"Robust value {robust_val:.3g} exceeds chosen game value {chosen_game.solve()[0]:.3g}"
            )

        # assert robust_val <= attacker_br_val

        results.append(
            DeceptiveGame(
                game=game,
                dgame=dgame,
                worst_case_game=worst_case_game,
                robust_x=robust_x,
                robust_y=robust_y,
                security_val=robust_val,
                attacker_robust_val=attacker_robust_val,
                attacker_nash_val=attacker_nash_val,
                attacker_br_val=attacker_br_val,
                full_knowledge_val=full_knowledge_val,
            )
        )
    return results


if __name__ == "__main__":
    NUM_GAMES = 10
    NUM_ROWS = 5
    NUM_COLS = 5
    NUM_DECEPTIONS = 50
    DECEPTION_BUDGETS = range(1, 6)

    rng = np.random.default_rng(42)
    games = rng.uniform(low=-5, high=5, size=(NUM_GAMES, NUM_ROWS, NUM_COLS))

    avg_security_vals = []
    avg_attacker_robust_vals = []
    avg_attacker_nash_vals = []
    avg_attacker_br_vals = []
    avg_full_knowledge_vals = []

    per_game_sec_vals = [[] for _ in range(NUM_GAMES)]
    per_game_attacker_robust_vals = [[] for _ in range(NUM_GAMES)]
    per_game_attacker_nash_vals = [[] for _ in range(NUM_GAMES)]
    per_game_attacker_br_vals = [[] for _ in range(NUM_GAMES)]
    per_game_fk_vals = [[] for _ in range(NUM_GAMES)]

    for deception_budget in DECEPTION_BUDGETS:
        sec_vals = []
        attacker_robust_vals = []
        attacker_nash_vals = []
        attacker_br_vals = []
        fk_vals = []
        for game_idx, game in enumerate(games):
            deception_results = deception_robust_solve(
                ZSGame(game), deception_budget, NUM_DECEPTIONS
            )
            mean_sec_val = np.mean([d.security_val for d in deception_results])
            mean_attacker_robust_val = np.mean(
                [d.attacker_robust_val for d in deception_results]
            )
            mean_attacker_nash_val = np.mean(
                [d.attacker_nash_val for d in deception_results]
            )
            mean_attacker_br_val = np.mean(
                [d.attacker_br_val for d in deception_results]
            )
            mean_fk_val = np.mean([d.full_knowledge_val for d in deception_results])
            sec_vals.append(mean_sec_val)
            attacker_robust_vals.append(mean_attacker_robust_val)
            attacker_nash_vals.append(mean_attacker_nash_val)
            attacker_br_vals.append(mean_attacker_br_val)
            fk_vals.append(mean_fk_val)
            per_game_sec_vals[game_idx].append(mean_sec_val)
            per_game_attacker_robust_vals[game_idx].append(mean_attacker_robust_val)
            per_game_attacker_nash_vals[game_idx].append(mean_attacker_nash_val)
            per_game_attacker_br_vals[game_idx].append(mean_attacker_br_val)
            per_game_fk_vals[game_idx].append(mean_fk_val)
        avg_security_vals.append(np.mean(sec_vals))
        avg_attacker_robust_vals.append(np.mean(attacker_robust_vals))
        avg_attacker_nash_vals.append(np.mean(attacker_nash_vals))
        avg_attacker_br_vals.append(np.mean(attacker_br_vals))
        avg_full_knowledge_vals.append(np.mean(fk_vals))

    # Plot averages
    (line1,) = plt.plot(
        list(DECEPTION_BUDGETS), avg_security_vals, label="Avg Security Value"
    )
    (line2,) = plt.plot(
        list(DECEPTION_BUDGETS),
        avg_attacker_robust_vals,
        label="Avg Attacker Robust Value",
    )
    (line3,) = plt.plot(
        list(DECEPTION_BUDGETS), avg_attacker_nash_vals, label="Avg Attacker Nash Value"
    )
    (line4,) = plt.plot(
        list(DECEPTION_BUDGETS),
        avg_attacker_br_vals,
        label="Avg Attacker Best Response Value",
    )
    (line5,) = plt.plot(
        list(DECEPTION_BUDGETS),
        avg_full_knowledge_vals,
        label="Avg Full Knowledge Value",
    )

    # # Plot per-game values as thin, dotted lines in the same color as the average
    # for vals, avg_line in zip(
    #     [
    #         per_game_sec_vals,
    #         per_game_attacker_robust_vals,
    #         per_game_attacker_nash_vals,
    #         per_game_attacker_br_vals,
    #         per_game_fk_vals,
    #     ],
    #     [line1, line2, line3, line4, line5],
    # ):
    #     avg_color = avg_line.get_color()
    #     for game_vals in vals:
    #         plt.plot(
    #             list(DECEPTION_BUDGETS),
    #             game_vals,
    #             color=avg_color,
    #             linestyle=":",
    #             linewidth=1,
    #             alpha=0.7,
    #         )

    plt.xlabel("Deception Budget")
    plt.ylabel("Value")
    plt.title("Averages vs Deception Budget")
    plt.legend()
    plt.tight_layout()
    plt.show()
