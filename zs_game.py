import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

FArray = npt.NDArray[np.float64]


class ZSGame:
    """
    Represents a two-player, zero-sum game.
    The row player minimizes the outcome, while the column player maximizes the outcome.

    Attributes:
        outcomes: Matrix of outcomes where the entry at (i, j) is the payoff to the column
        player when the row player plays i and the column player plays j.
    """

    outcomes: FArray

    def __init__(self, outcomes: FArray):
        """
        Construct a zero-sum game from a given payoff matrix.
        Args:
            game_matrix: A NumPy matrix representing the game payoffs.
        """
        self.outcomes = outcomes

    @classmethod
    def random(
        cls,
        num_rows: int,
        num_cols: int,
        rng: np.random.Generator,
        low: float = -5.0,
        high: float = 5.0,
    ):
        """
        Alternative constructor to initialize a random, 2 player, zero-sum game.
        Args:
            num_rows: Number of rows (actions for p1) in the game
            num_cols: Number of cols (actions for p2) in the game
            rng: Random number generator to use
            low: Lower bound on payoff entries
            high: Upper bound on payoff entries
        """
        instance = cls(rng.uniform(low=low, high=high, size=(num_rows, num_cols)))
        return instance

    def to_linprog(self) -> tuple[FArray, FArray, FArray, FArray, FArray]:
        """
        Converts the game to a linear program format.
        Returns:
            Tuple of (c, A_ub, b_ub, A_eq, b_eq) for the linear program.
        """
        n_rows, n_cols = self.outcomes.shape

        # State vector is action weights for column player + value of game (split into 2 since it is unbounded)
        # Column player wants to maximize the worst case outcome
        c = np.zeros(n_cols + 2)
        c[-2:] = [-1, 1]

        # Ensure v bounds every choice of row player from resulting payoffs of column players strategy
        # v <= G @ sigma_2 --> -G @ sigma_2 + v <= 0
        A_ub = np.hstack([-self.outcomes, np.ones((n_rows, 1)), -np.ones((n_rows, 1))])
        b_ub = np.zeros(n_rows)

        # Ensure that the column player's strategy is a probability distribution
        A_eq = np.ones((1, n_cols + 2))
        A_eq[0, -2:] = [0, 0]
        b_eq = np.array([1])

        return c, A_ub, b_ub, A_eq, b_eq

    def solve(self) -> tuple[float, FArray, FArray]:
        """
        Solves a zero-sum game between 2 players with the minimax theorem.
        Returns:
            Tuple of (value of game, row player's strategy, column player's strategy).
        """

        n_rows, _ = self.outcomes.shape

        c, A_ub, b_ub, A_eq, b_eq = self.to_linprog()
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

        if result.success:
            value = -result.fun
            col_strategy = result.x[:-2]
            row_strategy = -result.ineqlin.marginals[
                :n_rows
            ]  # Dual problem gives row strategy, write it out to see
            return value, row_strategy, col_strategy
        else:
            raise ValueError("LP failed to solve: " + result.message)

    def outcome(self, row_action: FArray, col_action: FArray) -> FArray:
        """
        Gets the outcome of the game given the actions of both players.
        Args:
            row_action: Action vector for the row player.
            col_action: Action vector for the column player.
        Returns:
            The payoff to the row player.
        """
        # assert row_action.shape == (self.outcomes.shape[0],)
        # assert col_action.shape == (self.outcomes.shape[1],)

        return row_action @ self.outcomes @ col_action

    def from_disturbance(self, disturbance: FArray) -> "ZSGame":
        """
        Generates a new ZSGame by adding a disturbance to the outcomes of this game.
        Args:
            disturbance: The disturbance to add to the outcomes.
        """
        instance = ZSGame(self.outcomes + disturbance)
        return instance

    @classmethod
    def random_disturbance(cls, game, rng: np.random.Generator, eps: float = 0.5):
        """
        Generates a new ZSGame by adding bounded, uniform noise to the outcomes of this game.
        The noise in bounded in such a way that the difference (in operator norm induced by 1
        norm) of the disturbed and original games is at most eps.
        Args:
            game: The game to disturb
            rng: Random number generator to use
            eps: Bound on the induced 1-norm of the disturbance
        """
        rng_rad = eps / (2 * game.outcomes.shape[0])
        disturbance = rng.uniform(low=-rng_rad, high=rng_rad, size=game.outcomes.shape)
        return game.from_disturbance(disturbance)

    def tree_flatten(self):
        return ((self.outcomes,), "ZSGame")

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


if __name__ == "__main__":
    print("Analyzing rock-paper-scissors")
    rps = ZSGame(np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]))
    val, row_strat, col_strat = rps.solve()
    print(f"Value: {val},\nRow Strategy: {row_strat},\nColumn Strategy: {col_strat}")

    print("Analyzing rock-paper-scissors (row player can't play scissors)")
    rps = ZSGame(np.array([[0, 1, -1], [-1, 0, 1]]))
    val, row_strat, col_strat = rps.solve()
    print(f"Value: {val},\nRow Strategy: {row_strat},\nColumn Strategy: {col_strat}")

    print("Analyzing wikipedia example")
    game = ZSGame(np.array([[-3.0, 1, -2], [1, -2, 2]]))
    val, row_strat, col_strat = game.solve()
    print(f"Value: {val},\nRow Strategy: {row_strat},\nColumn Strategy: {col_strat}")

    print("Analyzing disturbances")
    gmat = np.array([[-3, 0, 1]])
    dgmat = np.array([[-3, 1, 0]])
    print(np.linalg.norm(gmat, ord=1))
    print(np.linalg.norm(dgmat, ord=1))
    print(np.linalg.norm(gmat - dgmat, ord=1))

    game = ZSGame(np.array(gmat))
    val, row_strat, col_strat = game.solve()
    print(f"Value: {val},\nRow Strategy: {row_strat},\nColumn Strategy: {col_strat}")
    dgame = ZSGame(np.array(dgmat))
    dval, drow_strat, dcol_strat = dgame.solve()
    print(
        f"DISTURBED: Value: {dval},\nRow Strategy: {drow_strat},\nColumn Strategy: {dcol_strat}"
    )
    print(row_strat @ (gmat - dgmat) @ dcol_strat + row_strat @ dgmat @ dcol_strat)
    print(row_strat @ dgmat @ dcol_strat)
