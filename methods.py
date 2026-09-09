from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from zs_game import ZSGame

FArray = npt.NDArray[np.float64]

DV: float = 1e-3


@dataclass
class DeceptionResult:
    game: ZSGame
    deceiver_strategy: FArray
    deception: FArray
    victim_strategy: FArray
    honest_value: float
    obtained_value: float
    # True if the solver stopped on its time limit, so the result is only the best
    # incumbent found and the timing is a lower bound.
    truncated: bool = False


Solver = Callable[[ZSGame, float], DeceptionResult]


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
    # verified &= _is_victim_rational(dr)
    verified &= _does_deceiver_benefit(dr)
    return verified


class DeceptionMethod(StrEnum):
    """
    Serializable identity for each deception solver.

    The string values are the tokens used in params.yaml, DVC matrix coordinates,
    output filenames, and the `method` column of every result table.
    """

    NONCONVEX = "nonconvex"
    BINSEARCH_ROBUST = "binsearch_robust"
    BINSEARCH_NONROBUST = "binsearch_nonrobust"

    @property
    def label(self) -> str:
        """Human-readable name, used for plot legends."""
        return _LABELS[self]


_LABELS = {
    DeceptionMethod.NONCONVEX: "Nonconvex",
    DeceptionMethod.BINSEARCH_ROBUST: "Binary Search (Robust)",
    DeceptionMethod.BINSEARCH_NONROBUST: "Binary Search (Non-Robust)",
}


def solver_for(
    method: DeceptionMethod,
    dv: float = DV,
    time_limit: float = 60.0,
    mip_gap: float = 0.01,
    gurobi_seed: int = 0,
    gurobi_threads: int = 0,
) -> Solver:
    """
    Resolves a method to a solver with the common `f(game, budget)` call shape.

    Args:
        method: Which deception solver to use.
        dv: Binary search tolerance on the victim's rationality level (binsearch only).
        time_limit: Gurobi time limit in seconds (nonconvex only).
        mip_gap: Relative MIP gap at which Gurobi stops (nonconvex only).
        gurobi_seed: Gurobi's internal random seed.
        gurobi_threads: Gurobi thread count; 0 lets Gurobi choose.

    Returns:
        A callable taking (game, deception_budget) and returning a DeceptionResult.
    """
    # Imported here, not at module scope: the solver modules import DeceptionResult
    # from this one.
    from deceiver_binary_search import deception_binary_search
    from deceiver_nonconvex import deception_nonconvex

    match method:
        case DeceptionMethod.NONCONVEX:
            return lambda game, budget: deception_nonconvex(
                game,
                budget,
                time_limit=time_limit,
                mip_gap=mip_gap,
                gurobi_seed=gurobi_seed,
                gurobi_threads=gurobi_threads,
            )
        case DeceptionMethod.BINSEARCH_ROBUST | DeceptionMethod.BINSEARCH_NONROBUST:
            robustify = method is DeceptionMethod.BINSEARCH_ROBUST
            return lambda game, budget: deception_binary_search(
                game,
                budget,
                dv=dv,
                robustify=robustify,
                gurobi_seed=gurobi_seed,
                gurobi_threads=gurobi_threads,
            )
