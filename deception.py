from zs_game import ZSGame

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

FArray = npt.NDArray[np.float64]


@dataclass
class DeceptionResult:
    game: ZSGame
    deceiver_strategy: FArray
    deception: FArray
    victim_strategy: FArray
    honest_value: float
    obtained_value: float


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
