"""Draws (or reads off the command line) the batch of payoff matrices one stage solves."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

FArray = npt.NDArray[np.float64]

DEFAULT_SEED: int = 42

GAME_LOW: float = -5.0
GAME_HIGH: float = 5.0


def derive_rng(seed: int, *labels: str | int) -> np.random.Generator:
    """
    Builds an independent random stream keyed by a base seed and a set of labels.

    The stream depends only on (seed, labels), never on how many other streams have been
    drawn from, so callers may sample batches in any order or in separate processes and
    obtain identical results. Labels are folded with blake2b rather than the builtin
    hash(), which is salted per interpreter process.

    Args:
        seed: Base seed shared by the whole experiment suite.
        labels: Coordinates identifying this particular stream (e.g. "timing", 5).

    Returns:
        A NumPy Generator seeded from (seed, labels).
    """
    key = "|".join(str(label) for label in labels).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng([seed, int.from_bytes(digest, "big")])


def sample_games(
    seed: int,
    num_games: int,
    num_rows: int,
    num_cols: int,
    low: float = GAME_LOW,
    high: float = GAME_HIGH,
    label: str = "",
) -> FArray:
    """
    Draws a batch of security-style payoff matrices G[i, j] = v[j] - c[i].

    Each game gets a vector of num_cols resource values v ~ U[0, high] and a vector of
    num_rows security procedure costs c ~ U[0, -low], so entries span [low, high] but,
    unlike i.i.d. entries, every matrix has rank at most two.

    Args:
        seed: Base seed shared by the whole experiment suite.
        num_games: Number of matrices in the batch.
        num_rows: Rows (deceiver actions) per matrix.
        num_cols: Columns (victim actions) per matrix.
        low: Lower bound on payoff entries; -low bounds the procedure costs.
        high: Upper bound on payoff entries; bounds the resource values.
        label: Optional extra key, to keep identically shaped batches independent.

    Returns:
        An array of shape (num_games, num_rows, num_cols).
    """
    rng = derive_rng(seed, label, num_games, num_rows, num_cols)
    values = rng.uniform(low=0.0, high=high, size=(num_games, 1, num_cols))
    costs = rng.uniform(low=0.0, high=-low, size=(num_games, num_rows, 1))
    return values - costs


def save_games(path: str, games: FArray, seed: int, label: str) -> None:
    """Writes a game batch plus the provenance needed to regenerate it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, games=games, seed=seed, label=label)


def load_games(path: str) -> FArray:
    """Reads a game batch written by save_games."""
    with np.load(path, allow_pickle=False) as data:
        return data["games"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_rows", type=int, required=True)
    parser.add_argument("--num_cols", type=int, required=True)
    parser.add_argument("--num_games", type=int, default=1)
    parser.add_argument(
        "--outcomes",
        type=float,
        nargs="+",
        help="literal payoff entries, row major; entries are sampled if omitted",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--low", type=float, default=GAME_LOW)
    parser.add_argument("--high", type=float, default=GAME_HIGH)
    parser.add_argument(
        "--label",
        default="",
        help="extra seeding key, to keep identically shaped batches independent",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.outcomes is None:
        games = sample_games(
            seed=args.seed,
            num_games=args.num_games,
            num_rows=args.num_rows,
            num_cols=args.num_cols,
            low=args.low,
            high=args.high,
            label=args.label,
        )
    else:
        games = np.asarray(args.outcomes, dtype=float).reshape(
            -1, args.num_rows, args.num_cols
        )

    save_games(args.out, games, seed=args.seed, label=args.label)


if __name__ == "__main__":
    main()
