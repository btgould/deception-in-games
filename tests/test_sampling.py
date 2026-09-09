import subprocess
import sys

import numpy as np

import sampling


def test_sample_games_is_deterministic():
    a = sampling.sample_games(42, 5, 5, 5)
    b = sampling.sample_games(42, 5, 5, 5)
    assert np.array_equal(a, b)


def test_sample_games_stable_across_processes():
    """blake2b, unlike the builtin hash(), is not salted per interpreter."""
    code = (
        "import sampling;"
        "print(repr(sampling.sample_games(42, 2, 3, 3).tobytes().hex()))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    assert len(runs) == 1


def test_games_have_value_minus_cost_structure():
    """G[i, j] = v[j] - c[i] makes every 2x2 minor vanish."""
    games = sampling.sample_games(42, 5, 4, 6)
    minors = (
        games[:, :-1, :-1] - games[:, :-1, 1:] - games[:, 1:, :-1] + games[:, 1:, 1:]
    )
    assert np.allclose(minors, 0.0)


def test_games_respect_entry_bounds():
    games = sampling.sample_games(42, 5, 4, 6, low=-5.0, high=5.0)
    assert games.min() >= -5.0
    assert games.max() <= 5.0


def test_shape_keys_the_batch():
    a = sampling.sample_games(42, 5, 5, 5)
    b = sampling.sample_games(42, 5, 5, 6)[:, :, :5]
    assert not np.array_equal(a, b)


def test_distinct_labels_give_distinct_batches():
    a = sampling.sample_games(42, 5, 5, 5, label="compare")
    b = sampling.sample_games(42, 5, 5, 5, label="tolerance")
    assert not np.array_equal(a, b)


def test_batch_independent_of_other_sizes():
    """The bug this replaces drew every size from one advancing stream."""
    drawn_in_order = [sampling.sample_games(42, 4, n, n) for n in (2, 3, 4)]
    drawn_alone = sampling.sample_games(42, 4, 4, 4)
    assert np.array_equal(drawn_in_order[-1], drawn_alone)


def test_round_trip(tmp_path):
    games = sampling.sample_games(42, 3, 4, 5)
    path = str(tmp_path / "games.npz")
    sampling.save_games(path, games, seed=42, label="")
    assert np.array_equal(sampling.load_games(path), games)


def _run_cli(*args: str) -> None:
    subprocess.run([sys.executable, "sampling.py", *args], check=True)


def test_cli_samples_a_batch(tmp_path):
    path = tmp_path / "games.npz"
    _run_cli(
        "--seed", "7",
        "--num_games", "3",
        "--num_rows", "2",
        "--num_cols", "4",
        "--out", str(path),
    )  # fmt: skip
    assert np.array_equal(
        sampling.load_games(str(path)), sampling.sample_games(7, 3, 2, 4)
    )


def test_cli_writes_a_literal_matrix(tmp_path):
    """The RPS stage passes a flat, row-major matrix through the cmd."""
    path = tmp_path / "rps.npz"
    _run_cli(
        "--num_rows", "3",
        "--num_cols", "3",
        "--outcomes", "0", "1", "-1", "-1", "0", "1", "1", "-1", "0",
        "--out", str(path),
    )  # fmt: skip
    expected = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    assert np.array_equal(sampling.load_games(str(path)), expected[None])
