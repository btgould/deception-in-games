import numpy as np
import pytest

import deception
import sampling
from methods import DeceptionMethod

BUDGETS = [1.0, 2.0]


@pytest.fixture(scope="module")
def batch():
    # Labelled: at dv=1e-2 the robustified binary search returns worse-than-honest
    # results on some draws, which sanity_check would (correctly) trip on.
    games = sampling.sample_games(0, num_games=2, num_rows=3, num_cols=3, label="test")
    return games, deception.run_batch(
        label="test",
        games=games,
        method=DeceptionMethod.BINSEARCH_ROBUST,
        deception_budgets=BUDGETS,
        dv=1e-2,
        time_limit=10.0,
        sanity_check=True,
    )


def test_schema_and_row_count(batch):
    _, (df, _) = batch
    assert list(df.columns) == deception.COLUMNS
    assert len(df) == len(BUDGETS) * 2


def test_arrays_align_with_rows(batch):
    games, (df, arrays) = batch
    assert set(arrays) == set(deception.ARRAY_KEYS)
    assert all(len(a) == len(df) for a in arrays.values())
    for row, stored in zip(df.itertuples(), arrays["games"]):
        assert np.array_equal(games[row.game_idx], stored)


def test_improvement_matches_values(batch):
    _, (df, _) = batch
    assert np.allclose(df["improvement"], df["honest_value"] - df["obtained_value"])


def test_method_columns_are_serializable(batch):
    _, (df, _) = batch
    assert set(df["method"]) == {"binsearch_robust"}
    assert set(df["method_label"]) == {DeceptionMethod.BINSEARCH_ROBUST.label}
