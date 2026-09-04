import inspect

import pytest

from methods import DeceptionMethod, solver_for


@pytest.mark.parametrize("method", list(DeceptionMethod))
def test_enum_round_trips_through_str(method):
    assert DeceptionMethod(str(method)) is method
    assert method.label


def test_values_are_filename_safe():
    for method in DeceptionMethod:
        assert str(method).replace("_", "").isalnum()


@pytest.mark.parametrize("method", list(DeceptionMethod))
def test_solver_has_the_common_call_shape(method):
    solver = solver_for(method)
    assert len(inspect.signature(solver).parameters) == 2
