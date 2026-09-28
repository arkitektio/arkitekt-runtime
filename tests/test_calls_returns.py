"""``ensure_return_as_tuple`` spreads a hook's returns; only ``None`` means none."""

import pytest

from arkitekt_runtime.calls import ensure_return_as_tuple


@pytest.mark.parametrize("value", [0, "", [], {}, False])
def test_falsy_returns_are_kept(value: object) -> None:
    assert ensure_return_as_tuple(value) == (value,)


def test_none_is_no_return() -> None:
    assert ensure_return_as_tuple(None) == ()


def test_tuples_are_spread() -> None:
    assert ensure_return_as_tuple((1, 2)) == (1, 2)
