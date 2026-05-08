"""Tests for startup-batch experiment CSV helpers."""

from __future__ import annotations

import numpy as np
import pytest

from bioprocess_twin.experiments import (
    SI_STATE_COLUMNS,
    hourly_t_eval_for_startup_days,
    trajectory_to_dataframe,
)
from bioprocess_twin.models.stoichiometry import N_STATE


def test_hourly_t_eval_three_days_has_seventy_three_points() -> None:
    t = hourly_t_eval_for_startup_days(3.0)
    assert t.shape == (73,)
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(72.0)


def test_hourly_t_eval_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        hourly_t_eval_for_startup_days(0.0)


def test_trajectory_to_dataframe_columns() -> None:
    n = 5
    t = np.linspace(0.0, 4.0, n)
    y = np.ones((n, N_STATE))
    v = np.full(n, 10.0)
    df = trajectory_to_dataframe(t, y, v)
    assert list(df.columns[:2]) == ["t_hours", "volume_m3"]
    assert list(df.columns[2:]) == list(SI_STATE_COLUMNS)
    assert len(df.columns) == 2 + N_STATE


def test_trajectory_to_dataframe_bad_shape() -> None:
    with pytest.raises(ValueError, match="expected y shape"):
        trajectory_to_dataframe(np.array([0.0]), np.ones((1, 3)), np.ones(1))
