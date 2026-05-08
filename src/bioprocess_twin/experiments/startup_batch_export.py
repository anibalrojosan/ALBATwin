"""Shared helpers for startup-batch experiment scripts (CSV layout, hourly grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# SI layout column order matches ``StateVector.to_array()`` for 17 components.
SI_STATE_COLUMNS: tuple[str, ...] = (
    "X_ALG",
    "X_AOB",
    "X_NOB",
    "X_H",
    "X_S",
    "X_I",
    "S_S",
    "S_I",
    "S_IC",
    "S_ND",
    "S_NH",
    "S_NO2",
    "S_NO3",
    "S_N2",
    "S_PO4",
    "S_O2",
    "S_H2O",
)


def hourly_t_eval_for_startup_days(startup_days: float) -> np.ndarray:
    """Hourly sample grid ``t = 0, 1, …, startup_days * 24`` [h]."""
    if startup_days <= 0:
        raise ValueError("startup_days must be positive")
    t_end = float(startup_days) * 24.0
    n = int(round(t_end)) + 1
    return np.linspace(0.0, t_end, max(n, 2))


def trajectory_to_dataframe(t_hours: np.ndarray, y: np.ndarray, volume_m3: np.ndarray) -> pd.DataFrame:
    """Build a wide table: ``t_hours``, ``volume_m3``, then 17 SI columns."""
    if y.ndim != 2 or y.shape[1] != len(SI_STATE_COLUMNS):
        raise ValueError(f"expected y shape (n, {len(SI_STATE_COLUMNS)}), got {y.shape}")
    data: dict[str, np.ndarray] = {
        "t_hours": np.asarray(t_hours, dtype=np.float64).ravel(),
        "volume_m3": np.asarray(volume_m3, dtype=np.float64).ravel(),
    }
    for i, name in enumerate(SI_STATE_COLUMNS):
        data[name] = y[:, i]
    return pd.DataFrame(data)


__all__ = ["SI_STATE_COLUMNS", "hourly_t_eval_for_startup_days", "trajectory_to_dataframe"]
