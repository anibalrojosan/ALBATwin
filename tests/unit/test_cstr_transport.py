"""Unit tests for pure CSTR dilution algebra (phase1-04c-A)."""

from __future__ import annotations

import numpy as np
import pytest

from bioprocess_twin.models.stoichiometry import N_STATE
from bioprocess_twin.simulator.cstr_transport import (
    cstr_dilution_rate_g_m3_d,
    hrt_days,
    q_m3_per_day_from_hrt_days,
    q_m3_per_day_from_m3_per_hour,
)


def test_q_zero_gives_zero_transport() -> None:
    """Test that when Q = 0, the transport rate is zero."""
    y = np.arange(N_STATE, dtype=np.float64)
    y_in = np.arange(N_STATE, dtype=np.float64) + 10.0
    out = cstr_dilution_rate_g_m3_d(y, y_in, volume_m3=17.0, q_m3_per_d=0.0)
    assert out.shape == (N_STATE,)
    np.testing.assert_allclose(out, 0.0, atol=0.0)


def test_y_equals_y_in_gives_zero() -> None:
    """Test that when y = y_in, the transport rate is zero."""
    y = np.linspace(1.0, 2.0, N_STATE)
    out = cstr_dilution_rate_g_m3_d(y, y.copy(), volume_m3=10.0, q_m3_per_d=5.0)
    np.testing.assert_allclose(out, 0.0, atol=0.0)


def test_shape_validation() -> None:
    """Test that the function raises an error if the input arrays are not the correct shape."""
    y_ok = np.zeros(N_STATE)
    y_bad = np.zeros(N_STATE - 1)
    with pytest.raises(ValueError, match="y must have length"):
        cstr_dilution_rate_g_m3_d(y_bad, y_ok, volume_m3=1.0, q_m3_per_d=1.0)
    with pytest.raises(ValueError, match="y_in must have length"):
        cstr_dilution_rate_g_m3_d(y_ok, y_bad, volume_m3=1.0, q_m3_per_d=1.0)


def test_volume_and_q_validation() -> None:
    y = np.zeros(N_STATE)
    with pytest.raises(ValueError, match="volume_m3"):
        cstr_dilution_rate_g_m3_d(y, y, volume_m3=0.0, q_m3_per_d=1.0)
    with pytest.raises(ValueError, match="q_m3_per_d"):
        cstr_dilution_rate_g_m3_d(y, y, volume_m3=10.0, q_m3_per_d=-1.0)


def test_hrt_and_round_trip_casagli_order() -> None:
    """Test that the function returns the correct HRT and that the round trip conversion is correct."""
    v = 17.0
    q = 3.4
    tau = hrt_days(v, q)
    assert tau == pytest.approx(5.0)
    q_back = q_m3_per_day_from_hrt_days(v, tau)
    assert q_back == pytest.approx(q)
    assert q_m3_per_day_from_m3_per_hour(q / 24.0) == pytest.approx(q)


def test_q_m3_per_hour_conversion() -> None:
    """Test that the function returns the correct Q in m³ d⁻¹ for a given Q in m³ h⁻¹."""
    assert q_m3_per_day_from_m3_per_hour(1.0) == pytest.approx(24.0)
    with pytest.raises(ValueError):
        q_m3_per_day_from_m3_per_hour(-0.1)


def test_tracer_first_component_hand_check() -> None:
    """Single nonzero concentration in first pool; verify -(Q/V)*c."""
    y = np.zeros(N_STATE)
    c = 100.0
    y[0] = c
    y_in = np.zeros(N_STATE)
    v = 17.0
    q = 3.4
    out = cstr_dilution_rate_g_m3_d(y, y_in, volume_m3=v, q_m3_per_d=q)
    expected_first = -(q / v) * c
    assert out[0] == pytest.approx(expected_first)
    np.testing.assert_allclose(out[1:], 0.0, atol=0.0)


def test_hrt_requires_positive_q() -> None:
    """Test that the function raises an error if Q is negative."""
    with pytest.raises(ValueError, match="q_m3_per_d"):
        hrt_days(17.0, 0.0)
