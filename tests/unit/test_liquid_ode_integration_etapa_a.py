"""Short-horizon `integrate_liquid_ode` tests (phase1-04d Etapa A)."""

from __future__ import annotations

import numpy as np
import pytest

from bioprocess_twin.core import LiquidIntegrationResult, integrate_liquid_ode
from bioprocess_twin.core.state import StateVector
from bioprocess_twin.forcing import DielForcingSchedule
from bioprocess_twin.models.stoichiometry import N_STATE
from bioprocess_twin.simulator import LiquidOdeRhsProblem, evaluate_liquid_ode_rhs


def _stage6_state() -> StateVector:
    """Same representative state as ``test_liquid_ode_rhs``."""
    return StateVector(
        X_ALG=80.0,
        X_AOB=25.0,
        X_NOB=22.0,
        X_H=120.0,
        X_S=50.0,
        X_I=35.0,
        S_S=40.0,
        S_I=20.0,
        S_IC=35.0,
        S_ND=5.0,
        S_NH=12.0,
        S_NO2=1.2,
        S_NO3=5.5,
        S_N2=8.0,
        S_PO4=4.0,
        S_O2=7.0,
        S_H2O=0.0,
    )


def test_integrate_liquid_ode_shape_success_and_finite() -> None:
    """Test the shape of the integration result."""
    st = _stage6_state()
    y0 = st.to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="summer"))
    res = integrate_liquid_ode(
        problem,
        y0,
        (0.0, 1.0),
        t_eval=np.linspace(0.0, 1.0, 5),
    )
    assert isinstance(res, LiquidIntegrationResult)
    assert res.success
    assert res.t_hours.shape == (5,)
    assert res.y.shape == (5, N_STATE)
    assert np.all(np.isfinite(res.y))


def test_integrate_short_dt_matches_explicit_euler() -> None:
    """Nearly constant RHS over 0.01 h: one Euler step matches RK45 tightly."""
    st = _stage6_state()
    y0 = st.to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="spring"))
    dt_h = 0.01
    t0 = 0.0
    d_day = evaluate_liquid_ode_rhs(t0, y0, problem=problem)
    y_euler = y0 + (dt_h / 24.0) * d_day

    res = integrate_liquid_ode(
        problem,
        y0,
        (t0, t0 + dt_h),
        t_eval=np.array([t0, t0 + dt_h]),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.002,
    )
    assert res.success
    # Single-step Euler uses dydt only at (t0, y0); RK45 averages the field over substeps.
    np.testing.assert_allclose(res.y[-1], y_euler, rtol=2e-3, atol=1e-5)


def test_integrate_spans_multiple_days_clock_wrap() -> None:
    """Forcing wraps modulo 24 h; integration from 0–25 h should complete."""
    st = _stage6_state()
    y0 = st.to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="winter"))
    res = integrate_liquid_ode(
        problem,
        y0,
        (0.0, 25.0),
        t_eval=np.linspace(0.0, 25.0, 6),
        max_step=12.0,
    )
    assert res.success
    assert res.y.shape == (6, N_STATE)
    assert np.all(np.isfinite(res.y))


def test_bad_y0_length_raises() -> None:
    """Test that a bad initial state vector length raises an error."""
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="autumn"))
    with pytest.raises(ValueError, match="length"):
        integrate_liquid_ode(problem, np.zeros(3), (0.0, 1.0))


def test_bad_t_span_raises() -> None:
    """Test that a bad integration window raises an error."""
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="autumn"))
    with pytest.raises(ValueError, match="t_end"):
        integrate_liquid_ode(problem, _stage6_state().to_array(), (1.0, 0.0))
