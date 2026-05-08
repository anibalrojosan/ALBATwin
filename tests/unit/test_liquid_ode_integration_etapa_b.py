"""Robustness, tolerance, max_step, and dense-output tests for ``integrate_liquid_ode`` (phase1-04d Etapa B)."""

from __future__ import annotations

import numpy as np

from bioprocess_twin.core import integrate_liquid_ode
from bioprocess_twin.core.state import StateVector
from bioprocess_twin.forcing import DielForcingSchedule
from bioprocess_twin.models.stoichiometry import N_STATE
from bioprocess_twin.simulator import LiquidOdeRhsProblem


def _stage6_state() -> StateVector:
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


def test_rtol_strict_vs_relaxed_final_states_are_close() -> None:
    """Loose vs tight tolerances should yield similar endpoints on a short horizon."""
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="summer"))
    t_span = (0.0, 2.0)
    te = np.linspace(*t_span, 5)

    tight = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=te,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.25,
    )
    loose = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=te,
        rtol=1e-3,
        atol=1e-5,
        max_step=1.0,
    )
    assert tight.success and loose.success
    diff = np.linalg.norm(tight.y[-1] - loose.y[-1])
    scale = max(np.linalg.norm(tight.y[-1]), 1.0)
    assert diff / scale < 0.05


def test_max_step_small_typically_more_rhs_evaluations_than_large() -> None:
    """Smaller max_step tends to increase RHS evaluations (nfev)."""
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="spring"))
    t_span = (0.0, 2.0)
    te = np.linspace(*t_span, 11)

    small = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=te,
        max_step=0.5,
        return_ode_result=True,
    )
    large = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=te,
        max_step=24.0,
        return_ode_result=True,
    )
    assert small.success and large.success
    assert small.ode_result is not None and large.ode_result is not None
    assert small.ode_result.nfev >= large.ode_result.nfev


def test_dense_output_sol_matches_span_endpoints() -> None:
    """Dense interpolant agrees with initial state and stored trajectory at boundaries."""
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="winter"))
    t0, t1 = 0.0, 2.0
    te = np.linspace(t0, t1, 5)

    res = integrate_liquid_ode(
        problem,
        y0,
        (t0, t1),
        t_eval=te,
        dense_output=True,
        return_ode_result=True,
        max_step=1.0,
    )
    assert res.success
    assert res.ode_result is not None
    assert res.ode_result.sol is not None
    np.testing.assert_allclose(res.ode_result.sol(t0), y0, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(res.ode_result.sol(t1), res.y[-1], rtol=1e-10, atol=1e-9)


def test_dense_output_false_leaves_sol_none_when_raw_returned() -> None:
    """Without dense_output, SciPy does not build an interpolant."""
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="autumn"))
    res = integrate_liquid_ode(
        problem,
        y0,
        (0.0, 1.0),
        t_eval=np.linspace(0.0, 1.0, 4),
        dense_output=False,
        return_ode_result=True,
    )
    assert res.success
    assert res.ode_result is not None
    assert res.ode_result.sol is None


def test_stability_moderate_horizon_finite() -> None:
    """No blow-up over a few hours with default LSODA."""
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="summer"))
    res = integrate_liquid_ode(
        problem,
        y0,
        (0.0, 6.0),
        t_eval=np.linspace(0.0, 6.0, 7),
        max_step=3.0,
    )
    assert res.success
    assert res.y.shape == (7, N_STATE)
    assert np.all(np.isfinite(res.y))
