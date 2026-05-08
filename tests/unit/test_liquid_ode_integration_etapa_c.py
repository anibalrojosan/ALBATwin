"""Etapa C: output density, dense interpolation helper, first_step (phase1-04d)."""

from __future__ import annotations

import numpy as np
import pytest

from bioprocess_twin.core.simulation import integrate_liquid_ode, interpolate_liquid_trajectory
from bioprocess_twin.core.state import StateVector
from bioprocess_twin.forcing import DielForcingSchedule
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


def test_t_eval_coarse_vs_fine_same_endpoint_state() -> None:
    """Stored trajectory density must not change the terminal state for fixed solver settings."""
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="summer"))
    t_span = (0.0, 3.0)
    kwargs = dict(
        rtol=1e-8,
        atol=1e-10,
        max_step=1.0,
        method="LSODA",
    )
    ref = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=np.array([t_span[0], t_span[1]]),
        **kwargs,
    )
    coarse = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=np.linspace(*t_span, 5),
        **kwargs,
    )
    fine = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=np.linspace(*t_span, 40),
        **kwargs,
    )
    assert ref.success and coarse.success and fine.success
    np.testing.assert_allclose(coarse.y[-1], ref.y[-1], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fine.y[-1], ref.y[-1], rtol=1e-9, atol=1e-10)


def test_interpolate_liquid_trajectory_matches_scipy_sol() -> None:
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="spring"))
    t_span = (0.0, 2.0)
    te = np.linspace(*t_span, 5)
    res = integrate_liquid_ode(
        problem,
        y0,
        t_span,
        t_eval=te,
        dense_output=True,
        return_ode_result=True,
        max_step=0.75,
    )
    assert res.success and res.ode_result is not None and res.ode_result.sol is not None
    t_mid = 0.73
    if t_mid not in te:
        via_helper = interpolate_liquid_trajectory(res, np.array([t_mid]))
        direct = res.ode_result.sol(t_mid)
        np.testing.assert_allclose(via_helper[0], np.asarray(direct).ravel(), rtol=1e-12, atol=1e-12)


def test_interpolate_raises_without_dense_or_raw() -> None:
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="winter"))
    res = integrate_liquid_ode(
        problem,
        y0,
        (0.0, 1.0),
        t_eval=np.linspace(0.0, 1.0, 4),
        dense_output=False,
        return_ode_result=True,
    )
    assert res.success
    with pytest.raises(ValueError, match="dense_output"):
        interpolate_liquid_trajectory(res, np.array([0.5]))

    res2 = integrate_liquid_ode(problem, y0, (0.0, 1.0), t_eval=np.linspace(0.0, 1.0, 4))
    with pytest.raises(ValueError, match="return_ode_result"):
        interpolate_liquid_trajectory(res2, np.array([0.5]))


def test_interpolate_rejects_out_of_domain_times() -> None:
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="autumn"))
    res = integrate_liquid_ode(
        problem,
        y0,
        (1.0, 4.0),
        t_eval=np.linspace(1.0, 4.0, 5),
        dense_output=True,
        return_ode_result=True,
    )
    assert res.success
    with pytest.raises(ValueError, match="query times"):
        interpolate_liquid_trajectory(res, np.array([0.0]))
    with pytest.raises(ValueError, match="query times"):
        interpolate_liquid_trajectory(res, np.array([5.0]))


def test_first_step_smoke() -> None:
    y0 = _stage6_state().to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="summer"))
    res = integrate_liquid_ode(
        problem,
        y0,
        (0.0, 1.5),
        t_eval=np.linspace(0.0, 1.5, 4),
        first_step=0.05,
        max_step=0.5,
    )
    assert res.success
    assert np.all(np.isfinite(res.y))
