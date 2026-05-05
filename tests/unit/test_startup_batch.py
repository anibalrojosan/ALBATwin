"""Tests for batch startup integration with evaporation (SI 17 + volume)."""

from __future__ import annotations

import numpy as np
import pytest

from bioprocess_twin.core.state import StateVector, StateVectorVariant
from bioprocess_twin.forcing import DielForcingSchedule
from bioprocess_twin.forcing.diel_forcing_schedule import ForcingSample
from bioprocess_twin.models.stoichiometry import N_STATE
from bioprocess_twin.simulator import LiquidOdeRhsProblem, evaluate_liquid_ode_rhs
from bioprocess_twin.simulator.startup_batch import (
    StartupBatchProblem,
    default_reasonable_startup_y0,
    evaluate_startup_batch_rhs,
    evaporation_rate_m3_h,
    rain_inflow_m3_h,
    run_startup_batch,
    volume_derivative_m3_h,
)


def _rich_state() -> np.ndarray:
    return default_reasonable_startup_y0()


def test_evaluate_startup_rhs_matches_base_when_dVdt_zero() -> None:
    """With E=0 and no rain, augmented derivative matches ALBA / 24 on y, dV=0."""
    y = _rich_state()
    v0 = 17.0
    z = np.concatenate([y, np.array([v0])])
    schedule = DielForcingSchedule(
        season="summer",
        evaporation_m3_h=0.0,
        rain_mm_h=0.0,
    )
    problem = LiquidOdeRhsProblem(schedule=schedule)
    cfg = StartupBatchProblem(
        problem=problem,
        startup_days=0.01,
        y0=y,
        volume_m3_initial=v0,
        surface_area_m2=56.0,
        include_rain=True,
    )
    t = 10.5
    out = evaluate_startup_batch_rhs(t, z, problem_cfg=cfg)
    base = evaluate_liquid_ode_rhs(t, y, problem=problem) / 24.0
    assert out.shape == (N_STATE + 1,)
    np.testing.assert_allclose(out[:N_STATE], base, rtol=1e-12, atol=1e-12)
    assert out[-1] == pytest.approx(0.0)


def test_concentration_term_for_evaporation_scales_with_C_over_V() -> None:
    y = np.ones(N_STATE) * 2.0
    v0 = 10.0
    z = np.concatenate([y, np.array([v0])])
    schedule = DielForcingSchedule(
        season="summer",
        evaporation_m3_h=0.0,
        rain_mm_h=0.0,
    )
    problem = LiquidOdeRhsProblem(schedule=schedule)
    cfg = StartupBatchProblem(
        problem=problem,
        startup_days=0.01,
        y0=y,
        volume_m3_initial=v0,
        surface_area_m2=56.0,
    )
    d0 = volume_derivative_m3_h(
        schedule.at(0.0),
        cfg.surface_area_m2,
        include_rain=cfg.include_rain,
        evaporation_floor_m3_h=0.0,
    )
    assert d0 == pytest.approx(0.0)
    # Force net outflow: override evaporation to a positive constant
    sched2 = DielForcingSchedule(
        season="summer",
        evaporation_m3_h=0.1,
        rain_mm_h=0.0,
    )
    p2 = LiquidOdeRhsProblem(schedule=sched2)
    cfg2 = StartupBatchProblem(
        problem=p2,
        startup_days=0.01,
        y0=y,
        volume_m3_initial=v0,
        surface_area_m2=56.0,
    )
    out2 = evaluate_startup_batch_rhs(0.0, z, problem_cfg=cfg2)
    dV = out2[-1]
    assert dV == pytest.approx(-0.1)
    # r=0 for y=ones would not hold; compare only extra term: -(y/V)*dV
    base2 = evaluate_liquid_ode_rhs(0.0, y, problem=p2) / 24.0
    expected_extra = -(y / v0) * dV
    np.testing.assert_allclose(out2[:N_STATE] - base2, expected_extra, rtol=1e-10, atol=1e-10)


def test_run_startup_batch_shape_and_finitude() -> None:
    y = _rich_state()
    schedule = DielForcingSchedule(season="autumn", evaporation_m3_h=0.01, rain_mm_h=0.0)
    problem = LiquidOdeRhsProblem(schedule=schedule)
    cfg = StartupBatchProblem(
        problem=problem,
        startup_days=0.5,
        y0=y,
        volume_m3_initial=17.0,
        surface_area_m2=56.0,
        include_rain=False,
    )
    res = run_startup_batch(cfg, t_eval_hours=np.linspace(0.0, 12.0, 5))
    assert res.y.shape[1] == N_STATE
    assert res.volume_m3.shape[0] == res.t_hours.shape[0]
    assert np.all(np.isfinite(res.y))
    assert np.all(np.isfinite(res.volume_m3))
    assert res.y.shape[0] == 5


def test_volume_decreases_with_evaporation_no_rain() -> None:
    y = _rich_state()
    schedule = DielForcingSchedule(
        season="summer",
        evaporation_m3_h=0.02,
        rain_mm_h=0.0,
    )
    problem = LiquidOdeRhsProblem(schedule=schedule)
    v0 = 100.0
    cfg = StartupBatchProblem(
        problem=problem,
        startup_days=1.0,
        y0=y,
        volume_m3_initial=v0,
        surface_area_m2=56.0,
        include_rain=False,
    )
    res = run_startup_batch(cfg, t_eval_hours=np.array([0.0, 6.0, 12.0]))
    assert res.volume_m3[-1] < res.volume_m3[0]


def test_evaporation_floor_increases_loss_when_schedule_zero() -> None:
    sample = ForcingSample(
        t_hours=0.0,
        temperature_C=20.0,
        irradiance_umol_m2_s=0.0,
        evaporation_m3_h=0.0,
        rain_mm_h=None,
    )
    assert evaporation_rate_m3_h(sample, evaporation_floor_m3_h=1e-6) == pytest.approx(1e-6)
    s2 = ForcingSample(
        t_hours=0.0,
        temperature_C=20.0,
        irradiance_umol_m2_s=0.0,
        evaporation_m3_h=0.05,
        rain_mm_h=None,
    )
    assert evaporation_rate_m3_h(s2, evaporation_floor_m3_h=1e-6) == pytest.approx(0.05)


def test_rain_inflow_mm_to_m3() -> None:
    sample = ForcingSample(
        t_hours=0.0,
        temperature_C=20.0,
        irradiance_umol_m2_s=0.0,
        evaporation_m3_h=0.0,
        rain_mm_h=2.0,
    )
    q = rain_inflow_m3_h(sample, surface_area_m2=1000.0, include_rain=True)
    assert q == pytest.approx(2.0)


def test_default_reasonable_startup_y0_length_and_nonneg() -> None:
    y = default_reasonable_startup_y0()
    assert y.shape == (N_STATE,)
    assert np.all(y >= 0.0)
    StateVector.from_array(y, variant=StateVectorVariant.SI)
