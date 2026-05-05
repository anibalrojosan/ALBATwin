"""Tests for ODE-sized liquid RHS wrapper (phase1-04b, phase1-04c-B CSTR)."""

from __future__ import annotations

import numpy as np
import pytest

from bioprocess_twin.core.state import StateVector
from bioprocess_twin.forcing import DielForcingSchedule, to_env_conditions
from bioprocess_twin.models.stoichiometry import N_STATE
from bioprocess_twin.simulator import (
    CstrContinuousConfig,
    LiquidOdeRhsProblem,
    cstr_dilution_rate_g_m3_d,
    evaluate_liquid_ode_rhs,
    evaluate_liquid_rhs,
    hrt_days,
    make_liquid_rhs,
    state_vector_from_y,
)


def _stage6_state() -> StateVector:
    """Same representative state as ``test_liquid_rhs_stage6`` (avoid cross-test imports)."""
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


def test_evaluate_liquid_ode_rhs_matches_direct_stage6() -> None:
    st = _stage6_state()
    y = st.to_array()
    t_hours = 10.5
    schedule = DielForcingSchedule(season="summer")
    problem = LiquidOdeRhsProblem(schedule=schedule)
    env = to_env_conditions(schedule.at(t_hours), ph=problem.placeholder_ph_for_env)
    direct = evaluate_liquid_rhs(st, env, initial_ph=problem.initial_ph)
    wrapped = evaluate_liquid_ode_rhs(t_hours, y, problem=problem)
    assert wrapped.shape == (N_STATE,)
    np.testing.assert_allclose(wrapped, direct.dcdt_g_m3_d, rtol=1e-12, atol=1e-12)


def test_evaluate_liquid_ode_rhs_finite_and_state_vector_from_y() -> None:
    st = _stage6_state()
    y = st.to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="autumn"))
    dydt = evaluate_liquid_ode_rhs(6.0, y, problem=problem)
    assert dydt.shape == (N_STATE,)
    assert np.all(np.isfinite(dydt))
    assert state_vector_from_y(y).to_array().shape == (N_STATE,)


def test_make_liquid_rhs_closure() -> None:
    st = _stage6_state()
    y = st.to_array()
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="winter"))
    rhs = make_liquid_rhs(problem)
    a = rhs(12.0, y)
    b = evaluate_liquid_ode_rhs(12.0, y, problem=problem)
    np.testing.assert_allclose(a, b)


def test_different_clock_hours_change_forcing_and_dcdt() -> None:
    """Day vs night irradiance (summer) should not yield identical derivatives."""
    st = _stage6_state()
    y = st.to_array()
    schedule = DielForcingSchedule(season="summer")
    problem = LiquidOdeRhsProblem(schedule=schedule)
    s_noon = schedule.at(14.0)
    s_night = schedule.at(22.0)
    assert s_noon.irradiance_umol_m2_s > s_night.irradiance_umol_m2_s
    d_noon = evaluate_liquid_ode_rhs(14.0, y, problem=problem)
    d_night = evaluate_liquid_ode_rhs(22.0, y, problem=problem)
    assert not np.allclose(d_noon, d_night, rtol=0.0, atol=0.0)


def test_wrong_state_length_raises() -> None:
    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season="spring"))
    with pytest.raises(ValueError, match="length"):
        evaluate_liquid_ode_rhs(0.0, np.zeros(5), problem=problem)


def test_cstr_q_zero_equivalent_to_no_cstr() -> None:
    """With q_m3_per_d=0, dilution vanishes; RHS matches the default (no cstr) problem."""
    st = _stage6_state()
    y = st.to_array()
    t_hours = 8.0
    schedule = DielForcingSchedule(season="summer")
    base = LiquidOdeRhsProblem(schedule=schedule)
    y_in = np.ones(N_STATE)
    cstr_zero = CstrContinuousConfig(volume_m3=10.0, q_m3_per_d=0.0, y_in=y_in)
    with_q0 = LiquidOdeRhsProblem(schedule=schedule, cstr=cstr_zero)
    np.testing.assert_allclose(
        evaluate_liquid_ode_rhs(t_hours, y, problem=base),
        evaluate_liquid_ode_rhs(t_hours, y, problem=with_q0),
        rtol=1e-12,
        atol=1e-12,
    )


def test_cstr_rhs_is_bio_plus_dilution_term() -> None:
    """Test that the CSTR RHS is the sum of the biological RHS and the dilution term."""
    st = _stage6_state()
    y = st.to_array()
    t_hours = 11.0
    schedule = DielForcingSchedule(season="autumn")
    vol = 17.0
    q = 3.4
    y_in = np.linspace(0.1, 2.0, N_STATE)
    cfg = CstrContinuousConfig(volume_m3=vol, q_m3_per_d=q, y_in=y_in)
    no_cstr = LiquidOdeRhsProblem(schedule=schedule)
    with_cstr = LiquidOdeRhsProblem(schedule=schedule, cstr=cfg)
    bio = evaluate_liquid_ode_rhs(t_hours, y, problem=no_cstr)
    full = evaluate_liquid_ode_rhs(t_hours, y, problem=with_cstr)
    dilution = cstr_dilution_rate_g_m3_d(y, cfg.y_in, volume_m3=vol, q_m3_per_d=q)
    np.testing.assert_allclose(full, bio + dilution, rtol=1e-12, atol=1e-12)


def test_hrt_days_volume_over_flow() -> None:
    """Numeric check: V/Q coherence (example 17 m³ / 3.4 m³ d⁻¹ = 5 d)."""
    assert hrt_days(17.0, 3.4) == pytest.approx(5.0)


def test_cstr_from_influent_matches_si_array() -> None:
    """Test that the CSTR configuration from influent matches the StateVector array."""
    st = _stage6_state()
    arr = st.to_array()
    a = CstrContinuousConfig.from_influent(5.0, 1.0, st)
    b = CstrContinuousConfig(5.0, 1.0, arr)
    np.testing.assert_allclose(a.y_in, b.y_in, rtol=1e-12, atol=1e-12)
