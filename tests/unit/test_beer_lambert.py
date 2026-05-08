"""Unit tests for Beer–Lambert depth-averaged PAR and TSS proxy (phase1-04e)."""

from __future__ import annotations

import numpy as np
import pytest

from bioprocess_twin.core.state import StateVector
from bioprocess_twin.forcing import DielForcingSchedule
from bioprocess_twin.models.kinetic_parameters import default_alba
from bioprocess_twin.simulator import LiquidOdeRhsProblem, evaluate_liquid_ode_rhs
from bioprocess_twin.simulator.beer_lambert import (
    COD_TO_TSS_ALG,
    depth_averaged_irradiance_umol_m2_s,
    env_irradiance_umol_m2_s,
    tss_proxy_g_m3_from_state,
)


def test_tss_proxy_is_x_alg_over_cod_to_tss() -> None:
    st = StateVector(
        X_ALG=157.0,
        X_AOB=0.0,
        X_NOB=0.0,
        X_H=0.0,
        X_S=0.0,
        X_I=0.0,
        S_S=0.0,
        S_I=0.0,
        S_IC=0.0,
        S_ND=0.0,
        S_NH=0.0,
        S_NO2=0.0,
        S_NO3=0.0,
        S_N2=0.0,
        S_PO4=0.0,
        S_O2=0.0,
        S_H2O=0.0,
    )
    assert np.isclose(tss_proxy_g_m3_from_state(st), 157.0 / COD_TO_TSS_ALG)
    assert np.isclose(tss_proxy_g_m3_from_state(st.to_array()), 157.0 / COD_TO_TSS_ALG)


def test_depth_averaged_matches_surface_when_x_alg_zero() -> None:
    params = default_alba()
    i0 = 400.0
    out = depth_averaged_irradiance_umol_m2_s(i0, params.epsilon_light, 0.0, 0.35)
    assert np.isclose(out, i0)


def test_depth_averaged_positive_attenuation() -> None:
    params = default_alba()
    i0 = 300.0
    xa = 80.0
    h = 0.25
    eps = params.epsilon_light
    kappa = eps * xa
    kh = kappa * h
    expected = i0 * (-np.expm1(-kh)) / kh
    got = depth_averaged_irradiance_umol_m2_s(i0, eps, xa, h)
    assert np.isclose(got, expected)
    assert got < i0


def test_env_irradiance_none_passes_surface() -> None:
    got = env_irradiance_umol_m2_s(
        surface_irradiance_umol_m2_s=100.0,
        epsilon_cod_m2_per_g=0.067,
        x_alg_g_cod_m3=50.0,
        mixed_layer_depth_m=None,
    )
    assert np.isclose(got, 100.0)


def test_evaluate_liquid_ode_rhs_beer_lambert_changes_derivatives() -> None:
    """Same (t, y): mixed layer lowers effective I when X_ALG > 0 -> different dcdt."""
    st = StateVector(
        X_ALG=120.0,
        X_AOB=10.0,
        X_NOB=8.0,
        X_H=40.0,
        X_S=20.0,
        X_I=15.0,
        S_S=30.0,
        S_I=10.0,
        S_IC=25.0,
        S_ND=5.0,
        S_NH=8.0,
        S_NO2=0.5,
        S_NO3=4.0,
        S_N2=2.0,
        S_PO4=3.0,
        S_O2=5.0,
        S_H2O=0.0,
    )
    y = st.to_array()
    schedule = DielForcingSchedule(season="summer")
    t_hours = 14.0
    base = LiquidOdeRhsProblem(schedule=schedule)
    attenuated = LiquidOdeRhsProblem(schedule=schedule, mixed_layer_depth_m=0.3)
    d_base = evaluate_liquid_ode_rhs(t_hours, y, problem=base)
    d_att = evaluate_liquid_ode_rhs(t_hours, y, problem=attenuated)
    assert not np.allclose(d_base, d_att, rtol=0.0, atol=0.0)


def test_depth_negative_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        depth_averaged_irradiance_umol_m2_s(100.0, 0.067, 50.0, -0.1)
