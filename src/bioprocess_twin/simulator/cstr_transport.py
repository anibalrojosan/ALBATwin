"""
Pure CSTR dilution transport for the lumped liquid state (phase1-04c-A).

Vector transport term for a single perfectly mixed compartment with constant volume,
equal inflow and outflow rate:

    dC/dt_transport = (Q/V)(C_in - C)

See docs/theory/cstr_mass_balance_and_hrap_lumped_model.md (eq. 4.1 in-repo narrative).

Units are aligned with Stage 6 RHS output dcdt_g_m3_d: take Q in m³ d⁻¹
and V in m³ so Q/V is d⁻¹, making the transport contribution per day
(g m⁻³ d⁻¹ per component, consistent with each state's convention).

This module does not call evaluate_liquid_rhs yet.
"""

from __future__ import annotations

import numpy as np

from bioprocess_twin.models.stoichiometry import N_STATE


def q_m3_per_day_from_m3_per_hour(q_m3_per_h: float) -> float:
    """
    Convert volumetric flow from m³ h⁻¹ (e.g. ForcingSample.inflow_m3_h) to m³ d⁻¹.

    Uses 24 h per day (same convention as startup_batch time scaling).
    """
    if q_m3_per_h < 0:
        raise ValueError("q_m3_per_h must be non-negative")
    return float(q_m3_per_h) * 24.0


def hrt_days(volume_m3: float, q_m3_per_d: float) -> float:
    """Hydraulic retention time τ = V/Q [d], for q_m3_per_d > 0."""
    if volume_m3 <= 0:
        raise ValueError("volume_m3 must be positive")
    if q_m3_per_d <= 0:
        raise ValueError("q_m3_per_d must be positive for finite HRT")
    return float(volume_m3) / float(q_m3_per_d)


def q_m3_per_day_from_hrt_days(volume_m3: float, hrt_days: float) -> float:
    """Inflow rate Q = V/τ [m³ d⁻¹] from volume and HRT [d]."""
    if volume_m3 <= 0:
        raise ValueError("volume_m3 must be positive")
    if hrt_days <= 0:
        raise ValueError("hrt_days must be positive")
    return float(volume_m3) / float(hrt_days)


def cstr_dilution_rate_g_m3_d(
    y: np.ndarray,
    y_in: np.ndarray,
    *,
    volume_m3: float,
    q_m3_per_d: float,
) -> np.ndarray:
    """
    Transport-only contribution (Q/V)(y_in - y), per day, shape (N_STATE,).

    Parameters
    ----------
    y, y_in
        SI liquid state vectors (length N_STATE), same units as StateVector
        components (gCOD m⁻³, gN m⁻³, … per field).
    volume_m3
        Reactor liquid volume [m³].
    q_m3_per_d
        Volumetric inflow rate [m³ d⁻¹] (equal outflow, constant V).
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    yina = np.asarray(y_in, dtype=np.float64).ravel()
    if ya.size != N_STATE:
        raise ValueError(f"y must have length {N_STATE}, got {ya.size}")
    if yina.size != N_STATE:
        raise ValueError(f"y_in must have length {N_STATE}, got {yina.size}")
    if volume_m3 <= 0:
        raise ValueError("volume_m3 must be positive")
    if q_m3_per_d < 0:
        raise ValueError("q_m3_per_d must be non-negative")
    k = float(q_m3_per_d) / float(volume_m3)
    return k * (yina - ya)
