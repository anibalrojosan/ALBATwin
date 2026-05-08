"""Beer–Lambert light attenuation and depth-averaged PAR for the lumped pond (phase1-04e).

Si Casagli et al. (2021): SI 1.1 (extinction vs depth), SI 1.2 (COD ↔ dry algae mass).

Surface PAR I_0(t) comes from DielForcingSchedule / ForcingSample.
Extinction uses KineticParameters.epsilon_light (epsilon_cod in m² gCOD⁻¹,
MATH_MODEL.md §1.2.2) multiplied by algal COD concentration X_ALG [gCOD m⁻³]
to give an attenuation coefficient kappa [m⁻¹], equivalent to using TSS with
varepsilon_tss = epsilon_cod * 1.57 and SI 1.2 TSS proxy.
"""

from __future__ import annotations

import numpy as np

from bioprocess_twin.core.state import StateVector
from bioprocess_twin.models.stoichiometry import N_STATE

COD_TO_TSS_ALG = 1.57
"""g COD per g dry algal biomass (SI 1.2); TSS approx X_ALG/COD_TO_TSS_ALG [g m⁻³]."""


def tss_proxy_g_m3_from_state(y: StateVector | np.ndarray) -> float:
    """
    Algal-suspension TSS proxy [g m⁻³] from liquid state (SI 1.2).

    - Option 1 (implemented): Uses only algal biomass COD:

    TSS_proxy = X_ALG / COD_TO_TSS_ALG with X_ALG in gCOD m⁻³.

    - Option 2 (not implemented): Multi-particulate TSS as a sum over modeled solids
    (X_ALG, X_H, X_I, …) with documented COD → dry-mass factors per fraction where
    literature supports it; improves realism when non-algal suspended solids dominate.

    - Option 3 (not implemented): Prescribed TSS — constant calibration value, campaign
    average, CSV time series, or external sensor feed — for experiments or when the
    algal-only proxy is inadequate.

    Parameters
    ----------
    y
        StateVector or length-N_STATE SI array; X_ALG is index 0.

    Returns
    -------
    float
        Non-negative TSS proxy [g m⁻³].
    """
    if isinstance(y, StateVector):
        x_alg = float(y.X_ALG)
    else:
        arr = np.asarray(y, dtype=np.float64).ravel()
        if arr.size != N_STATE:
            raise ValueError(f"expected length {N_STATE}, got {arr.size}")
        x_alg = float(arr[0])
    x_alg = max(x_alg, 0.0)
    return x_alg / COD_TO_TSS_ALG


def depth_averaged_irradiance_umol_m2_s(
    surface_irradiance_umol_m2_s: float,
    epsilon_cod_m2_per_g: float,
    x_alg_g_cod_m3: float,
    mixed_layer_depth_m: float,
) -> float:
    """
    Depth-averaged PAR [µmol m⁻² s⁻¹] over a vertically well-mixed layer of depth h.

    With kappa = epsilon_cod * X_ALG [m⁻¹], depth-averaged PAR is: 

              averaged_I = I0 * (1 - exp(-kappa*h)) / (kappa*h)
    
    For vanishing kappa*h, the limit is I0. Non-positive mixed_layer_depth_m raises.

    Parameters
    ----------
    surface_irradiance_umol_m2_s
        I0 at the surface (same units as EnvConditions.irradiance_umol_m2_s).
    epsilon_cod_m2_per_g
        epsilon_cod [m² gCOD⁻¹] from kinetics (epsilon_light).
    x_alg_g_cod_m3
        Algal COD concentration [gCOD m⁻³].
    mixed_layer_depth_m
        Mixed-layer thickness $h$ [m], positive.

    Returns
    -------
    float
        averaged_I used as the scalar irradiance input to kinetics.
    """
    if mixed_layer_depth_m <= 0.0:
        raise ValueError(f"mixed_layer_depth_m must be positive, got {mixed_layer_depth_m}")
    i0 = max(float(surface_irradiance_umol_m2_s), 0.0)
    eps = float(epsilon_cod_m2_per_g)
    xa = max(float(x_alg_g_cod_m3), 0.0)
    h = float(mixed_layer_depth_m)
    kappa = eps * xa
    kh = kappa * h
    if kh <= 1e-18:
        return float(i0)
    # (1 - exp(-kh)) / kh = -expm1(-kh) / kh
    return float(i0 * (-np.expm1(-kh)) / kh)


def env_irradiance_umol_m2_s(
    *,
    surface_irradiance_umol_m2_s: float,
    epsilon_cod_m2_per_g: float,
    x_alg_g_cod_m3: float,
    mixed_layer_depth_m: float | None,
) -> float:
    """
    Irradiance passed to EnvConditions: surface value if mixed_layer_depth_m is
    None, otherwise depth-averaged Beer–Lambert effective PAR.
    """
    if mixed_layer_depth_m is None:
        return max(float(surface_irradiance_umol_m2_s), 0.0)
    return depth_averaged_irradiance_umol_m2_s(
        surface_irradiance_umol_m2_s,
        epsilon_cod_m2_per_g,
        x_alg_g_cod_m3,
        mixed_layer_depth_m,
    )


__all__ = [
    "COD_TO_TSS_ALG",
    "depth_averaged_irradiance_umol_m2_s",
    "env_irradiance_umol_m2_s",
    "tss_proxy_g_m3_from_state",
]
