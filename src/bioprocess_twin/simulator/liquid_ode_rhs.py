"""ODE-sized liquid RHS: Stage 6 with diel forcing (phase1-04a) and optional CSTR dilution (04c-B, 04c-C)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from bioprocess_twin.core.state import StateVector, StateVectorVariant
from bioprocess_twin.forcing.diel_forcing_schedule import (
    DielForcingSchedule,
    ForcingSample,
    to_env_conditions,
)
from bioprocess_twin.models.chemistry import AlbaDissociationConstantsRef, AlbaDissociationEnthalpy, PHSolverOptions
from bioprocess_twin.models.gas_transfer import GasTransferConditions
from bioprocess_twin.models.kinetic_parameters import KineticParameters, default_alba
from bioprocess_twin.models.kinetics import EnvConditions
from bioprocess_twin.models.stoichiometry import N_STATE

from .beer_lambert import env_irradiance_umol_m2_s
from .cstr_transport import cstr_dilution_rate_g_m3_d, q_m3_per_day_from_m3_per_hour
from .liquid_rhs import evaluate_liquid_rhs, state_vector_from_y

_CSTR_CLOCK_WRAP = 24.0


def _wrap_clock_hours(t_hours: float) -> float:
    """Same wrapping convention as DielForcingSchedule.at (daily repeating clock)."""
    return float(np.mod(t_hours, _CSTR_CLOCK_WRAP))


def q_m3_per_d_from_forcing_sample(sample: ForcingSample) -> float:
    """
    Volumetric inflow [m³ d⁻¹] from ``ForcingSample.inflow_m3_h``.

    If ``inflow_m3_h`` is missing, returns 0 (no dilution), matching constant CSTR with Q=0.
    """
    q_h = sample.inflow_m3_h
    if q_h is None:
        return 0.0
    return q_m3_per_day_from_m3_per_hour(float(q_h))


def _normalize_y_in_vector(raw: StateVector | np.ndarray) -> np.ndarray:
    """Normalize the influent state vector to the SI layout."""
    if isinstance(raw, StateVector):
        arr = raw.to_array(variant=StateVectorVariant.SI)
    else:
        arr = np.asarray(raw, dtype=np.float64).ravel()
    if arr.size != N_STATE:
        raise ValueError(f"cstr y_in must have length {N_STATE}, got {arr.size}")
    return arr


YInSchedule: TypeAlias = np.ndarray | StateVector | Callable[[float], np.ndarray | StateVector]


@dataclass(frozen=True, slots=True)
class CstrContinuousConfig:
    """
    Constant-parameter CSTR dilution for continuous-flow simulation.

    q_m3_per_d is volumetric inflow [m³ d⁻¹] (same outflow, constant volume);
    y_in is the SI influent state vector, same layout as StateVector / y.

    For flow taken from ``DielForcingSchedule.inflow_m3_h``, use ``CstrScheduleFlowConfig``.
    """

    volume_m3: float
    q_m3_per_d: float
    y_in: np.ndarray

    def __post_init__(self) -> None:
        if self.volume_m3 <= 0:
            raise ValueError("cstr volume_m3 must be positive")
        if self.q_m3_per_d < 0:
            raise ValueError("cstr q_m3_per_d must be non-negative")
        arr = np.asarray(self.y_in, dtype=np.float64).ravel()
        if arr.size != N_STATE:
            raise ValueError(f"cstr y_in must have length {N_STATE}, got {arr.size}")
        object.__setattr__(self, "y_in", arr.copy())

    @classmethod
    def from_influent(
        cls,
        volume_m3: float,
        q_m3_per_d: float,
        y_in: StateVector | np.ndarray,
    ) -> CstrContinuousConfig:
        """Build from a StateVector or a length N_STATE array."""
        if isinstance(y_in, StateVector):
            arr = y_in.to_array(variant=StateVectorVariant.SI)
        else:
            arr = np.asarray(y_in, dtype=np.float64).ravel()
        return cls(volume_m3=volume_m3, q_m3_per_d=q_m3_per_d, y_in=arr)


@dataclass(frozen=True, slots=True)
class CstrScheduleFlowConfig:
    """
    CSTR dilution with volumetric inflow from problem.schedule.at(t).inflow_m3_h.

    inflow_m3_h [m³ h⁻¹] is converted to [m³ d⁻¹] via × 24 (same as cstr_transport).
    If the sample has inflow_m3_h is None, dilution is zero for that instant.

    y_in may be a constant SI vector or callable(t_wrapped) where t_wrapped is
    in [0, 24) (daily repeating clock; same wrapping as DielForcingSchedule.at).
    """

    volume_m3: float
    y_in: YInSchedule

    def __post_init__(self) -> None:
        if self.volume_m3 <= 0:
            raise ValueError("cstr volume_m3 must be positive")
        if callable(self.y_in):
            return
        arr = _normalize_y_in_vector(self.y_in)
        object.__setattr__(self, "y_in", arr)


def effective_y_in_schedule(cfg: CstrScheduleFlowConfig, t_hours: float) -> np.ndarray:
    """Resolve y_in at clock time t_hours (wrapped to [0, 24) for callables)."""
    tw = _wrap_clock_hours(t_hours)
    spec = cfg.y_in
    if callable(spec):
        return _normalize_y_in_vector(spec(tw)).copy()
    assert isinstance(spec, np.ndarray)
    return np.asarray(spec, dtype=np.float64).ravel().copy()


CstrTransportConfig: TypeAlias = CstrContinuousConfig | CstrScheduleFlowConfig


@dataclass(frozen=True, slots=True)
class LiquidOdeRhsProblem:
    """
    Immutable bundle for evaluate_liquid_ode_rhs.

    Time: t_hours passed to the RHS is hours of day (clock time). Values outside
    [0,24) are reduced modulo 24 inside DielForcingSchedule.at (same convention as forcing).

    Optional cstr adds (Q/V)(y_in - y) per day, aligned with dcdt_g_m3_d.
    Use CstrContinuousConfig for constant Q, or CstrScheduleFlowConfig so Q
    follows schedule inflow_m3_h (phase1-04c-C).

    Optional mixed_layer_depth_m [m]: if set, EnvConditions.irradiance_umol_m2_s
    uses depth-averaged Beer–Lambert PAR (phase1-04e); if None, surface PAR from
    the schedule is unchanged.
    """

    schedule: DielForcingSchedule
    initial_ph: float = 7.0
    placeholder_ph_for_env: float = 7.0
    kinetic_parameters: KineticParameters | None = None
    theta_kla: float | None = None
    gas_conditions: GasTransferConditions | None = None
    delta_cat_an_mol_per_m3: float = 0.0
    options: PHSolverOptions | None = None
    k_ref: AlbaDissociationConstantsRef | None = None
    dh: AlbaDissociationEnthalpy | None = None
    cstr: CstrTransportConfig | None = None
    mixed_layer_depth_m: float | None = None


def evaluate_liquid_ode_rhs(t_hours: float, y: np.ndarray, *, problem: LiquidOdeRhsProblem) -> np.ndarray:
    """
    SciPy-style vector field: dy/dt in g m⁻³ d⁻¹, shape (N_STATE,).

    Builds EnvConditions from problem.schedule.at(t_hours) and
    to_env_conditions(..., ph=problem.placeholder_ph_for_env). Solved pH inside
    evaluate_liquid_rhs still comes from SI.6 charge balance (initial_ph is only the
    solver guess).

    If problem.cstr is a CstrContinuousConfig, adds constant-Q dilution.

    If it is a CstrScheduleFlowConfig, Q comes from problem.schedule.at(t_hours)
    (inflow_m3_h → m³ d⁻¹); y_in may vary with wrapped clock time for callables.

    If problem.mixed_layer_depth_m is set, EnvConditions.irradiance_umol_m2_s
    uses depth-averaged Beer–Lambert PAR from surface $I_0(t)$ and $X_{ALG}$
    (phase1-04e); otherwise surface irradiance from the schedule is used unchanged.
    """
    st = state_vector_from_y(y)
    sample = problem.schedule.at(t_hours)
    params = problem.kinetic_parameters if problem.kinetic_parameters is not None else default_alba()
    if problem.mixed_layer_depth_m is not None:
        i_env = env_irradiance_umol_m2_s(
            surface_irradiance_umol_m2_s=float(sample.irradiance_umol_m2_s),
            epsilon_cod_m2_per_g=float(params.epsilon_light),
            x_alg_g_cod_m3=float(st.X_ALG),
            mixed_layer_depth_m=float(problem.mixed_layer_depth_m),
        )
        env = EnvConditions(
            temperature_C=float(sample.temperature_C),
            pH=float(problem.placeholder_ph_for_env),
            irradiance_umol_m2_s=float(i_env),
        )
    else:
        env = to_env_conditions(sample, ph=problem.placeholder_ph_for_env)
    out = evaluate_liquid_rhs(
        st,
        env,
        kinetic_parameters=problem.kinetic_parameters,
        theta_kla=problem.theta_kla,
        gas_conditions=problem.gas_conditions,
        delta_cat_an_mol_per_m3=problem.delta_cat_an_mol_per_m3,
        initial_ph=problem.initial_ph,
        options=problem.options,
        k_ref=problem.k_ref,
        dh=problem.dh,
    )
    dcdt = np.asarray(out.dcdt_g_m3_d, dtype=np.float64).ravel()
    cfg = problem.cstr
    if cfg is None:
        return dcdt
    if isinstance(cfg, CstrContinuousConfig):
        transport = cstr_dilution_rate_g_m3_d(
            y,
            cfg.y_in,
            volume_m3=cfg.volume_m3,
            q_m3_per_d=cfg.q_m3_per_d,
        )
        return dcdt + transport
    if isinstance(cfg, CstrScheduleFlowConfig):
        q_d = q_m3_per_d_from_forcing_sample(sample)
        y_eff = effective_y_in_schedule(cfg, t_hours)
        transport = cstr_dilution_rate_g_m3_d(
            y,
            y_eff,
            volume_m3=cfg.volume_m3,
            q_m3_per_d=q_d,
        )
        return dcdt + transport
    raise TypeError(f"unsupported cstr config type: {type(cfg)!r}")


def make_liquid_rhs(problem: LiquidOdeRhsProblem) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return rhs(t, y) closed over problem for use with solve_ivp."""

    def rhs(t_hours: float, y: np.ndarray) -> np.ndarray:
        return evaluate_liquid_ode_rhs(t_hours, y, problem=problem)

    return rhs
