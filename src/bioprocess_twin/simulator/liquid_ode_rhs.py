"""ODE-sized liquid RHS: Stage 6 with diel forcing (phase1-04a) and optional CSTR dilution (phase1-04c-B)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bioprocess_twin.core.state import StateVector, StateVectorVariant
from bioprocess_twin.forcing.diel_forcing_schedule import DielForcingSchedule, to_env_conditions
from bioprocess_twin.models.chemistry import AlbaDissociationConstantsRef, AlbaDissociationEnthalpy, PHSolverOptions
from bioprocess_twin.models.gas_transfer import GasTransferConditions
from bioprocess_twin.models.kinetic_parameters import KineticParameters
from bioprocess_twin.models.stoichiometry import N_STATE

from .cstr_transport import cstr_dilution_rate_g_m3_d
from .liquid_rhs import evaluate_liquid_rhs, state_vector_from_y


@dataclass(frozen=True, slots=True)
class CstrContinuousConfig:
    """
    Constant-parameter CSTR dilution for continuous-flow simulation.

    q_m3_per_d is volumetric inflow [m³ d⁻¹] (same outflow, constant volume);
    y_in is the SI influent state vector, same layout as StateVector / y.

    Time-varying Q(t) or y_in(t) is not handled here (phase1-04c-C).
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
class LiquidOdeRhsProblem:
    """
    Immutable bundle for evaluate_liquid_ode_rhs.

    Time: t_hours passed to the RHS is hours of day (clock time). Values outside
    [0,24) are reduced modulo 24 inside DielForcingSchedule.at (same convention as forcing).

    Optional cstr adds (Q/V)(y_in - y) per day, aligned with dcdt_g_m3_d.
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
    cstr: CstrContinuousConfig | None = None


def evaluate_liquid_ode_rhs(t_hours: float, y: np.ndarray, *, problem: LiquidOdeRhsProblem) -> np.ndarray:
    """
    SciPy-style vector field: dy/dt in g m⁻³ d⁻¹, shape (N_STATE,).

    Builds EnvConditions from problem.schedule.at(t_hours) and
    to_env_conditions(..., ph=problem.placeholder_ph_for_env). Solved pH inside
    evaluate_liquid_rhs still comes from SI.6 charge balance (initial_ph is only the
    solver guess).

    If problem.cstr is set, adds the constant-parameter CSTR dilution term
    (Q/V)(y_in - y) from cstr_dilution_rate_g_m3_d (same units).
    """
    st = state_vector_from_y(y)
    sample = problem.schedule.at(t_hours)
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
    if problem.cstr is not None:
        cfg = problem.cstr
        transport = cstr_dilution_rate_g_m3_d(
            y,
            cfg.y_in,
            volume_m3=cfg.volume_m3,
            q_m3_per_d=cfg.q_m3_per_d,
        )
        dcdt = dcdt + transport
    return dcdt


def make_liquid_rhs(problem: LiquidOdeRhsProblem) -> Callable[[float, np.ndarray], np.ndarray]:
    """Return rhs(t, y) closed over problem for use with solve_ivp."""

    def rhs(t_hours: float, y: np.ndarray) -> np.ndarray:
        return evaluate_liquid_ode_rhs(t_hours, y, problem=problem)

    return rhs
