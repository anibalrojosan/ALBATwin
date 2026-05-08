"""
Batch startup integration (SI 17-state ALBA) with open-pond water balance.

Couples evaluate_liquid_ode_rhs (rates in g m⁻³ d⁻¹) to volume change from
evaporation and optional rain on surface_area_m2. Independent variable is
hours; biological rates are converted to per-hour (/24).

Volume ODE::

    dV/dt = Q_rain(t) - E(t)

Concentration correction for pure-water loss/gain (well-mixed lump, see
docs/theory/cstr_mass_balance_and_hrap_lumped_model.md §3.1)::

    dC_i/dt = r_i - (C_i/V) dV/dt

with r_i from ALBA in g m⁻³ d⁻¹, converted to g m⁻³ h⁻¹.

Only 17-component SI layout; proton closure / 18th state is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, overload

import numpy as np
from scipy.integrate import solve_ivp

from bioprocess_twin.core.state import StateVector, StateVectorVariant
from bioprocess_twin.forcing.diel_forcing_schedule import ForcingSample
from bioprocess_twin.models.stoichiometry import N_STATE

from .liquid_ode_rhs import LiquidOdeRhsProblem, evaluate_liquid_ode_rhs

# Augmented state: 17 ALBA components + volume [m³]
_STARTUP_N = N_STATE + 1
_VOLUME_INDEX = N_STATE

# Biological RHS from evaluate_liquid_ode_rhs is per day; integration uses hours.
_HOURS_PER_DAY = 24.0


def _as_length17_ndarray(y0: StateVector | np.ndarray) -> np.ndarray:
    """
    Convert StateVector to numpy array of length 17.
    """
    if isinstance(y0, StateVector):
        arr = y0.to_array(variant=StateVectorVariant.SI)
    else:
        arr = np.asarray(y0, dtype=np.float64).ravel()
    if arr.size != N_STATE:
        raise ValueError(f"startup y0 must have length {N_STATE} (SI layout), got {arr.size}")
    return arr


def rain_inflow_m3_h(sample: ForcingSample, surface_area_m2: float, *, include_rain: bool) -> float:
    """
    Convert mm h⁻¹ rain depth over surface_area_m2 to m³ h⁻¹ inflow on the pond surface.

    If include_rain is False or rain_mm_h is missing, returns 0.
    """
    if not include_rain:
        return 0.0
    if sample.rain_mm_h is None:
        return 0.0
    return float(sample.rain_mm_h) * float(surface_area_m2) / 1000.0


def evaporation_rate_m3_h(sample: ForcingSample, evaporation_floor_m3_h: float) -> float:
    """Pond-scale evaporation rate [m³ h⁻¹] with optional numerical floor."""
    return max(float(sample.evaporation_m3_h), float(evaporation_floor_m3_h))


def volume_derivative_m3_h(
    sample: ForcingSample,
    surface_area_m2: float,
    *,
    include_rain: bool,
    evaporation_floor_m3_h: float,
) -> float:
    """Net volumetric rate dV/dt [m³ h⁻¹] (rain minus evaporation)."""
    q_rain = rain_inflow_m3_h(sample, surface_area_m2, include_rain=include_rain)
    e = evaporation_rate_m3_h(sample, evaporation_floor_m3_h)
    return q_rain - e


@dataclass(frozen=True, slots=True)
class StartupBatchProblem:
    """
    Configuration for run_startup_batch.

    Time span is [0, startup_days * 24) hours; DielForcingSchedule.at(t)
    wraps clock time modulo 24 h so diurnal forcing repeats each simulated day.
    """

    problem: LiquidOdeRhsProblem
    startup_days: float
    y0: StateVector | np.ndarray
    volume_m3_initial: float
    surface_area_m2: float
    include_rain: bool = True
    evaporation_floor_m3_h: float = 0.0
    """Lower bound on evaporation [m³ h⁻¹] when the schedule would otherwise give zero."""

    volume_minimum_m3: float = 0.01
    """Integration stops (terminal event) if V reaches this level."""

    def __post_init__(self) -> None:
        if self.startup_days <= 0:
            raise ValueError("startup_days must be positive")
        if self.volume_m3_initial <= 0:
            raise ValueError("volume_m3_initial must be positive")
        if self.surface_area_m2 <= 0:
            raise ValueError("surface_area_m2 must be positive")
        if self.evaporation_floor_m3_h < 0:
            raise ValueError("evaporation_floor_m3_h must be non-negative")
        if self.volume_minimum_m3 <= 0:
            raise ValueError("volume_minimum_m3 must be positive")
        _ = _as_length17_ndarray(self.y0)


@dataclass(frozen=True, slots=True)
class StartupBatchResult:
    t_hours: np.ndarray
    """Sample times [h], shape (n,)."""

    y: np.ndarray
    """State trajectory, shape (n, 17) (SI)."""

    volume_m3: np.ndarray
    """Volume trajectory [m³], shape (n,)."""

    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class StartupIntegrationMetadata:
    """Wall-clock and SciPy counters for one ``run_startup_batch`` ``solve_ivp`` call."""

    solver_wall_time_s: float
    """Elapsed wall time spent inside ``solve_ivp`` only."""

    n_output_points: int
    """Length of the returned time grid (``len(t_hours)``)."""

    nfev: int
    """Number of RHS evaluations reported by SciPy (0 if unavailable)."""

    njev: int | None
    """Jacobian evaluations if reported by SciPy; else ``None``."""

    success: bool
    message: str


def evaluate_startup_batch_rhs(
    t_hours: float,
    z: np.ndarray,
    *,
    problem_cfg: StartupBatchProblem,
) -> np.ndarray:
    """
    Augmented RHS for startup: return dz/dt with z = concat(y, [V]).

    t_hours may exceed 24; forcing uses the schedule's modulo-24 convention.

    Units: dz[:17]/dt in g m⁻³ h⁻¹; dz[17]/dt in m³ h⁻¹.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size != _STARTUP_N:
        raise ValueError(f"augmented state must have length {_STARTUP_N}, got {z.size}")

    y_raw = z[:N_STATE]
    # Integrators can produce slightly negative values at a step; StateVector and
    # Monod terms require nonnegative inventories. Clip before calling Stage-6 RHS.
    y = np.maximum(np.asarray(y_raw, dtype=np.float64), 0.0)

    v = float(z[_VOLUME_INDEX])
    v_safe = max(v, float(problem_cfg.volume_minimum_m3))

    sample = problem_cfg.problem.schedule.at(t_hours)
    dVdt = volume_derivative_m3_h(
        sample,
        problem_cfg.surface_area_m2,
        include_rain=problem_cfg.include_rain,
        evaporation_floor_m3_h=problem_cfg.evaporation_floor_m3_h,
    )

    dydt_day = evaluate_liquid_ode_rhs(t_hours, y, problem=problem_cfg.problem)
    dydt_hour = np.asarray(dydt_day, dtype=np.float64) / _HOURS_PER_DAY

    # Concentration change due to volume change (same formula for all 17 SI components;
    # see SIMULATOR_MATH / lumped control volume with variable volume).
    conc_term = -(y / v_safe) * dVdt
    dydt = dydt_hour + conc_term

    return np.concatenate([dydt, np.array([dVdt], dtype=np.float64)])


def _event_volume_minimum(_t: float, z: np.ndarray, *, V_minimum: float) -> float:
    """
    Event function for volume minimum: return V - V_minimum.
    Allows the integrator to stop if V reaches V_minimum.
    """
    return float(z[_VOLUME_INDEX]) - float(V_minimum)


def default_reasonable_startup_y0() -> np.ndarray:
    """
    A documented nominal SI initial condition: elevated substrate and modest inoculum.

    Values are order-of-magnitude placeholders for development tests, not calibrated
    to Casagli et al. (2021).
    """
    st = StateVector(
        X_ALG=50.0,
        X_AOB=12.0,
        X_NOB=6.0,
        X_H=70.0,
        X_S=140.0,
        X_I=25.0,
        S_S=220.0,
        S_I=18.0,
        S_IC=40.0,
        S_ND=42.0,
        S_NH=14.0,
        S_NO2=0.08,
        S_NO3=4.0,
        S_N2=0.15,
        S_PO4=15.0,
        S_O2=6.0,
        S_H2O=0.0,
    )
    return st.to_array(variant=StateVectorVariant.SI)


@overload
def run_startup_batch(
    cfg: StartupBatchProblem,
    *,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = 6.0,
    t_eval_hours: np.ndarray | None = None,
    return_integration_metadata: Literal[False] = False,
) -> StartupBatchResult: ...


@overload
def run_startup_batch(
    cfg: StartupBatchProblem,
    *,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = 6.0,
    t_eval_hours: np.ndarray | None = None,
    return_integration_metadata: Literal[True],
) -> tuple[StartupBatchResult, StartupIntegrationMetadata]: ...


def run_startup_batch(
    cfg: StartupBatchProblem,
    *,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = 6.0,
    t_eval_hours: np.ndarray | None = None,
    return_integration_metadata: bool = False,
) -> StartupBatchResult | tuple[StartupBatchResult, StartupIntegrationMetadata]:
    """
    Integrate startup batch (SI + volume) from t=0 to t = startup_days * 24 h.

    If t_eval_hours is None, samples every hour on [0, startup_days * 24].

    If ``return_integration_metadata`` is True, returns ``(result, metadata)`` where
    ``metadata.solver_wall_time_s`` covers only the ``solve_ivp`` call.
    """
    y0 = _as_length17_ndarray(cfg.y0)
    t_end = float(cfg.startup_days) * _HOURS_PER_DAY
    z0 = np.concatenate([y0, np.array([float(cfg.volume_m3_initial)], dtype=np.float64)])

    def rhs(t: float, z: np.ndarray) -> np.ndarray:
        return evaluate_startup_batch_rhs(t, z, problem_cfg=cfg)

    vmin = float(cfg.volume_minimum_m3)

    def event_volume(t: float, z: np.ndarray) -> float:
        return _event_volume_minimum(t, z, V_minimum=vmin)

    event_volume.terminal = True  # type: ignore[attr-defined]
    event_volume.direction = -1.0  # type: ignore[attr-defined]

    if t_eval_hours is None:
        n_steps = int(np.ceil(t_end)) + 1
        t_eval = np.linspace(0.0, t_end, max(n_steps, 2))
    else:
        t_eval = np.asarray(t_eval_hours, dtype=np.float64)

    t_ivp0 = perf_counter()
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        z0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=event_volume,
    )
    solver_wall_time_s = perf_counter() - t_ivp0

    nfev = int(getattr(sol, "nfev", 0))
    njev_raw = getattr(sol, "njev", None)
    njev = int(njev_raw) if njev_raw is not None else None

    def _meta(ok: bool, msg: str, n_out: int) -> StartupIntegrationMetadata:
        return StartupIntegrationMetadata(
            solver_wall_time_s=float(solver_wall_time_s),
            n_output_points=int(n_out),
            nfev=nfev,
            njev=njev,
            success=ok,
            message=msg,
        )

    if sol.t.size == 0:
        res = StartupBatchResult(
            t_hours=np.array([0.0]),
            y=y0.reshape(1, -1),
            volume_m3=np.array([float(cfg.volume_m3_initial)]),
            success=False,
            message="integrator returned empty time grid",
        )
        meta = _meta(False, res.message, 0)
        if return_integration_metadata:
            return res, meta
        return res

    y_traj = sol.y[:N_STATE].T
    v_traj = sol.y[_VOLUME_INDEX]
    msg = sol.message or ""
    ok = bool(sol.success)
    th = np.asarray(sol.t, dtype=np.float64)

    res = StartupBatchResult(
        t_hours=th,
        y=np.asarray(y_traj, dtype=np.float64),
        volume_m3=np.asarray(v_traj, dtype=np.float64),
        success=ok,
        message=msg,
    )
    meta = _meta(ok, msg, int(th.size))

    if return_integration_metadata:
        return res, meta
    return res
