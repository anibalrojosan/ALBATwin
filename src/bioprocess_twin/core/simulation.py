"""Short-horizon time integration for the 17-component liquid ODE (phase1-04d Etapa A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from bioprocess_twin.core.state import StateVector, StateVectorVariant
from bioprocess_twin.models.stoichiometry import N_STATE
from bioprocess_twin.simulator.liquid_ode_rhs import LiquidOdeRhsProblem, evaluate_liquid_ode_rhs

_HOURS_PER_DAY = 24.0


def _as_length_n_state_y0(y0: StateVector | np.ndarray) -> np.ndarray:
    """Convert a StateVector or np.ndarray to a length-N_STATE np.ndarray."""
    if isinstance(y0, StateVector):
        arr = y0.to_array(variant=StateVectorVariant.SI)
    else:
        arr = np.asarray(y0, dtype=np.float64).ravel()
    if arr.size != N_STATE:
        raise ValueError(f"y0 must have length {N_STATE} (SI layout), got {arr.size}")
    return arr


@dataclass(frozen=True, slots=True)
class LiquidIntegrationResult:
    """Outcome of integrating the lumped liquid state in hours."""

    t_hours: np.ndarray
    """Sample times [h], shape (n,)."""

    y: np.ndarray
    """State trajectory (SI), shape (n, N_STATE)."""

    success: bool
    message: str


def integrate_liquid_ode(
    problem: LiquidOdeRhsProblem,
    y0: StateVector | np.ndarray,
    t_span_hours: tuple[float, float],
    *,
    t_eval: np.ndarray | None = None,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = 6.0,
    clip_nonnegative: bool = True,
    dense_output: bool = False,
) -> LiquidIntegrationResult:
    """
    Integrate dy/dt for the Stage-6 liquid path + optional CSTR transport.

    Independent variable is "elapsed time in hours". Forcing uses
    `problem.schedule.at(t)`, which wraps clock time modulo 24 h (repeating diel).

    Rates from `evaluate_liquid_ode_rhs` are per day (g m⁻³ d⁻¹); they are
    converted to per hour by dividing by 24 so that `solve_ivp` is consistent
    with `t` in hours (same convention as `startup_batch.run_startup_batch`).

    Parameters
    ----------
    problem
        Bundles schedule, kinetic options, and optional CSTR config.
    y0
        Initial SI state vector (length `N_STATE`).
    t_span_hours
        `(t_start, t_end)` integration window [h].
    t_eval
        Times at which to store the solution. If `None`, uses an hourly-ish grid
        on `[t_span_hours[0], t_span_hours[1]]` with at least two points.
    clip_nonnegative
        If True (default), clip concentrations to `>= 0` before each RHS evaluation
        (integrators may produce slightly negative intermediates).
    dense_output
        Passed through to ``solve_ivp`` (Etapa B may rely on dense output sampling).

    Returns
    -------
    LiquidIntegrationResult
        Trajectory samples and integrator status message.
    """
    y0_arr = _as_length_n_state_y0(y0)
    t0, t1 = float(t_span_hours[0]), float(t_span_hours[1])
    if t1 <= t0:
        raise ValueError(f"t_span_hours must have t_end > t_start, got ({t0}, {t1})")

    def rhs(t_hours: float, y: np.ndarray) -> np.ndarray:
        y_in = np.asarray(y, dtype=np.float64).ravel()
        y_use = np.maximum(y_in, 0.0) if clip_nonnegative else y_in
        d_day = evaluate_liquid_ode_rhs(float(t_hours), y_use, problem=problem)
        return np.asarray(d_day, dtype=np.float64) / _HOURS_PER_DAY

    if t_eval is None:
        span = t1 - t0
        n_steps = max(2, int(np.ceil(span)) + 1)
        te = np.linspace(t0, t1, n_steps)
    else:
        te = np.asarray(t_eval, dtype=np.float64)

    sol = solve_ivp(
        rhs,
        (t0, t1),
        y0_arr,
        method=method,
        t_eval=te,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense_output,
    )

    if sol.t.size == 0:
        return LiquidIntegrationResult(
            t_hours=np.array([t0]),
            y=y0_arr.reshape(1, -1),
            success=False,
            message=sol.message or "integrator returned empty time grid",
        )

    y_out = np.asarray(sol.y.T, dtype=np.float64)
    msg = sol.message or ""
    return LiquidIntegrationResult(
        t_hours=np.asarray(sol.t, dtype=np.float64),
        y=y_out,
        success=bool(sol.success),
        message=msg,
    )


__all__ = ["LiquidIntegrationResult", "integrate_liquid_ode"]
