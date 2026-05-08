"""Time integration for the 17-component liquid ODE (phase1-04d).

Independent variable for solve_ivp is elapsed time in hours. Forcing uses
LiquidOdeRhsProblem.schedule.at(t_hours), which maps clock time modulo 24 h.

Rates from evaluate_liquid_ode_rhs are expressed per day (g m⁻³ d⁻¹). They are
divided by 24 before integration so that derivatives match per hour, consistent
with bioprocess_twin.simulator.startup_batch.run_startup_batch.

**Solver choice:** default method="LSODA" switches between Adams and BDF and handles
many stiff biotech systems. For strongly stiff problems you can pass method="BDF"
(implicit, may benefit from an analytic Jacobian in future work).

**Tolerances:** rtol and atol are passed to SciPy in the usual relative/absolute
sense for the state vector. max_step is a cap on the integrator step in hours
(not the spacing of t_eval output).

**t_eval vs dense output:** t_eval selects output times; it does not force the
internal step size. With dense_output=True, SciPy builds a piecewise interpolant
ode_result.sol so you can evaluate the state at extra times without re-integrating.
If dense_output=False, ode_result.sol is None even when return_ode_result=True.

Downstream quantities (kinetic rates, env snapshots) are not stored on the SciPy
result; call evaluate_liquid_rhs / schedules at (t, y) as needed.

**Troubleshooting (integrator fails or success=False):**
- Read message on LiquidIntegrationResult and inspect SciPy ode_result.message if attached.
- Verify evaluate_liquid_ode_rhs(t0, y0, problem) is finite (no NaN/Inf) at the start.
- Reduce max_step (hours); increase rtol/atol only if you accept looser error control.
- Try method="BDF" for stiff behavior; optional first_step (hours) to limit the first internal step.
- Remember schedule.at wraps time modulo 24 h; check callables for sharp transitions at day boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    ode_result: Any | None = None
    """Raw solve_ivp result when requested; use .sol(t) if dense_output was True."""


def interpolate_liquid_trajectory(result: LiquidIntegrationResult, t_hours: np.ndarray) -> np.ndarray:
    """
    Interpolate the liquid state at query times using SciPy's dense solution.

    Requires integrate_liquid_ode(..., dense_output=True, return_ode_result=True).
    Query times must lie within the interpolant's domain [sol.t_min, sol.t_max]
    (integration interval), typically matching t_span_hours.

    Parameters
    ----------
    result
        Integration result carrying ode_result.sol.
    t_hours
        Query times [h], shape (m,) or scalar coercible to float.

    Returns
    -------
    numpy.ndarray
        State samples of shape (m, N_STATE) (or (1, N_STATE) for a scalar time).
    """
    if result.ode_result is None:
        raise ValueError("interpolate_liquid_trajectory requires return_ode_result=True on integrate_liquid_ode")
    sol_dense = result.ode_result.sol
    if sol_dense is None:
        raise ValueError(
            "interpolate_liquid_trajectory requires dense_output=True; ode_result.sol is None",
        )

    t_q = np.asarray(t_hours, dtype=np.float64).ravel()
    if t_q.size == 0:
        raise ValueError("t_hours must be non-empty")

    t_min = float(sol_dense.t_min)
    t_max = float(sol_dense.t_max)
    eps = 1e-12
    if np.any(t_q < t_min - eps) or np.any(t_q > t_max + eps):
        raise ValueError(
            f"query times must lie within [{t_min}, {t_max}] hours (interpolant domain); "
            f"got min {float(np.min(t_q)):.6g}, max {float(np.max(t_q)):.6g}",
        )

    raw = sol_dense(t_q)
    mat = np.asarray(raw, dtype=np.float64)
    if mat.ndim == 1:
        return mat.reshape(1, -1)
    # SciPy: shape (N_STATE, m)
    return mat.T


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
    return_ode_result: bool = False,
    first_step: float | None = None,
) -> LiquidIntegrationResult:
    """
    Integrate dy/dt for the Stage-6 liquid path + optional CSTR transport.

    Independent variable is elapsed time in hours. Forcing uses
    problem.schedule.at(t), which wraps clock time modulo 24 h (repeating diel).

    Rates from evaluate_liquid_ode_rhs are per day (g m⁻³ d⁻¹); they are
    converted to per hour by dividing by 24 so that solve_ivp is consistent
    with t in hours (same convention as startup_batch.run_startup_batch).

    Parameters
    ----------
    problem
        Bundles schedule, kinetic options, and optional CSTR config.
    y0
        Initial SI state vector (length N_STATE).
    t_span_hours
        (t_start, t_end) integration window [h].
    t_eval
        Times at which to store the solution. If None, uses an hourly-ish grid
        on ``[t_span_hours[0], t_span_hours[1]]`` with at least two points.
    method
        SciPy ODE method (default LSODA). Use BDF for stiff systems if needed.
    rtol, atol
        Relative and absolute tolerances for the state vector (SciPy semantics).
    max_step
        Maximum integrator step size in hours (None means no cap).
    clip_nonnegative
        If True (default), clip concentrations to >= 0 before each RHS evaluation
        (integrators may produce slightly negative intermediates).
    dense_output
        If True, SciPy retains a dense interpolant on ode_result.sol (when returned).
    return_ode_result
        If True, attach the raw SciPy object from solve_ivp as LiquidIntegrationResult.ode_result
        (optional inspection, dense sampling via ode_result.sol(t) when dense_output is True).
    first_step
        Optional suggested first integration step in hours; forwarded to solve_ivp when set.

    Returns
    -------
    LiquidIntegrationResult
        Trajectory samples, integrator status message, and optionally the SciPy result object.

    Notes
    -----
    See the module docstring for troubleshooting when integration fails.
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

    ivp_kwargs: dict[str, Any] = {
        "method": method,
        "t_eval": te,
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "dense_output": dense_output,
    }
    if first_step is not None:
        ivp_kwargs["first_step"] = float(first_step)

    sol = solve_ivp(
        rhs,
        (t0, t1),
        y0_arr,
        **ivp_kwargs,
    )

    raw = sol if return_ode_result else None

    if sol.t.size == 0:
        return LiquidIntegrationResult(
            t_hours=np.array([t0]),
            y=y0_arr.reshape(1, -1),
            success=False,
            message=sol.message or "integrator returned empty time grid",
            ode_result=raw,
        )

    y_out = np.asarray(sol.y.T, dtype=np.float64)
    msg = sol.message or ""
    return LiquidIntegrationResult(
        t_hours=np.asarray(sol.t, dtype=np.float64),
        y=y_out,
        success=bool(sol.success),
        message=msg,
        ode_result=raw,
    )


__all__ = ["LiquidIntegrationResult", "integrate_liquid_ode", "interpolate_liquid_trajectory"]
