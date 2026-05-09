#!/usr/bin/env python3
"""
Reproducible 3-day (hourly) startup-batch experiment: open pond + SI state + volume.

Writes ``trajectory.csv``, ``experiment_summary.md``, optional PNG plots under ``--output-dir``.

Timing policy: only total wall time inside ``solve_ivp`` (via ``StartupIntegrationMetadata``);
``estimated_seconds_per_output_sample`` in the summary is total / n rows (indicative only).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bioprocess_twin.experiments import (
    SI_STATE_COLUMNS,
    hourly_t_eval_for_startup_days,
    trajectory_to_dataframe,
)
from bioprocess_twin.forcing import DielForcingSchedule
from bioprocess_twin.simulator import (
    LiquidOdeRhsProblem,
    StartupBatchProblem,
    default_reasonable_startup_y0,
    run_startup_batch,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def build_experiment_summary_md(
    *,
    cfg: StartupBatchProblem,
    csv_path: Path,
    summary_path: Path,
    plots_dir: Path | None,
    meta,
    utc_iso: str,
    estimated_seconds_per_output_sample: float,
    scipy_ver: str,
    numpy_ver: str,
    python_exe: str,
) -> str:
    """Return Markdown body for ``experiment_summary.md`` and stdout."""
    njev_line = "n/a" if meta.njev is None else str(meta.njev)
    plots_line = f"`{plots_dir}`" if plots_dir is not None else "(not generated)"

    body = f"""# Startup batch experiment summary

    ## Configuration

    - **Season (Fig. 1 forcing):** `{cfg.problem.schedule.season}`
    - **startup_days:** `{cfg.startup_days}`
    - **include_rain:** `{cfg.include_rain}`
    - **volume_m3_initial:** `{cfg.volume_m3_initial}` m³
    - **surface_area_m2:** `{cfg.surface_area_m2}` m²
    - **evaporation_floor_m3_h:** `{cfg.evaporation_floor_m3_h}`
    - **volume_minimum_m3:** `{cfg.volume_minimum_m3}`
    - **LiquidOdeRhsProblem.cstr:** `{cfg.problem.cstr}`

    ## Integration

    - **success:** `{meta.success}`
    - **SciPy message:**

    ```
    {meta.message}
    ```

    ## Performance

    Wall times refer to the **single** ``solve_ivp`` call only.

    | Metric | Value |
    |--------|-------|
    | ``solver_wall_time_s_total`` | {meta.solver_wall_time_s:.6g} |
    | ``n_output_points`` | {meta.n_output_points} |
    | ``nfev`` | {meta.nfev} |
    | ``njev`` | {njev_line} |
    | ``estimated_seconds_per_output_sample`` | {estimated_seconds_per_output_sample:.6g} |

    *Note:* ``estimated_seconds_per_output_sample`` is total wall time divided by ``n_output_points``.
    *(Indicative only; not real per-internal-step CPU.)*

    ## Outputs

    - **Trajectory CSV:** `{csv_path}`
    - **This summary:** `{summary_path}`
    - **Plots directory:** {plots_line}

    ## Environment

    - **UTC run timestamp:** `{utc_iso}`
    - **Python:** `{python_exe}`
    - **SciPy:** `{scipy_ver}`
    - **NumPy:** `{numpy_ver}`
    """
    return body


def _plot_groups(t_hours: np.ndarray, y: np.ndarray, out_dir: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for --plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    th = np.asarray(t_hours, dtype=np.float64)

    groups: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("biomass_particulate", (0, 1, 2, 3, 4, 5)),
        ("soluble_cod", (6, 7)),
        ("inorganic_c_n", (8, 9, 10, 11, 12, 13)),
        ("phosphorus_oxygen", (14, 15)),
        ("water_balance", (16,)),
    )

    for fname, idxs in groups:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in idxs:
            ax.plot(th, y[:, i], label=SI_STATE_COLUMNS[i], linewidth=1.2)
        ax.set_xlabel("t [h]")
        ax.set_ylabel("concentration (model units)")
        ax.set_title(fname.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}.png", dpi=120)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 3-day hourly startup_batch experiment (default settings).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/experiments/output/startup_batch_3day"),
        help="Directory for CSV, Markdown summary, and optional plots.",
    )
    parser.add_argument("--season", type=str, default="summer", choices=("spring", "summer", "autumn", "winter"))
    parser.add_argument("--startup-days", type=float, default=3.0, help="Horizon in days (default: 3).")
    parser.add_argument("--volume-m3", type=float, default=17.0, help="Initial volume [m³].")
    parser.add_argument("--surface-m2", type=float, default=56.0, help="Surface area [m²].")
    parser.add_argument("--plots", action="store_true", help="Write grouped PNG plots (requires matplotlib).")
    args = parser.parse_args(argv)

    out_dir: Path = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = LiquidOdeRhsProblem(schedule=DielForcingSchedule(season=args.season))

    cfg = StartupBatchProblem(
        problem=problem,
        startup_days=float(args.startup_days),
        y0=default_reasonable_startup_y0(),
        volume_m3_initial=float(args.volume_m3),
        surface_area_m2=float(args.surface_m2),
        include_rain=False,
    )

    t_eval = hourly_t_eval_for_startup_days(float(args.startup_days))

    utc_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(
        "Simulation is running; this may take several minutes depending on horizon and hardware.",
        flush=True,
    )
    res, meta = run_startup_batch(cfg, t_eval_hours=t_eval, return_integration_metadata=True)

    df = trajectory_to_dataframe(res.t_hours, res.y, res.volume_m3)
    csv_path = out_dir / "trajectory.csv"
    df.to_csv(csv_path, index=False)

    n_out = max(meta.n_output_points, 1)
    est_per_sample = meta.solver_wall_time_s / float(n_out)

    import platform

    try:
        import scipy  # type: ignore[import-not-found]

        scipy_ver = scipy.__version__
    except Exception:
        scipy_ver = "unknown"

    numpy_ver = np.__version__
    python_exe = sys.executable or platform.python_version()

    plots_dir: Path | None = None
    if args.plots:
        plots_dir = out_dir / "plots"
        _plot_groups(res.t_hours, res.y, plots_dir)

    summary_path = out_dir / "experiment_summary.md"
    md = build_experiment_summary_md(
        cfg=cfg,
        csv_path=csv_path,
        summary_path=summary_path,
        plots_dir=plots_dir,
        meta=meta,
        utc_iso=utc_iso,
        estimated_seconds_per_output_sample=est_per_sample,
        scipy_ver=scipy_ver,
        numpy_ver=numpy_ver,
        python_exe=python_exe,
    )
    summary_path.write_text(md, encoding="utf-8")
    print(md)

    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
