"""Core logic and simulation engine."""

from __future__ import annotations

from typing import Any

__all__ = ["LiquidIntegrationResult", "integrate_liquid_ode", "interpolate_liquid_trajectory"]


def __getattr__(name: str) -> Any:
    if name == "LiquidIntegrationResult":
        from bioprocess_twin.core.simulation import LiquidIntegrationResult

        return LiquidIntegrationResult
    if name == "integrate_liquid_ode":
        from bioprocess_twin.core.simulation import integrate_liquid_ode

        return integrate_liquid_ode
    if name == "interpolate_liquid_trajectory":
        from bioprocess_twin.core.simulation import interpolate_liquid_trajectory

        return interpolate_liquid_trajectory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
