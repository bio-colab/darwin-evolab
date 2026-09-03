"""
objective_matrix.py — Multi-Objective Pareto Weight Configuration and Trade-off Matrix.

Allows users to steer evolutionary selection pressure toward low-power edge IoT,
ultra-high speed, minimum silicon area, or a balanced Pareto frontier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveMatrix:
    """Multi-objective weighting profile."""
    mode: str = "balanced"  # "power", "speed", "area", "balanced", "custom"
    accuracy_base: float = 70.0
    power_weight: float = 10.0
    delay_weight: float = 10.0
    area_weight: float = 10.0

    @classmethod
    def from_preset(cls, name: str) -> ObjectiveMatrix:
        preset = name.lower().strip()
        if preset in ("power", "low_power", "iot"):
            return cls(mode="power", power_weight=25.0, delay_weight=5.0, area_weight=5.0)
        elif preset in ("speed", "fast", "low_delay"):
            return cls(mode="speed", power_weight=5.0, delay_weight=25.0, area_weight=5.0)
        elif preset in ("area", "compact", "min_silicon"):
            return cls(mode="area", power_weight=5.0, delay_weight=5.0, area_weight=25.0)
        elif preset in ("balanced", "pareto"):
            return cls(mode="balanced", power_weight=10.0, delay_weight=10.0, area_weight=10.0)
        return cls(mode="balanced")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveMatrix:
        if "preset" in data:
            return cls.from_preset(data["preset"])
        return cls(
            mode="custom",
            accuracy_base=float(data.get("accuracy_base", 70.0)),
            power_weight=float(data.get("power_weight", 10.0)),
            delay_weight=float(data.get("delay_weight", 10.0)),
            area_weight=float(data.get("area_weight", 10.0)),
        )

    def calculate_score(
        self,
        is_functional: bool,
        accuracy: float,
        critical_delay_fo4: float,
        active_gates: int,
        toggle_ratio: float = 0.5,
    ) -> float:
        """Calculates normalized fitness [0.0, 100.0] respecting the user's objective matrix."""
        if not is_functional:
            # Functional correctness is the hard gate
            return accuracy * self.accuracy_base - (active_gates * 0.05)

        # Secondary bonuses
        area_bonus = self.area_weight / (1.0 + active_gates)
        delay_bonus = self.delay_weight / (1.0 + critical_delay_fo4)
        power_bonus = self.power_weight * max(0.0, 1.0 - toggle_ratio)

        total = self.accuracy_base + area_bonus + delay_bonus + power_bonus
        return min(100.0, max(0.0, total))
