"""Oracle: agreement + spec compliance + artifact validity. Fallback blocks physical claims."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models.datasheet_constraints import DatasheetConstraintVerifier
from .models.ngspice_bridge import SpiceSimulationResult
from .models.independent_verifier import IndependentDigitalVerifier as IndependentVerifier


@dataclass(frozen=True)
class OracleResult:
    final_metrics: dict[str, float]
    tool_used: str
    agreement: bool
    holdout_passed: bool
    spec_compliance: str = "UNKNOWN"
    artifact_valid: bool = False
    physical_claim: bool = False
    notes: tuple[str, ...] = ()


class ElectronicsOracle:
    def __init__(self) -> None:
        self.verifier = IndependentVerifier()

    def merge(
        self,
        analytical: SpiceSimulationResult | None,
        spice: SpiceSimulationResult | None,
        specs: dict[str, Any] | None = None,
        measurements: dict[str, Any] | None = None,
        electrical: Any | None = None,
    ) -> OracleResult:
        notes: list[str] = []
        spice_real = spice is not None and spice.success and spice.tool_used == "ngspice"
        chosen: SpiceSimulationResult | None = spice if spice_real else analytical
        if spice_real:
            tool = "ngspice"
        elif analytical is not None:
            tool = analytical.tool_used or "analytical_fallback"
        else:
            tool = "none"

        metrics: dict[str, float] = {}
        if chosen is not None:
            metrics = {
                "gain_db": float(chosen.gain_db),
                "bandwidth_mhz": float(chosen.bandwidth_mhz),
                "phase_margin_deg": float(chosen.phase_margin_deg),
                "power_mw": float(chosen.power_mw),
            }
            if chosen.raw_metrics:
                metrics.update({k: float(v) for k, v in chosen.raw_metrics.items() if isinstance(v, (int, float))})

        agreement = False
        if spice_real and analytical is not None:
            agreement = True
            for k in ("gain_db", "bandwidth_mhz"):
                av = getattr(analytical, k, 0.0)
                sv = getattr(spice, k, 0.0)
                if av and sv and abs(av - sv) / max(abs(av), 1e-9) > 0.2:
                    agreement = False
        elif not spice_real:
            notes.append("single_tool_no_agreement")

        artifact_valid = bool(chosen and chosen.success and metrics)

        meas = dict(measurements or {})
        meas.update(metrics)
        if electrical is not None:
            spec_compliance = DatasheetConstraintVerifier.from_electrical(electrical).check(meas)["verdict"]
        elif specs:
            spec_compliance = self._specs_verdict(meas, specs)
        else:
            spec_compliance = "UNKNOWN"

        physical_claim = tool == "ngspice" and artifact_valid
        if tool != "ngspice":
            notes.append("physical_claim_blocked_no_spice")

        return OracleResult(
            final_metrics=metrics,
            tool_used=tool,
            agreement=agreement,
            holdout_passed=False,
            spec_compliance=spec_compliance,
            artifact_valid=artifact_valid,
            physical_claim=physical_claim,
            notes=tuple(notes),
        )

    def _specs_verdict(self, meas: dict[str, Any], specs: dict[str, Any]) -> str:
        from .models.datasheet_constraints import Constraint

        constraints = []
        for name, rule in specs.items():
            if isinstance(rule, (list, tuple)) and len(rule) >= 2:
                op, limit = rule[0], rule[1]
                constraints.append(Constraint(str(name), str(op), float(limit), (str(name),)))
        if not constraints:
            return "UNKNOWN"
        return DatasheetConstraintVerifier(constraints).check(meas)["verdict"]
