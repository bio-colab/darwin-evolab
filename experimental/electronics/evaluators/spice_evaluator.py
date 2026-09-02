"""
spice_evaluator.py — Level 2: Parameter Sizing Evaluator for Analog Components.
Evaluates vector genomes (transistor widths/lengths, passive values) against analog specs.
Enhanced with JSON-config driven mode (opensource-analog-circuits style) — low-cost, non-breaking.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import FloatGenome, Individual
from ..models.ngspice_bridge import NGSpiceBridge, SpiceSimulationResult
from ..oracle import ElectronicsOracle


def derate_for_slow_corner(res: SpiceSimulationResult, vcc_v: float = 4.5, temp_c: float = 85.0) -> SpiceSimulationResult:
    """Paper-model PVT: low VCC and high temp shrink gain/bandwidth."""
    v_scale = vcc_v / 5.0
    t_scale = max(0.5, 1.0 - 0.003 * (temp_c - 25.0))
    return SpiceSimulationResult(
        success=res.success,
        gain_db=round(res.gain_db * v_scale * t_scale, 2),
        bandwidth_mhz=round(res.bandwidth_mhz * v_scale * t_scale, 2),
        phase_margin_deg=round(res.phase_margin_deg * (0.94 if temp_c > 25 else 1.0), 2),
        power_mw=round(res.power_mw * (1.0 + 0.002 * (temp_c - 25.0)), 2),
        execution_time_ms=res.execution_time_ms,
        tool_used=res.tool_used,
        raw_metrics=dict(res.raw_metrics or {}),
    )


def compute_objective(metrics: dict[str, float], specs: dict[str, list], objective_type: str = "sum_violations") -> float:
    """Normalized violation sum (lower is better) — mirrors benchmark/ngspice_benchmark.py:265."""
    total = 0.0
    for k, (op, target) in specs.items():
        v = metrics.get(k)
        if v is None:
            total += 1000.0
            continue
        if op == "<":
            viol = max(0.0, float(v) - float(target))
        elif op == ">":
            viol = max(0.0, float(target) - float(v))
        else:
            viol = 0.0
        norm = viol / abs(float(target)) if float(target) != 0 else viol
        total += norm
    return total


class AnalogSizingEvaluator(Evaluator):
    """Evaluates analog circuit parameter sizing for Gain, Bandwidth, Stability, and Power."""

    def __init__(
        self,
        target_gain_db: float = 40.0,
        target_ugbw_mhz: float = 10.0,
        min_phase_margin_deg: float = 45.0,
        max_power_mw: float = 5.0,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.target_gain_db = target_gain_db
        self.target_ugbw_mhz = target_ugbw_mhz
        self.min_phase_margin_deg = min_phase_margin_deg
        self.max_power_mw = max_power_mw
        self.config_path = Path(config_path) if config_path else None
        self.bridge = NGSpiceBridge()
        self.oracle = ElectronicsOracle()
        # optional config-driven specs (non-breaking: only if file exists)
        self._config: dict[str, Any] | None = None
        self._config_specs: dict[str, list] | None = None
        self._config_design_vars: dict[str, list] | None = None
        if self.config_path and self.config_path.exists():
            try:
                self._config = json.loads(self.config_path.read_text(encoding="utf-8"))
                self._config_specs = self._config.get("specs", {})
                self._config_design_vars = self._config.get("design_vars", {})
            except Exception:
                self._config = None

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def spec_descriptor(self) -> dict:
        """Archive key material: everything that changes what a score means."""
        return {
            "family": "analog_sizing",
            "target_gain_db": self.target_gain_db,
            "target_ugbw_mhz": self.target_ugbw_mhz,
            "min_phase_margin_deg": self.min_phase_margin_deg,
            "max_power_mw": self.max_power_mw,
            "config": self.config_path.name if self.config_path else None,
            "config_specs": self._config_specs or {},
        }

    def evaluate(self, target: Any, **kwargs: Any) -> FitnessResult:
        genome = target.genome if isinstance(target, Individual) else target
        if not isinstance(genome, FloatGenome):
            return FitnessResult(score=0.0, passed_holdout=False, artifacts={"error": "Target is not FloatGenome"})

        # Vector genes map to transistor parameters W1, L1, W2, L2, Bias
        genes = genome.values
        if len(genes) < 4:
            return FitnessResult(
                score=0.0,
                passed_holdout=False,
                artifacts={"error": "At least 4 sizing genes required"},
            )

        w1, l1, w2, l2 = [abs(g) + 0.18 for g in genes[:4]]

        # Construct netlist — EvoLab is optimizer, not simulator: delegates to simulators/
        netlist = f"""* Analog Sizing Stage
M1 out in vdd vdd pmos w={w1:.2f}u l={l1:.2f}u
M2 out in gnd gnd nmos w={w2:.2f}u l={l2:.2f}u
.ac dec 10 1k 100Meg
"""
        # Evaluator -> simulators (analytical + SPICE) -> Oracle
        analytical = self.bridge._analytical_cmos_inverter_chain(netlist)
        spice_res = self.bridge.run_netlist(netlist)
        # Oracle merges; spice_res is already SPICE-or-fallback, analytical is pure fallback
        oracle_res = self.oracle.merge(analytical, spice_res)
        # Use SPICE result when available, else analytical (oracle already prefers SPICE)
        sim_res = spice_res if spice_res.tool_used == "ngspice" and spice_res.success else analytical
        # Record oracle agreement for reporting
        oracle_agreement = oracle_res.agreement

        # Gain score
        gain_score = min(1.0, sim_res.gain_db / self.target_gain_db)
        # Bandwidth score
        ugbw_score = min(1.0, sim_res.bandwidth_mhz / self.target_ugbw_mhz)
        # Phase margin stability
        stability_score = 1.0 if sim_res.phase_margin_deg >= self.min_phase_margin_deg else 0.2
        # Power penalty
        power_score = max(0.0, 1.0 - (sim_res.power_mw / (self.max_power_mw * 2.0)))

        composite = 0.40 * gain_score + 0.30 * ugbw_score + 0.20 * stability_score + 0.10 * power_score

        # Silicon area penalty for toy evaluator
        area = w1 * l1 + w2 * l2
        area_penalty = 0.1 * (area / 100.0)

        hold = derate_for_slow_corner(sim_res)
        passed_holdout = (
            hold.gain_db >= self.target_gain_db
            and hold.bandwidth_mhz >= self.target_ugbw_mhz
            and hold.phase_margin_deg >= self.min_phase_margin_deg
            and hold.power_mw <= self.max_power_mw
        )
        if sim_res.tool_used != "ngspice":
            passed_holdout = False

        return FitnessResult(
            score=round(composite * 100.0, 4),
            passed_holdout=passed_holdout,
            sub_scores={
                "gain_score": round(gain_score * 100.0, 2),
                "ugbw_score": round(ugbw_score * 100.0, 2),
                "stability_score": round(stability_score * 100.0, 2),
                "power_score": round(power_score * 100.0, 2),
            },
            artifacts={
                "gain_db": sim_res.gain_db,
                "bandwidth_mhz": sim_res.bandwidth_mhz,
                "phase_margin_deg": sim_res.phase_margin_deg,
                "power_mw": sim_res.power_mw,
                "tool_used": sim_res.tool_used or "analytical_fallback",
                "oracle_agreement": oracle_agreement,
                "oracle_tool": oracle_res.tool_used,
                "physical_claim": oracle_res.physical_claim,
                "spec_compliance": oracle_res.spec_compliance,
                "artifact_valid": oracle_res.artifact_valid,
                "holdout_vcc_v": 4.5,
                "holdout_temp_c": 85.0,
                "holdout_gain_db": hold.gain_db,
                "holdout_bandwidth_mhz": hold.bandwidth_mhz,
                "area_um2": round(area, 2),
                "area_penalty": round(0.1 * (area / 100.0), 4),
            },
        )


class CircuitConfigEvaluator(Evaluator):
    """Config-driven NGSPICE evaluator (opensource-analog-circuits style).
    Loads `config.json` (design_vars/specs/metrics_patterns) and runs the
    real netlist file via NGSpiceBridge. Cheap, zero-PDK circuits only.
    """

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")
        self._cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.design_vars: dict[str, list] = self._cfg.get("design_vars", {})
        self.specs: dict[str, list] = self._cfg.get("specs", {})
        # fix upstream typo in chargepump pattern
        self.patterns: dict[str, str] = self._cfg.get("metrics_patterns", {})
        for k, pat in list(self.patterns.items()):
            if pat == r"^\s*vout\s+=\s+([\-\d\.eE+])":
                self.patterns[k] = r"^\s*vout\s+=\s+([\-\d\.eE+]+)"
        self.parser_type: str = self._cfg.get("metrics_parser_type", "regex")
        self.circuit_rel: str = self._cfg.get("circuit_file", "")
        # resolve circuit file relative to electronics root
        # config is at electronics/circuits/<name>/config.json, circuit_file is circuits/<name>/circuit.cir
        electronics_root = self.config_path.parent.parent.parent  # .../electronics
        candidate = electronics_root / self.circuit_rel
        if candidate.exists():
            self.circuit_file = candidate
        else:
            # fallback: relative to config parent
            self.circuit_file = (self.config_path.parent / Path(self.circuit_rel).name).resolve()
        self.names = list(self.design_vars.keys())
        self.lb = [float(self.design_vars[n][1]) for n in self.names]
        self.ub = [float(self.design_vars[n][2]) for n in self.names]
        self.defaults = [float(self.design_vars[n][0]) for n in self.names]
        self.dim = len(self.names)
        self.bridge = NGSpiceBridge()
        self.oracle = ElectronicsOracle()
        # Supply-rail estimate parsed once from the netlist text (used only for
        # labeled model estimates below — never presented as a measurement).
        self._vcc_est = self._parse_vcc_estimate(self.circuit_file)

    @staticmethod
    def _parse_vcc_estimate(circuit_file: Path) -> float:
        try:
            text = circuit_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return 5.0
        m = re.search(r"\.param\s+VCC\s*=\s*([\d.]+)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"^VCC\s+\S+\s+0\s+([\d.]+)", text, re.IGNORECASE | re.MULTILINE)
        try:
            return float(m.group(1)) if m else 5.0
        except ValueError:
            return 5.0

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def cost_estimate(self) -> str:
        return "cheap"

    def spec_descriptor(self) -> dict:
        """Archive key material: everything that changes what a score means."""
        return {
            "family": "circuit_config",
            "config_path": str(self.config_path.resolve()),
            "specs": {k: list(v) for k, v in self.specs.items()},
            "design_vars": {n: list(v) for n, v in self.design_vars.items()},
            "circuit_file": str(self.circuit_file),
        }

    def _denormalize(self, genes: list[float]) -> dict[str, float]:
        # genes may be normalized [0,1] or physical values; detect via range
        # If genes are within [0,1] approx and dim matches, treat as normalized
        is_normalized = all(0.0 <= g <= 1.0 for g in genes) and self.dim > 0
        if is_normalized:
            params: dict[str, float] = {}
            for i, name in enumerate(self.names):
                val = self.lb[i] + float(genes[i]) * (self.ub[i] - self.lb[i])
                params[name] = val
            return params
        # else treat as physical mapping by order
        return {name: float(genes[i]) for i, name in enumerate(self.names) if i < len(genes)}

    def evaluate(self, target: Any, **kwargs: Any) -> FitnessResult:
        genome = target.genome if isinstance(target, Individual) else target
        if not isinstance(genome, FloatGenome):
            return FitnessResult(score=0.0, passed_holdout=False, artifacts={"error": "Target is not FloatGenome"})
        genes = list(genome.values)
        if len(genes) < self.dim:
            return FitnessResult(
                score=0.0,
                passed_holdout=False,
                artifacts={"error": f"Expected {self.dim} genes, got {len(genes)}"},
            )
        # use first dim genes
        params = self._denormalize(genes[: self.dim])
        # run circuit file with param injection; pass configured .meas patterns so
        # ngspice-measured values land in raw_metrics under their spec names
        if self.circuit_file.exists():
            sim_res = self.bridge.run_circuit_file(
                self.circuit_file, params=params, meas_patterns=self.patterns
            )
        else:
            sim_res = self.bridge.run_netlist(f"* fallback {params}")

        # Collect metrics with explicit provenance. Nothing here may masquerade
        # as a measurement: every value is tagged measured_ngspice /
        # analytical_model / datasheet_formula / model_estimate, and spec keys
        # that cannot be derived are left unmeasured (objective penalty).
        src_tool = "measured_ngspice" if sim_res.tool_used == "ngspice" else "analytical_model"
        metrics: dict[str, float] = {}
        metric_source: dict[str, str] = {}
        unmeasured_specs: list[str] = []
        if self.parser_type == "ac_data":
            # sim_res contains gain/bw/pm from the AC table if ngspice succeeded
            metrics["gain"] = sim_res.gain_db
            metrics["ugf"] = sim_res.bandwidth_mhz * 1e6
            metrics["pm"] = sim_res.phase_margin_deg
            metrics["gain_db"] = sim_res.gain_db
            for k in ("gain", "ugf", "pm", "gain_db"):
                metric_source[k] = src_tool
        else:
            if sim_res.raw_metrics:
                for k, v in sim_res.raw_metrics.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
                        metric_source[k] = src_tool
            # derivation ladder for spec keys ngspice could not measure
            for spec_key in self.specs:
                if spec_key in metrics:
                    continue
                if spec_key == "freq":
                    # 555 astable datasheet law: f = 1.44/((R1+2*R2)*C1)
                    r1 = float(params.get("R1", 10e3))
                    r2 = float(params.get("R2", 10e3))
                    c1 = float(params.get("C1", 100e-9))
                    denom = (r1 + 2 * r2) * c1
                    metrics[spec_key] = 1.44 / denom if denom > 0 else 600.0
                    metric_source[spec_key] = "datasheet_formula"
                elif spec_key == "t_period":
                    r1 = float(params.get("R1", 10e3))
                    r2 = float(params.get("R2", 10e3))
                    c1 = float(params.get("C1", 100e-9))
                    metrics[spec_key] = (r1 + 2 * r2) * c1 / 1.44 if c1 > 0 else 0.002
                    metric_source[spec_key] = "datasheet_formula"
                elif spec_key == "vout_high":
                    # BJT/comparator pull-up estimate: rail minus one saturation drop
                    metrics[spec_key] = self._vcc_est - 0.7
                    metric_source[spec_key] = "model_estimate"
                elif spec_key == "vout_low":
                    metrics[spec_key] = 0.25
                    metric_source[spec_key] = "model_estimate"
                elif spec_key == "vout":
                    # mid-rail supply estimate; deliberately conservative
                    metrics[spec_key] = 0.5 * self._vcc_est
                    metric_source[spec_key] = "model_estimate"
                # else: leave unmeasured -> objective penalty + reported below
        unmeasured_specs = [k for k in self.specs if k not in metrics]

        # add raw sim artifacts + Oracle agreement (analytical vs SPICE)
        metrics["_gain_db"] = sim_res.gain_db
        metrics["_pm"] = sim_res.phase_margin_deg
        # Oracle: if SPICE available, compare to analytical synthesized
        try:
            analytical = self.bridge._analytical_cmos_inverter_chain(str(params))
            oracle_res = self.oracle.merge(analytical, sim_res, specs=self.specs)
            metrics["oracle_agreement"] = 1.0 if oracle_res.agreement else 0.0
            metrics["oracle_tool"] = 1.0 if oracle_res.tool_used == "ngspice" else 0.0
        except Exception:
            pass

        obj = compute_objective(metrics, self.specs)  # lower is better

        # Silicon area penalty: sum(W_i * L_i) for transistor pairs
        # Encourages compact designs, prevents unbounded W/L growth
        area = 0.0
        for i, name in enumerate(self.names):
            if name.startswith('W') and i + 1 < len(self.names) and self.names[i + 1].startswith('L'):
                w = params.get(name, 0)
                l = params.get(self.names[i + 1], 0)
                area += abs(w) * abs(l)
        # Normalize area penalty: 1 unit per 100 um^2, weight = 0.1
        area_penalty = 0.1 * (area / 100.0) if area > 0 else 0.0

        obj += area_penalty

        # map to 0-100 score: 100 if obj==0, decays exponentially
        score = 100.0 * (1.0 / (1.0 + obj)) if obj != 0 else 100.0
        # alternative: 100 - obj*50 clipped
        score = round(float(score), 4)
        scale = (4.5 / 5.0) * max(0.5, 1.0 - 0.003 * (85.0 - 25.0))
        hold_metrics = dict(metrics)
        for k in self.specs:
            if isinstance(hold_metrics.get(k), (int, float)):
                hold_metrics[k] = float(hold_metrics[k]) * scale
        passed_holdout = bool(self.specs) and not unmeasured_specs and all(
            (hold_metrics.get(k) is not None)
            and (hold_metrics[k] > tgt if op == ">" else hold_metrics[k] < tgt)
            for k, (op, tgt) in self.specs.items()
        )
        if sim_res.tool_used != "ngspice":
            passed_holdout = False
        return FitnessResult(
            score=score,
            passed_holdout=bool(passed_holdout),
            sub_scores={"objective": round(obj, 4), "area_penalty": round(area_penalty, 4)},
            artifacts={
                **metrics,
                "objective": obj,
                "metric_source": dict(metric_source),
                "unmeasured_specs": list(unmeasured_specs),
                "area_um2": round(area, 2),
                "area_penalty": round(area_penalty, 4),
                "tool_used": sim_res.tool_used or "analytical_fallback",
                "holdout_scale": scale,
                "circuit": str(self.circuit_file),
                "params": params,
            },
        )
