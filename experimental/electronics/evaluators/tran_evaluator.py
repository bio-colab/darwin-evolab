"""555 Timer Evaluators - Two-Level Approach.

Level A: Hybrid (Analytical + RC Sanity Check) - Fast, deterministic baseline
Level B: True Transient (Behavioral Model + SPICE Measurement) - Physics-accurate benchmark
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import FloatGenome, Individual

from ..instruments.oscilloscope import measure_transient, measure_waveform
from ..models.ngspice_bridge import NGSpiceBridge


RC_CIR = Path(__file__).resolve().parents[1] / "circuits" / "timer_555_astable" / "rc_sanity.cir"
BEHAVIORAL_CIR = Path(__file__).resolve().parents[1] / "circuits" / "timer_555_astable" / "astable_behavioral.cir"


class Timer555HybridEvaluator(Evaluator):
    """
    Level A: Hybrid evaluator for NE555 astable.
    
    - Frequency & Duty Cycle: Calculated analytically from datasheet formulas.
    - Vpp & Voltage Limits: Verified via ngspice transient on a simple RC circuit.
    
    This is a FAST baseline benchmark. It does NOT claim physical evidence for
    frequency/duty cycle, only for voltage levels.
    
    Use case: Quick optimization, algorithm comparison baseline.
    """
    
    def __init__(
        self,
        ngspice_path: str,
        target_freq_hz: float = 1000.0,
        target_vpp: float = 5.0,
        target_duty: float = 0.5,
        vcc: float = 5.0,
    ) -> None:
        super().__init__()
        self.bridge = NGSpiceBridge(ngspice_path=ngspice_path)
        self.target_freq_hz = target_freq_hz
        self.target_vpp = target_vpp
        self.target_duty = target_duty
        self.vcc = vcc
        self.defaults = [10000.0, 10000.0, 1e-7]

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, **kwargs: Any) -> FitnessResult:
        genome = target.genome if isinstance(target, Individual) else target
        if not isinstance(genome, FloatGenome) or len(genome.values) < 3:
            return FitnessResult(0.0, False, artifacts={"error": "need_3_params"})
        
        r1, r2, c1 = (abs(float(v)) for v in genome.values[:3])
        # Enforce bounds
        r1 = min(max(r1, 2e3), 50e3)
        r2 = min(max(r2, 2e3), 50e3)
        c1 = min(max(c1, 20e-9), 500e-9)
        
        # === STEP 1: Analytical Calculation (Datasheet Formulas) ===
        # f = 1.44 / ((R1 + 2*R2) * C1)
        # Duty = (R1 + R2) / (R1 + 2*R2)
        freq_analytical = 1.44 / ((r1 + 2.0 * r2) * c1)
        duty_analytical = (r1 + r2) / (r1 + 2.0 * r2)
        
        # === STEP 2: ngspice Sanity Check (RC Circuit) ===
        art = self.bridge.run_transient_file(
            RC_CIR, 
            {"R1": r1, "R2": r2, "C1": c1, "VCC": self.vcc}, 
            timeout_sec=5.0
        )
        
        vpp_measured = self.vcc
        voltage_valid = True
        
        if art.success and art.tool_used == "ngspice":
            meas = measure_transient(art, signal="v(out_node)")
            vmax = float(meas.get("vmax", 0))
            vmin = float(meas.get("vmin", 0))
            vpp_measured = vmax - vmin if vmax > 0 else self.vcc
            
            if vpp_measured > 0:
                voltage_valid = (vmin >= -0.5) and (vmax <= self.vcc + 0.5)
        
        # === STEP 3: Fitness Scoring ===
        freq_s = _log_proximity(freq_analytical, self.target_freq_hz)
        vpp_s = math.exp(-abs(vpp_measured - self.target_vpp) / 0.5)
        duty_s = math.exp(-abs(duty_analytical - self.target_duty) / 0.05)
        
        quality = 1.0 if voltage_valid else 0.1
        
        score = 100.0 * freq_s * vpp_s * duty_s * quality
        
        close = (
            freq_analytical > 0
            and abs(math.log(freq_analytical / self.target_freq_hz)) < math.log(1.01)
            and abs(vpp_measured - self.target_vpp) / self.target_vpp <= 0.02
            and abs(duty_analytical - self.target_duty) <= 0.02
        )
        
        return FitnessResult(
            score=round(score, 4),
            passed_holdout=bool(close),
            sub_scores={
                "freq": round(freq_s * 100, 2),
                "vpp": round(vpp_s * 100, 2),
                "duty": round(duty_s * 100, 2),
                "quality": round(quality * 100, 2),
            },
            artifacts={
                "tool_used": "hybrid_analytical_ngspice",
                "physical_claim": False,
                "benchmark_level": "A_hybrid_fast",
                "frequency_hz": round(freq_analytical, 6),
                "vpp": round(vpp_measured, 6),
                "duty_cycle": round(duty_analytical, 6),
                "voltage_valid": voltage_valid,
                "methodology": {
                    "frequency_source": "analytical_datasheet_formula",
                    "duty_source": "analytical_datasheet_formula",
                    "voltage_source": "ngspice_rc_sanity_check"
                }
            },
        )


def _log_proximity(value: float, target: float) -> float:
    if value <= 0 or target <= 0:
        return 0.0
    return math.exp(-abs(math.log(value / target)))


class Timer555TrueTransientEvaluator(Evaluator):
    """
    Level B: True Transient evaluator for NE555 astable.
    
    - Frequency, Duty Cycle, Vpp: ALL measured from ngspice transient simulation
      of a behavioral 555 oscillator model.
    - No analytical formulas used in fitness calculation.
    - Formulas are ONLY used as reference oracle for validation.
    
    This is the PHYSICS-ACCURATE benchmark. It claims physical evidence for
    all metrics because they emerge from the simulated circuit behavior.
    
    Use case: Final validation, physics-based algorithm comparison.
    """
    
    def __init__(
        self,
        ngspice_path: str,
        target_freq_hz: float = 1000.0,
        target_vpp: float = 5.0,
        target_duty: float = 0.5,
        vcc: float = 5.0,
    ) -> None:
        super().__init__()
        self.bridge = NGSpiceBridge(ngspice_path=ngspice_path)
        self.target_freq_hz = target_freq_hz
        self.target_vpp = target_vpp
        self.target_duty = target_duty
        self.vcc = vcc
        self.defaults = [10000.0, 10000.0, 1e-7]

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, **kwargs: Any) -> FitnessResult:
        genome = target.genome if isinstance(target, Individual) else target
        if not isinstance(genome, FloatGenome) or len(genome.values) < 3:
            return FitnessResult(0.0, False, artifacts={"error": "need_3_params"})
        
        r1, r2, c1 = (abs(float(v)) for v in genome.values[:3])
        # Enforce bounds
        r1 = min(max(r1, 2e3), 50e3)
        r2 = min(max(r2, 2e3), 50e3)
        c1 = min(max(c1, 20e-9), 500e-9)
        
        # === STEP 1: Reference Calculation (Datasheet Formulas) - FOR VALIDATION ONLY ===
        freq_reference = 1.44 / ((r1 + 2.0 * r2) * c1)
        duty_reference = (r1 + r2) / (r1 + 2.0 * r2)
        
        # === STEP 2: ngspice Transient Simulation (Behavioral Model) ===
        art = self.bridge.run_transient_file(
            BEHAVIORAL_CIR, 
            {"R1": r1, "R2": r2, "C1": c1, "VCC": self.vcc}, 
            timeout_sec=10.0
        )
        
        # Default values if simulation fails
        freq_measured = 0.0
        duty_measured = 0.0
        vpp_measured = 0.0
        voltage_valid = False
        waveform_quality = 0.0
        cycles_used = 0
        frequency_confidence = 0.0
        
        if art.success and art.tool_used == "ngspice":
            meas = measure_transient(art, signal="v(out)")
            
            freq_measured = float(meas.get("frequency_hz", 0))
            duty_measured = float(meas.get("duty_cycle", 0))
            vmax = float(meas.get("vmax", 0))
            vmin = float(meas.get("vmin", 0))
            vpp_measured = vmax - vmin if vmax > 0 else 0.0
            
            cycles_used = int(meas.get("cycles_used", 0))
            frequency_confidence = float(meas.get("frequency_confidence", 0))
            waveform_quality = float(meas.get("waveform_quality", 0))
            
            if vpp_measured > 0:
                voltage_valid = (vmin >= -0.5) and (vmax <= self.vcc + 0.5)
        
        # === STEP 3: Fitness Scoring (Based SOLELY on Measured Values) ===
        # If measurement failed or confidence is low, score is penalized heavily
        if freq_measured <= 0 or frequency_confidence < 0.3:
            return FitnessResult(
                score=0.0,
                passed_holdout=False,
                sub_scores={"freq": 0.0, "vpp": 0.0, "duty": 0.0, "quality": 0.0},
                artifacts={
                    "tool_used": "true_transient_ngspice",
                    "physical_claim": False,
                    "benchmark_level": "B_true_transient",
                    "failure_reason": "measurement_failed_or_low_confidence",
                    "frequency_hz": freq_measured,
                    "duty_cycle": duty_measured,
                    "vpp": vpp_measured,
                    "frequency_confidence": frequency_confidence,
                    "cycles_used": cycles_used,
                }
            )
        
        freq_s = _log_proximity(freq_measured, self.target_freq_hz)
        vpp_s = math.exp(-abs(vpp_measured - self.target_vpp) / 0.5)
        duty_s = math.exp(-abs(duty_measured - self.target_duty) / 0.05)
        
        quality = waveform_quality if waveform_quality > 0 else (1.0 if voltage_valid else 0.1)
        
        score = 100.0 * freq_s * vpp_s * duty_s * quality
        
        close = (
            abs(math.log(freq_measured / self.target_freq_hz)) < math.log(1.01)
            and abs(vpp_measured - self.target_vpp) / self.target_vpp <= 0.02
            and abs(duty_measured - self.target_duty) <= 0.02
            and voltage_valid
        )
        
        return FitnessResult(
            score=round(score, 4),
            passed_holdout=bool(close),
            sub_scores={
                "freq": round(freq_s * 100, 2),
                "vpp": round(vpp_s * 100, 2),
                "duty": round(duty_s * 100, 2),
                "quality": round(quality * 100, 2),
            },
            artifacts={
                "tool_used": "true_transient_ngspice",
                "physical_claim": True,
                "benchmark_level": "B_true_transient",
                "frequency_hz": round(freq_measured, 6),
                "vpp": round(vpp_measured, 6),
                "duty_cycle": round(duty_measured, 6),
                "voltage_valid": voltage_valid,
                "cycles_used": cycles_used,
                "frequency_confidence": round(frequency_confidence, 4),
                "waveform_quality": round(waveform_quality, 4),
                "validation": {
                    "formula_freq_hz": round(freq_reference, 6),
                    "formula_duty": round(duty_reference, 6),
                    "freq_error_percent": abs(freq_measured - freq_reference) / freq_reference * 100 if freq_reference > 0 else None,
                    "duty_error_percent": abs(duty_measured - duty_reference) / duty_reference * 100 if duty_reference > 0 else None,
                }
            },
        )


class Synthetic555Evaluator(Evaluator):
    """Benchmark A: clean astable model + measured synthetic wave. No SPICE."""

    def __init__(self, target_freq_hz: float = 1000.0, target_vpp: float = 5.0, target_duty: float = 0.5) -> None:
        super().__init__()
        self.target_freq_hz = target_freq_hz
        self.target_vpp = target_vpp
        self.target_duty = target_duty

    @property
    def deterministic(self) -> bool:
        return True

    def evaluate(self, target: Any, **kwargs: Any) -> FitnessResult:
        from ..instruments.oscilloscope import measure_waveform

        genome = target.genome if isinstance(target, Individual) else target
        if not isinstance(genome, FloatGenome) or len(genome.values) < 3:
            return FitnessResult(0.0, False, artifacts={"error": "need_3_params"})
        r1, r2, c1 = (abs(float(v)) for v in genome.values[:3])
        r1 = min(max(r1, 2e3), 50e3)
        r2 = min(max(r2, 2e3), 50e3)
        c1 = min(max(c1, 20e-9), 500e-9)
        freq = 1.44 / ((r1 + 2.0 * r2) * c1)
        duty = (r1 + r2) / (r1 + 2.0 * r2)
        fs = 100_000.0
        n = int(fs * 0.02)
        high = max(1, int(fs / max(freq, 1e-9) * duty))
        period = max(2, int(fs / max(freq, 1e-9)))
        wave = [5.0 if (i % period) < high else 0.0 for i in range(n)]
        meas = measure_waveform(wave, fs)
        freq_s = _log_proximity(float(meas["frequency_hz"]), self.target_freq_hz)
        vpp_s = math.exp(-abs(float(meas["vpp"]) - self.target_vpp) / 0.5)
        duty_s = math.exp(-abs(float(meas["duty_cycle"]) - self.target_duty) / 0.05)
        conf = float(meas.get("frequency_confidence") or 0.0)
        score = 100.0 * freq_s * vpp_s * duty_s * max(conf, 0.05)
        return FitnessResult(
            score=round(score, 4),
            passed_holdout=False,
            artifacts={
                "tool_used": "synthetic",
                "physical_claim": False,
                "formula_freq_hz": freq,
                "formula_duty": duty,
                "frequency_hz": meas.get("frequency_hz"),
                "vpp": meas.get("vpp"),
                "duty_cycle": meas.get("duty_cycle"),
                "cycles_used": meas.get("cycles_used"),
                "frequency_confidence": meas.get("frequency_confidence"),
                "R1": r1,
                "R2": r2,
                "C1": c1,
            },
        )

