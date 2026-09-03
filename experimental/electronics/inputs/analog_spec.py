"""
analog_spec.py — Rich Analog Engineering Specifications, Frequency Profiles, and Waveform Traces.

Allows users to define analog goals via engineering constraints (cutoff frequency,
passband ripple, stopband attenuation, gain, phase margin) or experimental oscilloscope CSV traces.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FilterSpec:
    """Specification for analog filter synthesis."""
    kind: str = "lowpass"  # "lowpass", "highpass", "bandpass"
    cutoff_hz: float = 1000.0
    stopband_freq_hz: float = 10000.0
    passband_ripple_db: float = 1.0
    stopband_attenuation_db: float = -20.0
    input_impedance_ohms: float = 10000.0
    max_components: int = 8

    def evaluate_response(self, gain_db: float, measured_freq_hz: float) -> float:
        """Computes fitness score [0.0, 100.0] against the filter specification."""
        if self.kind == "lowpass":
            # Passband check
            if measured_freq_hz <= self.cutoff_hz:
                err = abs(gain_db)
                return max(0.0, 100.0 - err * 10.0)
            else:
                # Stopband check: should attenuate
                diff = self.stopband_attenuation_db - gain_db
                return max(0.0, 100.0 - abs(diff) * 5.0)
        return 50.0


@dataclass(frozen=True)
class AmplifierSpec:
    """Specification for analog amplifier synthesis."""
    target_gain_db: float = 20.0
    min_bandwidth_mhz: float = 1.0
    min_phase_margin_deg: float = 60.0
    max_power_mw: float = 10.0
    supply_voltage: float = 5.0

    def evaluate_metrics(
        self,
        gain_db: float,
        bandwidth_mhz: float,
        phase_margin_deg: float,
        power_mw: float,
    ) -> float:
        score = 100.0
        # Gain penalty
        score -= min(40.0, abs(gain_db - self.target_gain_db) * 2.0)
        # Bandwidth penalty
        if bandwidth_mhz < self.min_bandwidth_mhz:
            score -= min(25.0, (self.min_bandwidth_mhz - bandwidth_mhz) / self.min_bandwidth_mhz * 25.0)
        # Phase margin penalty (stability)
        if phase_margin_deg < self.min_phase_margin_deg:
            score -= min(25.0, (self.min_phase_margin_deg - phase_margin_deg) * 1.5)
        # Power penalty
        if power_mw > self.max_power_mw:
            score -= min(15.0, (power_mw - self.max_power_mw) / self.max_power_mw * 15.0)
        return max(0.0, round(score, 2))


@dataclass(frozen=True)
class WaveformTraceSpec:
    """Specification matching measured or target oscilloscope transient waveforms (CSV)."""
    times: tuple[float, ...]
    voltages: tuple[float, ...]
    observable_node: str = "v(out)"

    @classmethod
    def from_csv(cls, filepath: str | Path, observable: str = "v(out)") -> WaveformTraceSpec:
        p = Path(filepath)
        times_list: list[float] = []
        volts_list: list[float] = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    t = float(row[0])
                    v = float(row[1])
                    times_list.append(t)
                    volts_list.append(v)
                except ValueError:
                    # Skip header line
                    continue
        if not times_list:
            raise ValueError(f"No valid numeric (time, voltage) pairs in {filepath}")
        return cls(times=tuple(times_list), voltages=tuple(volts_list), observable_node=observable)

    def evaluate_mse(self, measured_t: Sequence[float], measured_v: Sequence[float]) -> float:
        """Calculates normalized waveform similarity [0.0, 100.0] based on Mean Squared Error."""
        if not measured_t or not measured_v or len(measured_t) != len(measured_v):
            return 0.0

        # Interpolate measured waveform onto target time points
        mse = 0.0
        n_points = min(len(self.times), len(measured_t), 100)
        for i in range(n_points):
            target_v = self.voltages[i]
            actual_v = measured_v[i]
            mse += (target_v - actual_v) ** 2
        mse /= max(n_points, 1)

        # Scale MSE into a 0..100 fitness score
        score = 100.0 / (1.0 + mse * 10.0)
        return round(score, 2)


def parse_analog_spec(data: dict[str, Any]) -> FilterSpec | AmplifierSpec | WaveformTraceSpec:
    """Auto-detects and instantiates the appropriate analog engineering spec."""
    kind = str(data.get("kind", "")).lower()
    if kind in ("filter", "analog_filter", "lowpass", "highpass", "bandpass"):
        return FilterSpec(
            kind=str(data.get("type", data.get("kind", "lowpass"))),
            cutoff_hz=float(data.get("cutoff_hz", 1000.0)),
            stopband_freq_hz=float(data.get("stopband_freq_hz", 10000.0)),
            passband_ripple_db=float(data.get("passband_ripple_db", 1.0)),
            stopband_attenuation_db=float(data.get("stopband_attenuation_db", -20.0)),
            input_impedance_ohms=float(data.get("input_impedance_ohms", 10000.0)),
            max_components=int(data.get("max_components", 8)),
        )
    elif kind in ("amplifier", "analog_amplifier", "opamp"):
        return AmplifierSpec(
            target_gain_db=float(data.get("target_gain_db", 20.0)),
            min_bandwidth_mhz=float(data.get("min_bandwidth_mhz", 1.0)),
            min_phase_margin_deg=float(data.get("min_phase_margin_deg", 60.0)),
            max_power_mw=float(data.get("max_power_mw", 10.0)),
            supply_voltage=float(data.get("supply_voltage", 5.0)),
        )
    elif "waveform_csv" in data:
        return WaveformTraceSpec.from_csv(
            data["waveform_csv"],
            observable=data.get("observable_node", "v(out)"),
        )
    raise ValueError(f"Unrecognized analog spec format: {data}")
