"""Ideal virtual scope: measurements on a sampled voltage trace."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def measure_waveform(
    voltage: Sequence[float],
    sample_rate_hz: float,
    mid: float | None = None,
) -> dict[str, Any]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    v = [float(x) for x in voltage]
    n = len(v)
    if n < 2:
        return {"error": "short_trace", "n": n}

    vmax = max(v)
    vmin = min(v)
    vpp = vmax - vmin
    vavg = sum(v) / n
    vrms = math.sqrt(sum(x * x for x in v) / n)
    level = vmin + vpp / 2.0 if mid is None else mid

    crossings: list[int] = []
    for i in range(1, n):
        a, b = v[i - 1] - level, v[i] - level
        if a <= 0 < b:
            crossings.append(i)

    periods = []
    for i in range(1, len(crossings)):
        dt = (crossings[i] - crossings[i - 1]) / sample_rate_hz
        if dt > 0:
            periods.append(dt)
    period = sum(periods) / len(periods) if periods else 0.0
    freq = 1.0 / period if period else 0.0
    jitter = 0.0
    if len(periods) > 1:
        mean_p = period
        jitter = math.sqrt(sum((p - mean_p) ** 2 for p in periods) / len(periods))

    high_time = sum(1 for x in v if x >= level)
    duty = high_time / n if n else 0.0

    lo = vmin + 0.1 * vpp
    hi = vmin + 0.9 * vpp
    rise = _edge_time(v, sample_rate_hz, lo, hi)
    fall = _edge_time(v, sample_rate_hz, hi, lo)

    cycles = len(periods)
    rel_jit = (jitter / period) if period > 0 else 1.0
    if cycles <= 0:
        conf = 0.0
    else:
        conf = min(1.0, cycles / 8.0) * math.exp(-min(rel_jit, 3.0))
    return {
        "vmax": round(vmax, 6),
        "vmin": round(vmin, 6),
        "vpp": round(vpp, 6),
        "vrms": round(vrms, 6),
        "vavg": round(vavg, 6),
        "frequency_hz": round(freq, 6),
        "period_s": round(period, 9),
        "duty_cycle": round(duty, 6),
        "rise_time_s": rise,
        "fall_time_s": fall,
        "zero_crossings": len(crossings),
        "jitter_s": round(jitter, 9),
        "cycles_used": cycles,
        "frequency_confidence": round(conf, 4),
        "waveform_quality": round(conf, 4),
        "sample_rate_hz": sample_rate_hz,
        "n": n,
        "mode": "ideal",
    }


def _edge_time(v: list[float], fs: float, start: float, end: float) -> float | None:
    rising = end > start
    i0 = None
    for i, x in enumerate(v):
        hit = x >= start if rising else x <= start
        if hit:
            i0 = i
            break
    if i0 is None:
        return None
    for j in range(i0, len(v)):
        hit = v[j] >= end if rising else v[j] <= end
        if hit:
            return round((j - i0) / fs, 9)
    return None


def measure_transient(artifact, signal=None):
    """Time-domain scope measurement on a transient artifact.

    ngspice prints adaptive-timestep tables: samples cluster around switching
    edges, so sample-index arithmetic is NOT time arithmetic (assuming a
    uniform grid from the first few intervals inflates frequency and skews
    duty). Every interval here is therefore measured in true seconds with
    linear interpolation at the level crossings. Key set is identical to
    measure_waveform so callers can consume either interchangeably.
    """
    t = tuple(getattr(artifact, "t", ()) or ())
    signals = dict(getattr(artifact, "signals", {}) or {})
    meta = dict(getattr(artifact, "metadata", {}) or {})
    tool = getattr(artifact, "tool_used", "none")
    if len(t) < 2 or not signals:
        return {"error": "no_waveform", "tool_used": tool, **{k: meta[k] for k in meta if k != "signals"}}
    name = signal if signal in signals else next(iter(signals))
    tt = [float(x) for x in t]
    v = [float(x) for x in signals[name]]
    n = len(v)
    if n != len(tt) or n < 2:
        return {"error": "short_trace", "tool_used": tool, "signal": name, "n": n}
    total = tt[-1] - tt[0]
    if total <= 0:
        return {"error": "bad_timebase", "tool_used": tool, "signal": name}

    vmax = max(v)
    vmin = min(v)
    vpp = vmax - vmin
    level = vmin + vpp / 2.0

    # Rising crossings of `level`, linearly interpolated in true time.
    crossings: list[float] = []
    for i in range(1, n):
        a, b = v[i - 1] - level, v[i] - level
        if a <= 0 < b:
            frac = (-a) / (b - a)
            crossings.append(tt[i - 1] + frac * (tt[i] - tt[i - 1]))

    periods = [crossings[i] - crossings[i - 1] for i in range(1, len(crossings))]
    period = sum(periods) / len(periods) if periods else 0.0
    freq = 1.0 / period if period > 0 else 0.0
    jitter = 0.0
    if len(periods) > 1:
        jitter = math.sqrt(sum((p - period) ** 2 for p in periods) / len(periods))

    # Time-weighted duty: integral of the time the signal spends >= level.
    high_time = 0.0
    for i in range(1, n):
        a, b = v[i - 1], v[i]
        dt = tt[i] - tt[i - 1]
        if dt <= 0:
            continue
        if a >= level and b >= level:
            high_time += dt
        elif a >= level and b < level:
            frac = (a - level) / (a - b) if a != b else 0.0
            high_time += dt * frac
        elif a < level and b >= level:
            frac = (level - a) / (b - a) if b != a else 1.0
            high_time += dt * (1.0 - frac)
    duty = high_time / total

    # Time-weighted mean and rms (trapezoidal over the true timebase).
    vavg = 0.0
    vrms_acc = 0.0
    for i in range(1, n):
        dt = tt[i] - tt[i - 1]
        if dt <= 0:
            continue
        vavg += 0.5 * (v[i - 1] + v[i]) * dt
        vrms_acc += 0.5 * (v[i - 1] ** 2 + v[i] ** 2) * dt
    vavg /= total
    vrms = math.sqrt(vrms_acc / total) if total > 0 else 0.0

    lo = vmin + 0.1 * vpp
    hi = vmin + 0.9 * vpp
    rise = _edge_time_t(tt, v, lo, hi)
    fall = _edge_time_t(tt, v, hi, lo)

    cycles = len(periods)
    rel_jit = (jitter / period) if period > 0 else 1.0
    if cycles <= 0:
        conf = 0.0
    else:
        conf = min(1.0, cycles / 8.0) * math.exp(-min(rel_jit, 3.0))
    return {
        "vmax": round(vmax, 6),
        "vmin": round(vmin, 6),
        "vpp": round(vpp, 6),
        "vrms": round(vrms, 6),
        "vavg": round(vavg, 6),
        "frequency_hz": round(freq, 6),
        "period_s": round(period, 9),
        "duty_cycle": round(duty, 6),
        "rise_time_s": rise,
        "fall_time_s": fall,
        "zero_crossings": len(crossings),
        "jitter_s": round(jitter, 9),
        "cycles_used": cycles,
        "frequency_confidence": round(conf, 4),
        "waveform_quality": round(conf, 4),
        "sample_rate_hz": round(n / total, 6),
        "n": n,
        "signal": name,
        "tool_used": tool,
        "mode": "time_domain",
    }


def _edge_time_t(
    tt: list[float], v: list[float], start: float, end: float
) -> float | None:
    """First start->end level transition, interpolated on the true timebase."""
    rising = end > start
    t0 = None
    for i, x in enumerate(v):
        hit = x >= start if rising else x <= start
        if hit:
            t0 = tt[i]
            break
    if t0 is None:
        return None
    for j in range(1, len(v)):
        crossed = (
            v[j - 1] < end <= v[j] if rising else v[j - 1] > end >= v[j]
        )
        if not crossed or tt[j] <= t0:
            continue
        denom = v[j] - v[j - 1]
        frac = (end - v[j - 1]) / denom if denom else 0.0
        t_hit = tt[j - 1] + frac * (tt[j] - tt[j - 1])
        if t_hit <= t0:
            continue
        return round(t_hit - t0, 9)
    return None


def attach_scope(artifacts: dict[str, Any]) -> dict[str, Any]:
    """If artifacts already carry a waveform, add scope measurements. No synthesis."""
    wf = artifacts.get("waveform")
    fs = artifacts.get("sample_rate_hz")
    if not wf or not fs:
        return artifacts
    artifacts = dict(artifacts)
    artifacts["scope"] = measure_waveform(wf, float(fs))
    return artifacts
