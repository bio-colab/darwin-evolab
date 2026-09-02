"""DatasheetConstraintVerifier — measurement vs published limit. PASS / FAIL / UNKNOWN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..components.specs import ElectricalLimits, TimingSpec


PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Constraint:
    name: str
    op: str
    limit: float
    measure_keys: tuple[str, ...]


def _cmp(op: str, value: float, limit: float) -> bool:
    if op == ">=":
        return value >= limit
    if op == ">":
        return value > limit
    if op == "<=":
        return value <= limit
    if op == "<":
        return value < limit
    return False


class DatasheetConstraintVerifier:
    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self.constraints = list(constraints or [])

    @classmethod
    def from_electrical(cls, elec: ElectricalLimits) -> DatasheetConstraintVerifier:
        return cls(
            [
                Constraint("voh_min", ">=", elec.v_oh_min_v, ("voh", "vmax", "vout", "vout_high")),
                Constraint("vol_max", "<=", elec.v_ol_max_v, ("vol", "vmin", "vout_low")),
                Constraint("icc_max_ua", "<=", elec.icc_quiescent_max_ua, ("icc_ua", "quiescent_icc_ua")),
            ]
        )

    @classmethod
    def from_timing(cls, timing: TimingSpec, vcc: float, temp_c: float) -> DatasheetConstraintVerifier:
        limit = timing.lookup_published_tpd_ns(vcc, temp_c)
        return cls([Constraint("tpd_max_ns", "<=", limit, ("tpd_ns", "worst_delay_ns", "delay_ns"))])

    def check(self, measurements: dict[str, Any]) -> dict[str, Any]:
        rows = []
        overall = PASS
        for c in self.constraints:
            raw = None
            key_used = None
            for k in c.measure_keys:
                if k in measurements and measurements[k] is not None:
                    raw = measurements[k]
                    key_used = k
                    break
            if raw is None:
                rows.append({"name": c.name, "verdict": UNKNOWN, "limit": c.limit, "op": c.op})
                if overall == PASS:
                    overall = UNKNOWN
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                rows.append({"name": c.name, "verdict": UNKNOWN, "limit": c.limit, "op": c.op})
                if overall == PASS:
                    overall = UNKNOWN
                continue
            ok = _cmp(c.op, value, c.limit)
            verdict = PASS if ok else FAIL
            if verdict == FAIL:
                overall = FAIL
            elif verdict == UNKNOWN and overall == PASS:
                overall = UNKNOWN
            rows.append(
                {
                    "name": c.name,
                    "verdict": verdict,
                    "measured": value,
                    "key": key_used,
                    "op": c.op,
                    "limit": c.limit,
                }
            )
        return {"verdict": overall, "checks": rows}
