"""User request → CircuitSpec. Schema is the brake; no LLM call."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .models.logic import PARTS_FOR_FUNCTION
from .scenarios import list_electronics_scenarios

ALLOWED_ACTIONS = frozenset({
    "run_scenario",
    "digital",
    "analog_sizing",
    "boolean_expression",
    "verilog_rtl",
    "analog_filter",
    "analog_amplifier",
    "waveform_matching",
})
ALLOWED_ENGINES = frozenset({"ga", "greedy"})


@dataclass(frozen=True)
class CircuitSpec:
    action: str
    scenario: str | None = None
    functions_needed: tuple[str, ...] = ()
    num_inputs: int | None = None
    num_outputs: int | None = None
    engine: str = "ga"
    notes: tuple[str, ...] = ()
    expressions: dict[str, str] | None = None
    verilog_code: str | None = None
    objective: str | None = None
    waveform_csv: str | None = None


def validate_spec(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    extra = set(data) - {
        "action",
        "scenario",
        "functions_needed",
        "num_inputs",
        "num_outputs",
        "engine",
        "notes",
        "expressions",
        "verilog_code",
        "verilog_file",
        "objective",
        "filter_type",
        "cutoff_hz",
        "stopband_attenuation_db",
        "gain_db",
        "bandwidth_mhz",
        "waveform_csv",
    }
    if extra:
        errors.append(f"unknown_fields:{sorted(extra)}")
    action = data.get("action", "run_scenario")
    if action not in ALLOWED_ACTIONS:
        errors.append(f"bad_action:{action}")
    scenario = data.get("scenario")
    allowed = set(list_electronics_scenarios())
    if scenario is not None and scenario not in allowed:
        errors.append(f"unknown_scenario:{scenario}")
    engine = data.get("engine", "ga")
    if engine not in ALLOWED_ENGINES:
        errors.append(f"bad_engine:{engine}")
    needed = data.get("functions_needed") or []
    if needed and not isinstance(needed, (list, tuple)):
        errors.append("functions_needed_not_list")
    else:
        for fn in needed:
            if str(fn).upper() not in PARTS_FOR_FUNCTION:
                errors.append(f"unknown_function:{fn}")
    for key in ("num_inputs", "num_outputs"):
        if key in data and data[key] is not None:
            try:
                v = int(data[key])
            except (TypeError, ValueError):
                errors.append(f"bad_{key}")
            else:
                if v < 1 or v > 8:
                    errors.append(f"range_{key}")
    return errors


def spec_from_dict(data: dict[str, Any]) -> tuple[CircuitSpec | None, list[str]]:
    errors = validate_spec(data)
    if errors:
        return None, errors
    needed = tuple(str(x).upper() for x in (data.get("functions_needed") or ()))
    notes = data.get("notes") or ()
    if isinstance(notes, str):
        notes = (notes,)
    return (
        CircuitSpec(
            action=str(data.get("action", "run_scenario")),
            scenario=data.get("scenario"),
            functions_needed=needed,
            num_inputs=None if data.get("num_inputs") is None else int(data["num_inputs"]),
            num_outputs=None if data.get("num_outputs") is None else int(data["num_outputs"]),
            engine=str(data.get("engine", "ga")),
            notes=tuple(notes),
            expressions=data.get("expressions"),
            verilog_code=data.get("verilog_code"),
            objective=data.get("objective"),
            waveform_csv=data.get("waveform_csv"),
        ),
        [],
    )


def parse_request(text: str) -> tuple[CircuitSpec | None, list[str]]:
    """Accept JSON spec only. Free prose is rejected so an LLM cannot skip the schema."""
    raw = text.strip()
    if not raw:
        return None, ["empty"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, ["not_json"]
    if not isinstance(data, dict):
        return None, ["not_object"]
    return spec_from_dict(data)


def spec_to_prepare_args(spec: CircuitSpec) -> dict[str, Any]:
    if spec.action == "analog_sizing":
        return {"scenario": "analog_sizing"}
    if spec.scenario:
        return {"scenario": spec.scenario}
    if spec.functions_needed == ("XOR", "AND") and (spec.num_inputs in (None, 2)):
        return {"scenario": "half_adder"}
    if spec.functions_needed == ("XOR", "AND", "OR"):
        return {"scenario": "full_adder"}
    if spec.functions_needed:
        return {"scenario": None, "error": "no_scenario_for_functions"}
    return {"scenario": "half_adder"}
