"""
inputs package — Advanced engineering input parsers for the darwin-evolab electronics track.
"""
from .boolean_expr import BooleanExpressionParser, BooleanParseResult, parse_boolean_spec
from .verilog_reader import VerilogModuleSpec, VerilogRTLReader, parse_verilog_spec
from .analog_spec import AmplifierSpec, FilterSpec, WaveformTraceSpec, parse_analog_spec
from .objective_matrix import ObjectiveMatrix

__all__ = [
    "BooleanExpressionParser",
    "BooleanParseResult",
    "parse_boolean_spec",
    "VerilogModuleSpec",
    "VerilogRTLReader",
    "parse_verilog_spec",
    "FilterSpec",
    "AmplifierSpec",
    "WaveformTraceSpec",
    "parse_analog_spec",
    "ObjectiveMatrix",
]
