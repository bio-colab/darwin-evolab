"""Datasheet archive + live 74HC catalog.

Runnable analog circuits are under circuits/. These folders keep specs,
notes, and thin wrappers around CircuitConfigEvaluator.
"""
from .specs import LogicFunction, ElectricalLimits, TimingSpec, GateUnit, ICPackageSpec
from .catalog_74xx import CATALOG_74XX, get_ic_spec

__all__ = [
    "LogicFunction", "ElectricalLimits", "TimingSpec", "GateUnit", "ICPackageSpec",
    "CATALOG_74XX", "get_ic_spec",
]
