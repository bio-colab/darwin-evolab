"""Comparator model - delegates to circuits/comparator_simple."""
from pathlib import Path
from ...models.ngspice_bridge import NGSpiceBridge
CIRCUIT_PATH = Path(__file__).parent.parent.parent / "circuits" / "comparator_simple" / "circuit.cir"
__all__ = ["NGSpiceBridge", "CIRCUIT_PATH"]
