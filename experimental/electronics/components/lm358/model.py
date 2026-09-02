"""LM358 model - delegates to ptm180nm_opamp netlist as reproducible proxy.

Real LM358 netlist would be here; currently re-uses ptm180nm_opamp for low-cost.
Follows protocols/02 - swap .cir when ngspice model ready.
"""
from pathlib import Path
from ...models.ngspice_bridge import NGSpiceBridge

CIRCUIT_PATH = Path(__file__).parent.parent.parent / "circuits" / "ptm180nm_opamp" / "circuit.cir"

def get_circuit_path() -> Path:
    return CIRCUIT_PATH

__all__ = ["NGSpiceBridge", "CIRCUIT_PATH", "get_circuit_path"]
