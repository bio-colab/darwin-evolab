"""Timer 555 model."""
from pathlib import Path
from ...models.ngspice_bridge import NGSpiceBridge
CIRCUIT_PATH = Path(__file__).parent.parent.parent / "circuits" / "timer_555_astable" / "circuit.cir"
__all__ = ["NGSpiceBridge", "CIRCUIT_PATH"]
