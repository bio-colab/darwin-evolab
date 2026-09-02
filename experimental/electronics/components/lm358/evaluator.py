"""LM358 evaluator - re-export of CircuitConfigEvaluator for ptm op-amp proxy."""
from pathlib import Path
from ...evaluators.spice_evaluator import CircuitConfigEvaluator

CONFIG_PATH = Path(__file__).parent.parent.parent / "circuits" / "ptm180nm_opamp" / "config.json"

def get_evaluator() -> CircuitConfigEvaluator:
    return CircuitConfigEvaluator(CONFIG_PATH)

__all__ = ["CircuitConfigEvaluator", "get_evaluator", "CONFIG_PATH"]
