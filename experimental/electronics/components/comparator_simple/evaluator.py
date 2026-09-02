"""Comparator evaluator."""
from pathlib import Path
from ...evaluators.spice_evaluator import CircuitConfigEvaluator
CONFIG_PATH = Path(__file__).parent.parent.parent / "circuits" / "comparator_simple" / "config.json"
def get_evaluator(): return CircuitConfigEvaluator(CONFIG_PATH)
__all__ = ["CircuitConfigEvaluator", "get_evaluator", "CONFIG_PATH"]
