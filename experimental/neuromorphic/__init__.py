"""
experimental/neuromorphic — Drosophila connectome neuromorphic circuits module.

Provides biological neural motif synthesis (Hassenstein-Reichardt EMD, Olfactory
lateral inhibition) using Cartesian Genetic Programming with synthesizable
Verilog-2001 export.
"""
from .adapter import NeuromorphicAdapter
from .evaluator import NeuromorphicEvaluator
from .exporter import NeuromorphicExporter
from .genome import NeuromorphicCircuitGenome, NeuromorphicNode, NeuromorphicOp
from .motifs import generate_emd_dataset, generate_olfactory_dataset
from .spec import NeuromorphicSpec

__all__ = [
    "NeuromorphicAdapter",
    "NeuromorphicCircuitGenome",
    "NeuromorphicEvaluator",
    "NeuromorphicExporter",
    "NeuromorphicNode",
    "NeuromorphicOp",
    "NeuromorphicSpec",
    "generate_emd_dataset",
    "generate_olfactory_dataset",
]
