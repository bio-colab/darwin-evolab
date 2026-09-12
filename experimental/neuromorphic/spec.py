"""
spec.py — Specification models for Drosophila connectome neuromorphic circuits.

Provides structured configuration for biological neural motifs such as the
Hassenstein-Reichardt Elementary Motion Detector (EMD) and Olfactory Glomerular
lateral inhibition networks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NeuromorphicSpec:
    """Specification model configuring biological motif synthesis."""

    motif_name: str = "hassenstein_reichardt_emd"
    num_inputs: int = 2
    num_outputs: int = 2
    sequence_length: int = 24
    max_nodes: int = 16
    target_accuracy: float = 95.0
    clocked: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.motif_name not in ("hassenstein_reichardt_emd", "olfactory_lateral_inhibition"):
            raise ValueError(
                f"Unsupported motif_name: {self.motif_name!r}. "
                "Supported: 'hassenstein_reichardt_emd', 'olfactory_lateral_inhibition'"
            )
        if self.num_inputs < 1:
            raise ValueError("num_inputs must be >= 1")
        if self.num_outputs < 1:
            raise ValueError("num_outputs must be >= 1")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        if self.sequence_length < 4:
            raise ValueError("sequence_length must be >= 4")
