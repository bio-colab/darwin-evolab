"""RealASTGenome: 5-Layer Semantic AST Genome for Darwin-EvoLab."""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from ..genome import EvolabGenome
from .analyzer import DeepAnalyzer
from .crossover import IntelligentCrossover
from .mutator import IntelligentMutator
from .optimizer import ASTOptimizer
from .scope import SymbolTable
from .types import NodeMetadata, TypeInfo


@dataclass
class RealASTGenome(EvolabGenome):
    """Full-featured, semantics-aware AST genome with metadata, symbols, and type inference."""

    tree: ast.AST
    source_code: str = ""
    language: str = "python"
    metadata_cache: dict[int, NodeMetadata] = field(default_factory=dict, repr=False)
    symbol_table: SymbolTable = field(default_factory=SymbolTable, repr=False)
    type_info: TypeInfo = field(default_factory=TypeInfo, repr=False)

    def __post_init__(self):
        if not self.source_code:
            try:
                self.source_code = ast.unparse(self.tree)
            except Exception:
                self.source_code = ""
        self._analyze_tree()

    def _analyze_tree(self) -> None:
        analyzer = DeepAnalyzer()
        self.metadata_cache, self.symbol_table, self.type_info = analyzer.analyze(self.tree)

    @classmethod
    def from_code(cls, code: str, language: str = "python") -> RealASTGenome:
        tree = ast.parse(code)
        return cls(tree=tree, source_code=code, language=language)

    def to_code(self) -> str:
        if not self.source_code:
            self.source_code = ast.unparse(self.tree)
        return self.source_code

    def clone(self) -> RealASTGenome:
        tree_copy = copy.deepcopy(self.tree)
        return RealASTGenome(
            tree=tree_copy,
            source_code=self.source_code,
            language=self.language,
        )

    def fingerprint(self) -> str:
        raw_dump = ast.dump(self.tree, annotate_fields=False)
        return hashlib.sha256(raw_dump.encode("utf-8")).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, RealASTGenome):
            raise TypeError(f"Cannot compute distance between RealASTGenome and {type(other)}")

        code_a = self.to_code()
        code_b = other.to_code()
        matcher = difflib.SequenceMatcher(None, code_a, code_b)
        text_dist = 1.0 - matcher.ratio()

        # Semantic distance based on shared symbol and node count
        syms_a = set(self.symbol_table.symbols.keys())
        syms_b = set(other.symbol_table.symbols.keys())
        union_syms = syms_a | syms_b
        sym_dist = 1.0 - (len(syms_a & syms_b) / max(1, len(union_syms)))

        return round(0.6 * text_dist + 0.4 * sym_dist, 6)

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "RealASTGenome",
            "fingerprint": self.fingerprint(),
            "code": self.to_code(),
            "language": self.language,
            "symbols": [s.name for s in self.symbol_table.get_all_symbols_in_scope("global")],
            "node_count": len(self),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "node_count": len(self),
            "symbol_count": sum(len(v) for v in self.symbol_table.symbols.values()),
            "scopes": list(self.symbol_table.scopes.keys()),
            "critical_nodes": sum(1 for m in self.metadata_cache.values() if m.is_critical),
            "fingerprint": self.fingerprint(),
        }

    def __len__(self) -> int:
        return sum(1 for _ in ast.walk(self.tree))

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> RealASTGenome:
        suspicion_map = kwargs.get("suspicion_map")
        if suspicion_map is not None:
            from ..suspicion import FaultGuidedASTMutator
            tree, _desc = FaultGuidedASTMutator(rng=rng).mutate(self.tree, suspicion_map)
            if tree is not None:
                try:
                    return RealASTGenome(
                        tree=tree,
                        source_code=ast.unparse(tree),
                        language=self.language,
                    )
                except Exception:
                    pass
        mutator = IntelligentMutator(self, rng=rng)
        return mutator.mutate()

    def crossover(
        self, other: EvolabGenome, rng: random.Random | None = None, **kwargs: Any
    ) -> RealASTGenome:
        if not isinstance(other, RealASTGenome):
            return self.clone()
        crossover_op = IntelligentCrossover(rng=rng)
        child_a, _ = crossover_op.crossover(self, other)
        return child_a

    def optimize(self) -> RealASTGenome:
        optimizer = ASTOptimizer()
        return optimizer.optimize(self)
