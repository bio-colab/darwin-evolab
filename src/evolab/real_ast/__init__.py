"""RealAST package: 5-Layer Semantic AST Architecture for Darwin-EvoLab."""
from __future__ import annotations

from .analyzer import DeepAnalyzer, DependencyAnalyzer
from .crossover import IntelligentCrossover
from .genome import RealASTGenome
from .mutator import BalancedMutator, IntelligentMutator
from .optimizer import ASTOptimizer
from .scope import SymbolTable
from .types import CriticalityLevel, NodeMetadata, NodeType, Symbol, TypeInfo

__all__ = [
    "ASTOptimizer",
    "BalancedMutator",
    "CriticalityLevel",
    "DeepAnalyzer",
    "DependencyAnalyzer",
    "IntelligentCrossover",
    "IntelligentMutator",
    "NodeMetadata",
    "NodeType",
    "RealASTGenome",
    "Symbol",
    "SymbolTable",
    "TypeInfo",
]
