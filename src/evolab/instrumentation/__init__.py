"""
instrumentation package — Modular Target Instrumentation, Security Oracles, and Transfer Memory.
"""
from __future__ import annotations

from .invariants import (
    BoundedExecutionTimeInvariant,
    ExpectedBehaviorInvariant,
    HoldoutGeneralizationInvariant,
    Invariant,
    LatencyDilationInvariant,
    NoEnvironmentIntrospectionInvariant,
    NoMemorySafetyViolationInvariant,
    NoPrivilegeBoundaryViolationInvariant,
    NoResourceExhaustionInvariant,
    NoStateCorruptionInvariant,
    Observation,
    OracleVerdict,
    Target,
    Violation,
    ViolationSeverity,
)
from .oracles import (
    BehavioralDelta,
    DifferentialEvaluator,
    DifferentialExecutor,
    SecurityEvaluator,
    SecurityOracle,
)
from .taxonomy import (
    FaultCategory,
    classify_fault,
)
from .symbolic_guard import (
    NumericalPreFilter,
    StructuralProofGuard,
    SymbolicGuard,
)
from .transfer import (
    BehavioralSecurityDescriptor,
    CrossTargetTransferMemory,
    InvariantBehavioralDescriptor,
    MemoryArena,
    PermutationSymmetricInputSummary,
    TransferableBehavioralMotif,
    TransferablePrimitive,
)

__all__ = [
    "BehavioralDelta",
    "BehavioralSecurityDescriptor",
    "BoundedExecutionTimeInvariant",
    "CrossTargetTransferMemory",
    "DifferentialEvaluator",
    "DifferentialExecutor",
    "ExpectedBehaviorInvariant",
    "FaultCategory",
    "HoldoutGeneralizationInvariant",
    "Invariant",
    "InvariantBehavioralDescriptor",
    "LatencyDilationInvariant",
    "MemoryArena",
    "NoEnvironmentIntrospectionInvariant",
    "NoMemorySafetyViolationInvariant",
    "NoPrivilegeBoundaryViolationInvariant",
    "NoResourceExhaustionInvariant",
    "NoStateCorruptionInvariant",
    "NumericalPreFilter",
    "Observation",
    "OracleVerdict",
    "PermutationSymmetricInputSummary",
    "SecurityEvaluator",
    "SecurityOracle",
    "StructuralProofGuard",
    "SymbolicGuard",
    "Target",
    "TransferableBehavioralMotif",
    "TransferablePrimitive",
    "Violation",
    "ViolationSeverity",
    "classify_fault",
]
