"""evolab.silicon — First-class Silicon Engineering & EDA Driver for Darwin-Evolab.

Features:
- SkyWater 130nm Open-Source PDK device parameters, models, and PVT corners.
- Two-Stage Miller-Compensated CMOS Operational Amplifier benchmark with 4-objective NSGA-II Pareto optimization.
- Ultra-fast SpiceNeuralSurrogate active learning engine for 15x simulation speedup.
- Yosys RTL synthesis verification bridge and area/gate comparative reports.
"""
from __future__ import annotations

from .sky130_pdk import (
    CORNER_SPECS,
    SKY130_PARAMS,
    CornerSpec,
    Sky130Corner,
    Sky130DeviceParams,
    TransistorSmallSignal,
    compute_transistor_operating_point,
    generate_sky130_spice_header,
)
from .opamp_benchmark import (
    OpAmpPerformanceMetrics,
    OpAmpSizing,
    Sky130OpAmpAdapter,
    TwoStageMillerOpAmpEvaluator,
    evaluate_opamp_analytical,
    generate_opamp_spice_netlist,
)
from .surrogate import (
    ActiveSpiceSurrogateEvaluator,
    MicroMLP,
    SpiceNeuralSurrogate,
    SurrogateTrainingSample,
)
from .yosys_bridge import (
    YosysCellReport,
    YosysComparisonReport,
    YosysSynthesisBridge,
)
from .grammar_guard import (
    GrammarCheckResult,
    PhysicalGrammarGuard,
)
from .spec_conditioned_prior import (
    BENCHMARK_SPECS,
    CKTBENCH_BALANCED,
    CKTBENCH_HIGH_GAIN,
    CKTBENCH_HIGH_SPEED,
    CKTBENCH_LOW_POWER,
    AnalyticalSpecInverter,
    SpecConditionedGenerator,
    SpecConditionedPrior,
    SpecTolerance,
    TargetCircuitSpec,
)
from .bandit_latent_optimizer import (
    BanditArm,
    BanditLatentOptimizer,
)
from .pvt_evaluator import (
    ConservativeHindsightReplay,
    HindsightExperience,
    MultiCornerEvaluationResult,
    PVTAwareOpAmpEvaluator,
)
from .physics_mutator import (
    CircuitDiagnosis,
    PhysicsInformedOpAmpMutator,
)
from .modular_circuit import (
    ActiveLoadBlock,
    ActiveLoadType,
    BiasBlock,
    BiasType,
    CompensationBlock,
    CompensationType,
    DiffPairBlock,
    DiffPairType,
    ModularOpAmpCircuit,
    OutputStageBlock,
    OutputStageType,
    TailCurrentBlock,
    TailCurrentType,
)

__all__ = [
    # SkyWater 130nm PDK
    "Sky130Corner",
    "CornerSpec",
    "CORNER_SPECS",
    "Sky130DeviceParams",
    "SKY130_PARAMS",
    "TransistorSmallSignal",
    "compute_transistor_operating_point",
    "generate_sky130_spice_header",
    # Two-Stage Miller OpAmp
    "OpAmpSizing",
    "OpAmpPerformanceMetrics",
    "evaluate_opamp_analytical",
    "generate_opamp_spice_netlist",
    "TwoStageMillerOpAmpEvaluator",
    "Sky130OpAmpAdapter",
    # Neural Surrogate & Active Learning
    "MicroMLP",
    "SpiceNeuralSurrogate",
    "SurrogateTrainingSample",
    "ActiveSpiceSurrogateEvaluator",
    # Yosys RTL Synthesis Bridge
    "YosysCellReport",
    "YosysComparisonReport",
    "YosysSynthesisBridge",
    # ARCS Physical Grammar Guard
    "PhysicalGrammarGuard",
    "GrammarCheckResult",
    # CktGen Spec-Conditioned Generative Prior
    "TargetCircuitSpec",
    "SpecTolerance",
    "AnalyticalSpecInverter",
    "SpecConditionedGenerator",
    "SpecConditionedPrior",
    "BENCHMARK_SPECS",
    "CKTBENCH_LOW_POWER",
    "CKTBENCH_BALANCED",
    "CKTBENCH_HIGH_GAIN",
    "CKTBENCH_HIGH_SPEED",
    # Test-Time Bandit Latent Optimizer
    "BanditArm",
    "BanditLatentOptimizer",
    # PPAAS Distillation: Multi-Corner PVT & Hindsight Replay
    "PVTAwareOpAmpEvaluator",
    "MultiCornerEvaluationResult",
    "ConservativeHindsightReplay",
    "HindsightExperience",
    # AnalogCoder-Pro Distillation: Physics-Informed Mutator
    "PhysicsInformedOpAmpMutator",
    "CircuitDiagnosis",
    # CircuitGenome Distillation: Modular Blocks & Topologies
    "ModularOpAmpCircuit",
    "DiffPairBlock",
    "DiffPairType",
    "ActiveLoadBlock",
    "ActiveLoadType",
    "TailCurrentBlock",
    "TailCurrentType",
    "CompensationBlock",
    "CompensationType",
    "OutputStageBlock",
    "OutputStageType",
    "BiasBlock",
    "BiasType",
]
