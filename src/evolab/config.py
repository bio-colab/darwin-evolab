"""Configuration models for EvolutionEngine and modular GA components."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass
class SpeciationConfig:
    """Configuration for speciation distance and threshold dynamics."""
    enabled: bool = True
    threshold: float = 0.65
    c1: float = 0.6
    c2: float = 0.4
    c3: float = 0.0
    metric: str = "composite"  # "composite", "euclidean", "maxdelta"


@dataclass
class QualityDiversityConfig:
    """Configuration for MAP-Elites behavioral archive."""
    enabled: bool = True
    grid_x: int = 8
    grid_y: int = 6
    k: int = 16  # Target archive resolution / capacity
    scale_x: float = 10.0
    scale_y: float = 2.5
    active_selection: bool = False  # If True, sample parents from diverse archive cells


@dataclass
class MemoryConfig:
    """Configuration for temporal memory and environmental change detection."""
    enabled: bool = False
    max_size: int = 1000  # Max capacity for solution memory buffer
    max_injection_rate: float = 0.15
    change_window: int = 20
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    staleness_tau: float = 100.0
    quarantine_gens: int = 5


@dataclass
class EngineConfig:
    """Comprehensive, clean configuration for EvolutionEngine."""
    # Core population & budget parameters
    population_size: int = 16
    generations: int = 100
    elite_count: int = 2
    mutation_rate: float = 0.15
    mutation_boost: float = 1.0
    early_stop_fitness: float | None = None
    seed: int | None = None
    genome_size: int = 16
    stagnation_patience: int = 15
    immigrant_fraction: float = 0.0
    fitness_range: tuple[float, float] = (0.0, 100.0)
    allow_1d: bool = False

    # Fitness sharing & scheduling
    fitness_sharing: bool = True
    sharing_mode: str = "dynamic"  # "dynamic", "static", "off"
    exploit_after_frac: float = 2.0 / 3.0
    hybrid_light_share: float = 0.7
    mutation_enabled: bool = True
    crossover_rate: float = 0.8
    local_search_steps: int = 0
    crossover_mode: str = "single_point"  # "single_point", "blend", "uniform"

    # Sub-component configurations
    speciation: SpeciationConfig = field(default_factory=SpeciationConfig)
    qd: QualityDiversityConfig = field(default_factory=QualityDiversityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Evaluation & constraints
    eval_repeats: int = 1
    stability_penalty: float = 0.0
    hard_constraints: Sequence[Callable] = field(default_factory=tuple)
    record_population_snapshots: bool = False
    record_archive_solutions: bool = False
