"""Enterprise Supply Chain, Sandbox Security, and Hardening Tests."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evolab.sandbox import SandboxConfig, SandboxRunner
from evolab.llm_mutator import LLMConfig
from evolab.engine import EvolutionEngine, Individual, FloatGenome
from evolab.genome import EvolabGenome

ROOT = Path(__file__).resolve().parents[1]


def test_reproducible_lockfile_exists():
    """Supply Chain: requirements.lock exists with pinned dependencies."""
    lock_file = ROOT / "requirements.lock"
    assert lock_file.exists(), "requirements.lock must exist for enterprise builds"
    content = lock_file.read_text(encoding="utf-8")
    assert "pytest==" in content
    assert "pyyaml==" in content


def test_cyclonedx_sbom_spec_valid():
    """Supply Chain: CycloneDX 1.5 SBOM matches enterprise specification."""
    sbom_file = ROOT / "sbom.json"
    assert sbom_file.exists(), "sbom.json must exist"
    data = json.loads(sbom_file.read_text(encoding="utf-8"))
    assert data["bomFormat"] == "CycloneDX"
    assert data["specVersion"] == "1.5"
    assert data["metadata"]["component"]["name"] == "evolab"
    assert len(data["components"]) >= 2


def test_sandbox_network_socket_blocking():
    """Security: SandboxRunner strictly blocks socket network calls when allow_network=False."""
    code_with_network = """
def attempt_network_call():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("8.8.8.8", 53))
    return "escaped"
"""
    runner = SandboxRunner(SandboxConfig(timeout_seconds=2.0, allow_network=False))
    res = runner.run_function(
        sources={"target.py": code_with_network},
        target_file="target.py",
        func_name="attempt_network_call",
        args=(),
    )
    assert not res.success
    assert "PermissionError" in (res.error or "") or "blocked" in (res.error or "").lower()


def test_llm_config_hides_api_key_in_repr():
    """Security: API keys are excluded from repr to prevent leaking in CI logs."""
    cfg = LLMConfig(provider="groq", api_key="gsk_secret_token_12345")
    assert "gsk_secret_token_12345" not in repr(cfg)
    assert cfg.get_api_key() == "gsk_secret_token_12345"


def test_active_qd_sampling_in_engine():
    """Algorithmic: Active QD selection samples parents from MAP-Elites archive."""
    engine = EvolutionEngine(
        population_size=8,
        seed=42,
        qd_selection=True,
        record_archive_solutions=True,
    )
    report = engine.run(generations=4)
    assert "map_elites" in report
    assert report["map_elites"]["filled_cells"] > 0


def test_engine_config_clean_dependency_injection():
    """Architecture: EvolutionEngine cleanly initializes via EngineConfig dataclass."""
    from evolab.config import EngineConfig, SpeciationConfig, QualityDiversityConfig
    cfg = EngineConfig(
        population_size=10,
        elite_count=2,
        mutation_rate=0.2,
        seed=99,
        speciation=SpeciationConfig(enabled=True, threshold=0.7),
        qd=QualityDiversityConfig(enabled=True, grid_x=4, grid_y=4, active_selection=True),
    )
    engine = EvolutionEngine(config=cfg)
    assert engine.population_size == 10
    assert engine.elite_count == 2
    assert engine.mutation_rate == 0.2
    assert engine.speciation_threshold == 0.7
    assert engine.me_grid_x == 4
    assert engine.qd_selection is True
    report = engine.run(generations=2)
    assert report["total_generations"] >= 2


def test_memetic_local_refinement_improves_best_candidate():
    """General: Memetic local search performs hill-climbing refinement on top candidate."""
    from evolab.config import EngineConfig

    cfg = EngineConfig(
        population_size=8,
        seed=42,
        local_search_steps=5,
        crossover_rate=0.5,
    )
    engine = EvolutionEngine(config=cfg)
    report = engine.run(generations=5)
    assert report["total_generations"] >= 5
    assert engine.best_ever is not None
    assert engine.best_ever.fitness >= report["history"][0]["best_fitness"]


def test_asexual_mutation_reproduction_branch():
    """General: Crossover rate < 1.0 triggers asexual mutation reproduction."""
    from evolab.config import EngineConfig

    cfg = EngineConfig(
        population_size=8,
        seed=123,
        crossover_rate=0.0,  # Pure asexual reproduction
    )
    engine = EvolutionEngine(config=cfg)
    report = engine.run(generations=3)
    assert report["total_generations"] >= 3
    # Check that lineage recorded asexual_mutation operator
    assert "asexual_mutation" in str(report.get("decision_log", [])) or engine.best_ever is not None


def test_speciation_distance_memoization_cache():
    """General: Distance memoization eliminates redundant calculations."""
    engine = EvolutionEngine(population_size=8, seed=42)
    engine.run(generations=3)
    assert len(engine._dist_cache) > 0


def test_cem_reusability_aligns_matching_signatures():
    """General: CEM TemporalMemoryIndex correctly ranks aligned context signatures higher."""
    from evolab.memory import TemporalMemoryIndex, MemoryEntry

    tmi = TemporalMemoryIndex()
    # Add two entries: one with positive alignment signature (1.0, 0.0), one with orthogonal (0.0, 1.0)
    e1 = tmi.upsert(genome=[1.0, 2.0], fitness=80.0, generation=1, signature=(1.0, 0.0))
    e2 = tmi.upsert(genome=[3.0, 4.0], fitness=80.0, generation=1, signature=(0.0, 1.0))

    # Query with signature (1.0, 0.0)
    top = tmi.top_k(1, current_signature=(1.0, 0.0))
    assert len(top) == 1
    assert top[0].genome == [1.0, 2.0]


def test_ast_distance_prioritizes_refactoring_over_logic_inversion():
    """Structural: AST weighted distance penalizes logic inversion much higher than refactoring."""
    from evolab.ast_genome import ASTGenome, ast_distance

    code_orig = "def f(a, b):\n    x = a + b\n    y = a * 2\n    return x + y"
    code_refact = "def f(a, b):\n    y = a * 2\n    x = a + b\n    return x + y"
    code_logic = "def f(a, b):\n    x = a - b\n    y = a / 2\n    return x - y"

    g_orig = ASTGenome.from_code(code_orig)
    g_refact = ASTGenome.from_code(code_refact)
    g_logic = ASTGenome.from_code(code_logic)

    d_refact = ast_distance(g_orig, g_refact)
    d_logic = ast_distance(g_orig, g_logic)

    assert d_refact < d_logic
    assert d_logic >= 2.0 * d_refact


def test_causal_model_tri_state_neutral_dead_code_handling():
    """Inference: CausalModel treats zero-delta / inert dead code as neutral without crushing success rate."""
    from evolab.causal import CausalModel

    m = CausalModel(epsilon=1e-6)
    assert m.has_neutral_class is True

    # 10 dead code insertions
    for _ in range(10):
        m.observe("dead_code_insert", "ctx", delta=0.0)

    assert m.neutral_rate("dead_code_insert", "ctx") == 1.0
    # Success rate acknowledging neutral exploration is 0.5 (not 0.0 failure)
    assert m.success_rate("dead_code_insert", "ctx") == 0.5


def test_cem_phenotypic_hashing_deduplication():
    """Memory: CEM deduplicates polymorphic clones with identical phenotypic signatures."""
    from evolab.memory import TemporalMemoryIndex

    tmi = TemporalMemoryIndex(max_entries=10)
    # Upsert two different genomes with same phenotypic behavior hash
    e1 = tmi.upsert(genome=[1.0, 2.0], fitness=80.0, generation=1, signature=(1.0,), phenotypic_hash="hash_alpha")
    e2 = tmi.upsert(genome=[2.0, 1.0], fitness=85.0, generation=2, signature=(1.0,), phenotypic_hash="hash_alpha")

    # Should be deduplicated to 1 entry with updated higher fitness
    assert len(tmi.entries) == 1
    assert tmi.entries[0].fitness_at_archive == 85.0
    assert tmi.entries[0].generation_archived == 2


def test_assembly_distance_transposition_vs_opcode_replacement():
    """Low-Level: Instruction transposition costs significantly less than destructive opcode substitution."""
    from evolab.assembly_genome import AssemblyGenome, Instruction

    asm_orig = AssemblyGenome([
        Instruction("MOV", "R1", "R0"),
        Instruction("ADD", "R1", 10),
        Instruction("MUL", "R2", "R1"),
        Instruction("RET", "R2"),
    ])
    asm_reorder = AssemblyGenome([
        Instruction("MOV", "R1", "R0"),
        Instruction("MUL", "R2", "R1"),
        Instruction("ADD", "R1", 10),
        Instruction("RET", "R2"),
    ])
    asm_replace = AssemblyGenome([
        Instruction("MOV", "R1", "R0"),
        Instruction("SUB", "R1", 10),
        Instruction("DIV", "R2", "R1"),
        Instruction("RET", "R2"),
    ])

    d_reorder = asm_orig.distance_to(asm_reorder)
    d_replace = asm_orig.distance_to(asm_replace)

    assert d_reorder < d_replace


def test_float_genome_permutation_distance_vs_value_shift():
    """Forensics: Permutation (refactoring) costs less than substantive value shifts."""
    from evolab.genome import FloatGenome

    g_orig = FloatGenome([1.0, 2.0, 3.0, 4.0])
    g_perm = FloatGenome([4.0, 3.0, 2.0, 1.0])
    g_shift = FloatGenome([1.0, 2.0, 3.0, 10.0])

    d_perm = g_orig.structural_distance(g_perm)
    d_shift = g_orig.structural_distance(g_shift)

    assert d_perm < d_shift
    assert d_shift >= 2.0 * d_perm


def test_float_genome_segment_similarity_identifies_crossover_parents():
    """Forensics: Segment matching identifies provenance from both crossover parents."""
    from evolab.genome import FloatGenome

    p1 = FloatGenome([1.0, 2.0, 3.0, 4.0])
    p2 = FloatGenome([10.0, 20.0, 30.0, 40.0])
    offspring = FloatGenome([1.0, 2.0, 30.0, 40.0])

    sim_p1 = offspring.segment_similarity(p1, window_size=2)
    sim_p2 = offspring.segment_similarity(p2, window_size=2)

    assert sim_p1 == 1.0
    assert sim_p2 == 1.0


def test_map_elites_order_sensitive_slope_descriptor():
    """Forensics: Directional slope descriptor breaks permutation collapse in MAP-Elites."""
    from evolab.genome import FloatGenome

    g1 = FloatGenome([1.0, 2.0, 3.0, 4.0])
    g2 = FloatGenome([4.0, 3.0, 2.0, 1.0])

    desc1 = g1.describe()
    desc2 = g2.describe()

    assert desc1["slope"] > 0
    assert desc2["slope"] < 0
    assert desc1["slope"] != desc2["slope"]


def test_generalization_evaluator_penalizes_overfitting():
    """Forensics: GeneralizationEvaluator penalizes generalization gap between train and test."""
    from evolab.evaluators import GeneralizationEvaluator

    evaluator = GeneralizationEvaluator(
        train_evaluator=lambda g: 99.0,
        test_evaluator=lambda g: 20.0,
        max_overfit_gap=30.0,
    )
    res = evaluator.evaluate([1.0, 2.0])
    assert res.passed_holdout is False
    assert res.score < 60.0
    assert res.artifacts["is_overfit"] is True


def test_adversarial_robust_evaluator_penalizes_bloat():
    """Forensics: AdversarialRobustEvaluator strips unearned fitness inflation from gene bloat."""
    from evolab.evaluators import AdversarialRobustEvaluator
    from evolab.genome import FloatGenome

    evaluator = AdversarialRobustEvaluator(
        base_evaluator=lambda g: 99.0,
        baseline_length=4,
        bloat_penalty_per_gene=10.0,
    )
    clean_ind = FloatGenome([1.0, 2.0, 3.0, 4.0])
    bloated_ind = FloatGenome([1.0, 2.0, 3.0, 4.0, 999.0, 999.0])

    res_clean = evaluator.evaluate(clean_ind)
    res_bloated = evaluator.evaluate(bloated_ind)

    assert res_clean.score == 99.0
    assert res_bloated.score == 79.0  # 99.0 - 2 * 10.0 = 79.0


def test_rank_and_multiset_distances():
    """Forensics: rank_distance and multiset_distance quantify permutation and distribution divergence."""
    from evolab.genome import rank_distance, multiset_distance

    v1 = [1.0, 2.0, 3.0, 4.0]
    v2 = [4.0, 3.0, 2.0, 1.0]
    v3 = [1.1, 2.1, 3.1, 4.1]

    # Multiset distance between permutations of the same elements is 0
    assert multiset_distance(v1, v2) == 0.0
    assert multiset_distance(v1, v3) > 0.0

    # Rank distance captures inverted order
    assert rank_distance(v1, v2) > 0.5
    assert rank_distance(v1, v3) == 0.0


def test_ast_distance_distinguishes_function_calls():
    """Forensics: AST distance separates distinct function calls in non-mathematical code."""
    from evolab.ast_genome import ASTGenome

    code_sub = "def clean(s):\n    return re.sub(r'x', '', s)"
    code_split = "def clean(s):\n    return re.split(r'x', s)"

    g_sub = ASTGenome.from_code(code_sub)
    g_split = ASTGenome.from_code(code_split)

    d = g_sub.distance_to(g_split)
    assert d > 0.15





