"""
tests/test_holland_schema.py
Validation of John H. Holland's Schema Theorem and Optimal Trial Allocation (Holland 1975) — Milestone M3.
"""
import pytest
from evolab.holland import Schema, SchemaTracker, HollandTrialAllocator
from evolab.engine import EvolutionEngine
from evolab.genome import Individual, FloatGenome


def test_schema_properties():
    """Verifies calculation of Holland's order o(H) and defining length delta(H)."""
    # H1: (*, 1, *, *, 0, *)
    # Defined positions at index 1 and 4 -> order=2, delta = 4 - 1 = 3
    h1 = Schema(("*", 1, "*", "*", 0, "*"))
    assert h1.order == 2
    assert h1.defining_length == 3

    # H2: All wildcards (*, *, *, *)
    h2 = Schema(("*", "*", "*", "*"))
    assert h2.order == 0
    assert h2.defining_length == 0

    # H3: Single fixed gene (1, *, *, *)
    h3 = Schema((1, "*", "*", "*"))
    assert h3.order == 1
    assert h3.defining_length == 0

    # H4: Fully defined schema (1, 0, 1, 0)
    h4 = Schema((1, 0, 1, 0))
    assert h4.order == 4
    assert h4.defining_length == 3


def test_schema_matching_and_survival_bound():
    """Verifies schema matching and Holland's survival probability lower bound."""
    h = Schema(("*", 1, "*", "*", 0, "*"))
    
    # Matches: index 1 is 1, index 4 is 0
    c_match = [0, 1, 0, 1, 0, 1]
    assert h.matches(c_match) is True

    # Mismatch at index 1
    c_mismatch = [0, 0, 0, 1, 0, 1]
    assert h.matches(c_mismatch) is False

    # Mismatch length
    assert h.matches([0, 1]) is False

    # Survival bound: l=6, delta=3, order=2, p_c=0.7, p_m=0.01
    # Disruption crossover = 0.7 * (3 / 5) = 0.42
    # Disruption mutation  = 2 * 0.01 = 0.02
    # S(H) >= 1 - 0.42 - 0.02 = 0.56
    bound = h.survival_bound(p_c=0.7, p_m=0.01, chromosome_length=6)
    assert pytest.approx(bound, 0.001) == 0.56


def test_holland_trial_allocator_bandit_dynamics():
    """Verifies that HollandTrialAllocator balances exploration and exploits higher-payoff arms."""
    allocator = HollandTrialAllocator(arms=["arm_a", "arm_b"], c_exploration=1.0)
    
    # Round 1 & 2: each arm must be sampled at least once initially
    arm1 = allocator.select_arm()
    allocator.update(arm1, 2.0)
    
    arm2 = allocator.select_arm()
    assert arm2 != arm1
    allocator.update(arm2, 20.0)

    # Over 50 subsequent rounds, the superior arm (arm2 with reward 20.0)
    # must be allocated strictly more trials than arm1
    for _ in range(50):
        chosen = allocator.select_arm()
        reward = 20.0 if chosen == arm2 else 2.0
        allocator.update(chosen, reward)

    desc = allocator.describe()
    assert desc["total_trials"] == 52
    assert desc["counts"][arm2] > desc["counts"][arm1]
    assert desc["allocation_ratios"][arm2] > 0.60
    assert desc["mean_rewards"][arm2] == pytest.approx(20.0, 0.01)


def test_schema_tracker_empirical_bound():
    """Verifies SchemaTracker computes population metrics and Holland's expected lower bound."""
    schema = Schema((1, "*", 0))
    tracker = SchemaTracker([schema])

    # Construct synthetic population of 4 individuals
    # ind1: [1, 1, 0] -> matches schema, fitness 80.0
    # ind2: [1, 0, 0] -> matches schema, fitness 100.0
    # ind3: [0, 1, 0] -> no match, fitness 40.0
    # ind4: [0, 0, 1] -> no match, fitness 20.0
    pop = [
        Individual(genome=FloatGenome([1, 1, 0]), species="default", fitness=80.0),
        Individual(genome=FloatGenome([1, 0, 0]), species="default", fitness=100.0),
        Individual(genome=FloatGenome([0, 1, 0]), species="default", fitness=40.0),
        Individual(genome=FloatGenome([0, 0, 1]), species="default", fitness=20.0),
    ]

    count = tracker.count_instances(schema, pop)
    assert count == 2

    # Schema mean fitness: (80 + 100) / 2 = 90.0
    f_H = tracker.schema_fitness(schema, pop)
    assert f_H == 90.0

    # Population mean fitness: (80 + 100 + 40 + 20) / 4 = 60.0
    f_bar = tracker.population_mean_fitness(pop)
    assert f_bar == 60.0

    # Ratio f_H / f_bar = 90.0 / 60.0 = 1.5
    # delta(H) = 2 - 0 = 2, order = 2, l = 3
    # survival = 1.0 - 0.5 * (2/2) - 2 * 0.05 = 1.0 - 0.5 - 0.1 = 0.4
    # Expected lower bound = m_t * 1.5 * 0.4 = 2 * 1.5 * 0.4 = 1.2
    bound = tracker.compute_holland_lower_bound(schema, pop, p_c=0.5, p_m=0.05, chromosome_length=3)
    assert pytest.approx(bound, 0.01) == 1.20


def test_engine_holland_allocator_integration():
    """Verifies that EvolutionEngine runs cleanly with holland_allocator=True and produces telemetry."""
    def simple_sphere(ind):
        # Shifted sphere: optimum near all 0.5
        vals = ind.genome if isinstance(ind.genome, list) else getattr(ind.genome, "values", ind.genome)
        err = sum((x - 0.5) ** 2 for x in vals)
        return max(0.0, 100.0 - err * 20.0)

    engine = EvolutionEngine(
        fitness_fn=simple_sphere,
        population_size=12,
        genome_size=4,
        generations=6,
        holland_allocator=True,
        holland_c=1.2,
        seed=42,
    )

    assert engine.holland_allocator_enabled is True
    assert engine._holland_allocator is not None

    report = engine.run(generations=6)

    # Holland metadata attached to report
    assert "holland_allocator" in report
    holland_data = report["holland_allocator"]
    assert holland_data["total_trials"] > 0
    assert "light" in holland_data["counts"]
    assert "semantic" in holland_data["counts"]

    # History contains generational allocation telemetry
    history = report["history"]
    assert len(history) > 0
    assert "holland_allocation" in history[0]
    assert "light" in history[0]["holland_allocation"]


def test_engine_default_unperturbed():
    """Verifies that the default EvolutionEngine maintains exact backward compatibility without Holland allocator."""
    engine = EvolutionEngine(population_size=10, genome_size=3)
    assert engine.holland_allocator_enabled is False
    assert engine._holland_allocator is None
