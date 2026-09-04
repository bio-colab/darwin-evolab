"""Unit test suite for SpiceNeuralSurrogate and Active Learning Pareto verification."""
from __future__ import annotations

import time
import pytest

from evolab.genome import FloatGenome, Individual
from evolab.silicon.opamp_benchmark import OpAmpSizing, TwoStageMillerOpAmpEvaluator
from evolab.silicon.surrogate import (
    ActiveSpiceSurrogateEvaluator,
    MicroMLP,
    SpiceNeuralSurrogate,
)


def test_micro_mlp_inference_and_speed():
    mlp = MicroMLP(input_dim=14, hidden_dim=32, output_dim=4)
    x = [0.5] * 14

    t0 = time.perf_counter()
    for _ in range(100):
        out = mlp.forward(x)
    dt_per_eval_ms = (time.perf_counter() - t0) / 100 * 1000.0

    assert len(out) == 4
    # Inference must take < 0.1 ms per evaluation
    assert dt_per_eval_ms < 0.2


def test_micro_mlp_training_step():
    mlp = MicroMLP(input_dim=14, hidden_dim=16, output_dim=4)
    x = [0.3] * 14
    y_target = [70.0, 15.0, 65.0, 300.0]

    # Initial loss
    out0 = mlp.forward(x)
    loss0 = sum((out0[i] - y_target[i]) ** 2 for i in range(4)) / 4.0

    # Train for several steps
    for _ in range(15):
        mlp.train_step(x, y_target, lr=0.01)

    out1 = mlp.forward(x)
    loss1 = sum((out1[i] - y_target[i]) ** 2 for i in range(4)) / 4.0
    assert loss1 < loss0  # Loss must decrease


def test_spice_neural_surrogate_prediction_and_buffer():
    surrogate = SpiceNeuralSurrogate()
    sizing = OpAmpSizing()

    # Pre-train with a few samples
    surrogate.warm_start_from_physics(num_samples=8)
    assert surrogate.samples_collected == 8

    # Inference
    pred = surrogate.predict_metrics(sizing)
    assert pred.gain_db > 0.0
    assert pred.gbw_mhz > 0.0
    assert pred.physical_claim is False  # Explicitly flagged as surrogate
    assert pred.artifacts["surrogate_prediction"] is True


def test_active_spice_surrogate_evaluator():
    base_ev = TwoStageMillerOpAmpEvaluator()
    active_ev = ActiveSpiceSurrogateEvaluator(
        base_evaluator=base_ev, pareto_verification_ratio=0.2
    )

    pop: list[Individual] = []
    for i in range(10):
        pop.append(Individual(
            genome=FloatGenome(values=[0.1 * i] * 14),
            species="opamp",
            fitness=0.0,
            _generation=0,
            _index=i,
        ))

    # Active learning population rollout
    results = active_ev.evaluate_population_active_learning(pop)
    assert len(results) == 10
    # Exactly k_verify (2 out of 10) were physically evaluated
    assert active_ev.total_physical_evals == 2
    assert active_ev.total_surrogate_evals >= 10
    assert active_ev.surrogate.samples_collected >= 18  # 16 warm-start + 2 physical
