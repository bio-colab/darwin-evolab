"""tests/test_neural_profiler_and_operators.py — Tests for ActivationProfiler and Advanced Weight Operators."""
from __future__ import annotations

import numpy as np
import pytest

from evolab.neural.benchmark import ControlledToyBenchmark
from evolab.neural.genome import NeuralWeightGenome
from evolab.neural.operators import NullSpaceProjectionMutation, RankRPerturbationMutation
from evolab.neural.profiler import ActivationProfiler

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def test_activation_profiler_numpy():
    """Verify ActivationProfiler accurately traces and profiles NumPy model representations."""
    bench = ControlledToyBenchmark(seed=42)
    spec = bench.get_spec()
    profiler = ActivationProfiler(discrepancy_threshold=1.0, top_k_units=4)

    profile = profiler.profile(
        model=spec.base_weights,
        failing_inputs=spec.fail_to_pass_x,
        anchor_inputs=spec.pass_to_pass_x,
    )

    assert "layer1" in profile.layers
    assert "layer2" in profile.layers
    # In ControlledToyBenchmark, hidden units 12, 13, 14, 15 are the active representation for Class 3
    l1_stats = profile.layers["layer1"]
    assert l1_stats.discrepancy is not None
    assert l1_stats.contrast_ratio is not None
    assert set([12, 13, 14, 15]).issubset(set(l1_stats.active_indices))
    assert len(profile.anchor_activations) > 0


def test_activation_profiler_torch():
    """Verify ActivationProfiler forward hooks on PyTorch nn.Module."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed; skipping PyTorch profiler test.")

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )

    x_fail = np.random.randn(10, 8).astype(np.float32)
    x_anchor = np.random.randn(20, 8).astype(np.float32)

    profiler = ActivationProfiler()
    profile = profiler.profile(model=model, failing_inputs=x_fail, anchor_inputs=x_anchor)

    assert len(profile.layers) > 0
    assert profile.summary != ""


def test_compute_null_space_basis():
    """Verify mathematical exactness of orthogonal null space projection."""
    profiler = ActivationProfiler()
    rng = np.random.RandomState(42)

    # Create low-rank anchor activations: 20 samples lying in a 4-dimensional subspace of R^16
    V_subspace = rng.randn(4, 16)
    coeffs = rng.randn(20, 4)
    X_anchor = coeffs @ V_subspace  # shape (20, 16), rank 4

    P_null, s = profiler.compute_null_space_basis(X_anchor, tol=1e-4)

    assert P_null.shape == (16, 16)
    # P_null must be symmetric and idempotent: P^2 = P
    np.testing.assert_allclose(P_null @ P_null, P_null, atol=1e-5)
    np.testing.assert_allclose(P_null.T, P_null, atol=1e-5)

    # Product X_anchor @ P_null must be identically zero
    residual = X_anchor @ P_null
    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-7)


def test_null_space_projection_mutation():
    """Verify NullSpaceProjectionMutation guarantees zero perturbation on anchor representations."""
    rng = np.random.RandomState(100)

    # 30 anchor samples in R^8 entering a layer with 4 outputs
    d_out, d_in = 4, 8
    # Make rank 3 in R^8 so non-trivial null space exists
    V_sub = rng.randn(3, d_in)
    X_anchor = rng.randn(30, 3) @ V_sub  # shape (30, 8)

    op = NullSpaceProjectionMutation(
        target_layer="layer2.weight",
        anchor_activations=X_anchor,
        eta=0.1,
    )

    g = NeuralWeightGenome(base_model_id="test_model")
    layer_shapes = {"layer2.weight": (d_out, d_in)}

    child_g = op.mutate(g, layer_shapes=layer_shapes)
    delta_w = child_g.sparse_patches["layer2.weight"]

    assert delta_w.shape == (d_out, d_in)
    # Output perturbation on anchors: Delta Y = X_anchor @ Delta_W^T
    delta_y = X_anchor @ delta_w.T
    # Must be zero within numerical tolerance
    np.testing.assert_allclose(delta_y, np.zeros_like(delta_y), atol=1e-7)
    assert child_g.edit_lineage[-1]["op"] == "null_space_projection"


def test_rank_r_perturbation_mutation():
    """Verify RankRPerturbationMutation generates bounded rank-r updates."""
    op = RankRPerturbationMutation(rank=3, eta=0.08, decay_rate=0.5)
    g = NeuralWeightGenome(base_model_id="test_model")
    layer_shapes = {"layer1.weight": (16, 8)}

    child_g = op.mutate(g, layer_shapes=layer_shapes)
    delta_w = child_g.sparse_patches["layer1.weight"]

    assert delta_w.shape == (16, 8)
    # Check effective numerical rank of delta_w <= rank (3)
    _, s, _ = np.linalg.svd(delta_w)
    effective_rank = int(np.sum(s > 1e-5))
    assert effective_rank <= 3
    assert child_g.edit_lineage[-1]["op"] == "rank_r_perturbation"
    assert child_g.edit_lineage[-1]["rank"] == 3
