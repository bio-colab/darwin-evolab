"""tests/test_neural_genome.py — Tests for Neural Weight-Space Programming (WSP) in Darwin-Evolab."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from evolab.adapters import get_domain_adapter
from evolab.genome import Individual
from evolab.neural.adapter import NeuralWeightAdapter
from evolab.neural.analyst import MockWeightAnalyst, WeightHypothesis
from evolab.neural.benchmark import ControlledToyBenchmark, NeuralProblemSpec, NeuralWeightEvaluator
from evolab.neural.genome import NeuralWeightGenome
from evolab.neural.operators import (
    CounterfactualIntronPruner,
    LLMHypothesisMutation,
    RankOnePerturbationMutation,
    SpectralRescaleMutation,
)


def test_neural_weight_genome_clone_and_fingerprint():
    """Verify deep cloning, fingerprint determinism, and immutability."""
    p1 = np.array([[1.0, -0.5], [0.0, 2.0]], dtype=float)
    g1 = NeuralWeightGenome(
        base_model_id="mlp_test",
        sparse_patches={"layer1": p1},
        scale=1.0,
    )
    fp1 = g1.fingerprint()
    assert isinstance(fp1, str)
    assert len(fp1) == 64

    # Clone must have identical fingerprint
    g_clone = g1.clone()
    assert g_clone.fingerprint() == fp1

    # In-place modification of clone must not mutate original
    g_clone.sparse_patches["layer1"][0, 0] = 99.0
    assert g_clone.fingerprint() != fp1
    assert g1.sparse_patches["layer1"][0, 0] == 1.0


def test_neural_weight_genome_dense_delta():
    """Verify calculation of effective delta from sparse patches and low-rank factors."""
    p = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    u = np.array([[1.0], [0.0]], dtype=float)
    v = np.array([[0.0], [2.0]], dtype=float)  # outer product u*v^T = [[0, 2], [0, 0]]

    g = NeuralWeightGenome(
        base_model_id="mlp_test",
        sparse_patches={"w": p},
        low_rank_factors={"w": (u, v)},
        scale=2.0,
    )
    delta = g.get_dense_delta("w")
    assert delta is not None
    # (p + u*v^T) * 2 = ([[1, 2], [3, 4]] + [[0, 2], [0, 0]]) * 2 = [[2, 8], [6, 8]]
    expected = np.array([[2.0, 8.0], [6.0, 8.0]], dtype=float)
    np.testing.assert_allclose(delta, expected)


def test_neural_weight_genome_distance_metric():
    """Verify Frobenius functional distance metric between weight genomes."""
    g1 = NeuralWeightGenome(
        base_model_id="mlp",
        sparse_patches={"w": np.array([[1.0, 0.0], [0.0, 1.0]])},
    )
    g2 = NeuralWeightGenome(
        base_model_id="mlp",
        sparse_patches={"w": np.array([[1.0, 0.0], [0.0, 4.0]])},
    )
    # Distance to self must be 0
    assert g1.distance_to(g1) == 0.0

    # Distance between g1 and g2: norm of [[0, 0], [0, -3]] is 3.0
    dist = g1.distance_to(g2)
    assert pytest.approx(dist, rel=1e-5) == 3.0


def test_neural_weight_genome_apply_to_numpy_and_pytorch():
    """Verify weight application to both pure NumPy model and PyTorch Module."""
    # 1. NumPy dict model
    base_dict = {"layer1.weight": np.zeros((2, 2), dtype=float)}
    g = NeuralWeightGenome(
        base_model_id="test",
        sparse_patches={"layer1.weight": np.array([[1.5, 2.5], [3.5, 4.5]])},
        scale=1.0,
    )
    patched = g.apply_to_model(base_dict)
    np.testing.assert_allclose(patched["layer1.weight"], [[1.5, 2.5], [3.5, 4.5]])

    # 2. PyTorch nn.Module
    if not HAS_TORCH:
        return

    linear = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.zero_()

    g_torch = NeuralWeightGenome(
        base_model_id="test",
        sparse_patches={"weight": np.array([[1.0, 2.0], [3.0, 4.0]])},
        scale=1.0,
    )
    g_torch.apply_to_model(linear)
    np.testing.assert_allclose(linear.weight.detach().cpu().numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_prune_inactive_weights():
    """Verify parsimony enforcement through threshold ablation."""
    p = np.array([[1.0, 1e-6], [-1e-5, 2.0]], dtype=float)
    g = NeuralWeightGenome(base_model_id="test", sparse_patches={"w": p})
    pruned = g.prune_inactive_weights(threshold=1e-4)
    assert pruned == 2
    assert g.sparse_patches["w"][0, 1] == 0.0
    assert g.sparse_patches["w"][1, 0] == 0.0
    assert g.sparse_patches["w"][0, 0] == 1.0


def test_algebraic_operators():
    """Verify spectral SVD and Rank-1 mutation operators."""
    g = NeuralWeightGenome(base_model_id="test")
    W = np.random.RandomState(42).randn(8, 4)

    # 1. SpectralRescaleMutation
    svd_op = SpectralRescaleMutation(singular_index=0, scale_factor=1.5)
    g_svd = svd_op.mutate(g, layer_weights={"layer1": W})
    assert "layer1" in g_svd.sparse_patches
    assert len(g_svd.edit_lineage) == 1
    assert g_svd.edit_lineage[0]["op"] == "svd_spectral_rescale"

    # 2. RankOnePerturbationMutation
    rank1_op = RankOnePerturbationMutation(eta=0.2)
    g_r1 = rank1_op.mutate(g, layer_shapes={"layer1": (8, 4)})
    assert "layer1" in g_r1.sparse_patches
    # Check that rank of the delta matrix is 1
    u, s, vt = np.linalg.svd(g_r1.sparse_patches["layer1"])
    assert np.count_nonzero(np.round(s, 6)) == 1

    # 3. LLMHypothesisMutation
    hyp = {"layer": "layer1", "delta": np.ones((8, 4)) * 0.5, "id": "test_hyp"}
    llm_op = LLMHypothesisMutation(hyp)
    g_llm = llm_op.mutate(g)
    np.testing.assert_allclose(g_llm.sparse_patches["layer1"], 0.5)


def test_controlled_toy_benchmark_defect_and_invariants():
    """Verify the controlled synthetic benchmark and Dual-Invariant evaluation."""
    bench = ControlledToyBenchmark(seed=42)
    spec = bench.get_spec()
    evaluator = NeuralWeightEvaluator(spec)

    # 1. Defective baseline genome (no patch)
    g_base = NeuralWeightGenome(base_model_id=spec.base_model_id)
    res_base = evaluator.evaluate(g_base)
    # Defective model has 0% recall on Class 3, 100% on Classes 0-2
    assert res_base.sub_scores["target_defect_recall"] == 0.0
    assert res_base.sub_scores["anchor_suite_accuracy"] == 100.0
    assert res_base.score == 0.0
    assert res_base.passed_holdout is False

    # 2. Perfect repair patch
    diff_w2 = bench.clean_weights["layer2.weight"] - spec.base_weights["layer2.weight"]
    diff_b2 = bench.clean_weights["layer2.bias"] - spec.base_weights["layer2.bias"]
    g_perfect = NeuralWeightGenome(
        base_model_id=spec.base_model_id,
        sparse_patches={"layer2.weight": diff_w2, "layer2.bias": diff_b2},
    )
    res_perfect = evaluator.evaluate(g_perfect)
    assert res_perfect.sub_scores["target_defect_recall"] == 100.0
    assert res_perfect.sub_scores["anchor_suite_accuracy"] == 100.0
    assert res_perfect.score == 100.0
    assert res_perfect.passed_holdout is True

    # 3. Regressive patch (fixes Class 3 but breaks Class 0)
    diff_regressive = np.copy(diff_w2)
    diff_regressive[0, :] = -10.0  # Destroys Class 0
    g_regressive = NeuralWeightGenome(
        base_model_id=spec.base_model_id,
        sparse_patches={"layer2.weight": diff_regressive, "layer2.bias": diff_b2},
    )
    res_regressive = evaluator.evaluate(g_regressive)
    assert res_regressive.sub_scores["anchor_suite_accuracy"] < 100.0
    assert res_regressive.passed_holdout is False
    assert res_regressive.score < 50.0  # Penalized heavily by dual-invariant rule


def test_neural_weight_adapter_lifecycle():
    """Verify NeuralWeightAdapter population building, evaluation, and solution export."""
    adapter = get_domain_adapter("neural_weight_space")
    assert isinstance(adapter, NeuralWeightAdapter)
    assert adapter.name == "neural_weight_space"

    spec = adapter.parse_spec({"scenario": "toy_4class_defect_repair", "seed": 42})
    assert spec.target_class == 3

    pop = adapter.build_population(spec, size=8, rng=np.random.RandomState(42))
    assert len(pop) == 8
    assert all(isinstance(ind.genome, NeuralWeightGenome) for ind in pop)

    evaluator = adapter.build_evaluator(spec)
    for ind in pop:
        res = evaluator.evaluate(ind)
        ind.fitness = res.score
        assert isinstance(res.score, float)

    # Best individual export
    best_ind = max(pop, key=lambda i: i.fitness)
    export_data = adapter.export_solution(best_ind, spec)
    assert "status" in export_data
    assert "fitness_score" in export_data
    assert "genome_serialization" in export_data


def test_neural_weight_genome_patch_scope_and_zero_drift():
    """Verify patch_scope ensures zero parameter drift on exit even if exceptions occur."""
    base_dict = {"layer1.weight": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)}
    original_copy = np.copy(base_dict["layer1.weight"])

    g = NeuralWeightGenome(
        base_model_id="test",
        sparse_patches={"layer1.weight": np.array([[0.5, -0.5], [1.0, -1.0]])},
        scale=1.0,
    )

    # 1. Normal exit
    with g.patch_scope(base_dict) as patched:
        np.testing.assert_allclose(patched["layer1.weight"], [[1.5, 1.5], [4.0, 3.0]])

    # Verify complete reversion to original state
    np.testing.assert_allclose(base_dict["layer1.weight"], original_copy)

    # 2. Exit upon exception
    with pytest.raises(RuntimeError):
        with g.patch_scope(base_dict):
            raise RuntimeError("Simulated evaluation failure")

    np.testing.assert_allclose(base_dict["layer1.weight"], original_copy)

    # 3. PyTorch nn.Module if available
    if HAS_TORCH:
        linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(original_copy))
        g_torch = NeuralWeightGenome(
            base_model_id="test",
            sparse_patches={"weight": np.array([[0.5, -0.5], [1.0, -1.0]])},
            scale=1.0,
        )
        with g_torch.patch_scope(linear):
            np.testing.assert_allclose(linear.weight.detach().cpu().numpy(), [[1.5, 1.5], [4.0, 3.0]])
        np.testing.assert_allclose(linear.weight.detach().cpu().numpy(), original_copy)


def test_neural_weight_genome_serialization_and_deserialization():
    """Verify byte-accurate genome reconstruction from serialized COO representations."""
    p = np.array([[0.0, 2.5], [-1.5, 0.0]], dtype=float)
    u = np.array([[1.0], [2.0]], dtype=float)
    v = np.array([[0.5], [-0.5]], dtype=float)

    g = NeuralWeightGenome(
        base_model_id="mlp_serdes",
        sparse_patches={"w1": p},
        low_rank_factors={"w2": (u, v)},
        scale=1.25,
        edit_lineage=[{"op": "init", "val": 42}],
    )

    ser = g.serialize(include_coo=True)
    assert "sparse_coo" in ser
    assert "low_rank_factors" in ser
    assert ser["scale"] == 1.25

    reconstructed = NeuralWeightGenome.deserialize(ser)
    assert reconstructed.base_model_id == g.base_model_id
    assert reconstructed.scale == g.scale
    assert reconstructed.fingerprint() == g.fingerprint()
    np.testing.assert_allclose(reconstructed.get_dense_delta("w1"), g.get_dense_delta("w1"))
    np.testing.assert_allclose(reconstructed.get_dense_delta("w2"), g.get_dense_delta("w2"))
    assert reconstructed.edit_lineage == g.edit_lineage


def test_neural_weight_genome_functional_distance():
    """Verify functional divergence distance on probe sets with forward function."""
    g1 = NeuralWeightGenome(
        base_model_id="mlp_dist",
        sparse_patches={"w": np.array([[1.0, 0.0], [0.0, 1.0]])},
    )
    g2 = NeuralWeightGenome(
        base_model_id="mlp_dist",
        sparse_patches={"w": np.array([[2.0, 0.0], [0.0, 2.0]])},
    )

    model = {"w": np.zeros((2, 2), dtype=float)}
    probe_inputs = np.array([[1.0, 1.0], [2.0, 3.0]], dtype=float)

    def forward_fn(m, x):
        return np.dot(x, m["w"])

    dist_func = g1.distance_to(g2, probe_inputs=probe_inputs, forward_fn=forward_fn, model=model)
    dist_param = g1.distance_to(g2)
    assert dist_func > dist_param


def test_controlled_benchmark_deep_architecture():
    """Verify deep 6-layer GELU architecture (8 -> 32 -> 16 -> 48 -> 32 -> 16 -> 4) defect repair."""
    bench = ControlledToyBenchmark(seed=42, architecture="deep")
    spec = bench.get_spec()
    evaluator = NeuralWeightEvaluator(spec, min_anchor_acc=99.7, min_fail_acc=96.0)

    # 1. Defective baseline
    g_base = NeuralWeightGenome(base_model_id=spec.base_model_id)
    res_base = evaluator.evaluate(g_base)
    assert res_base.sub_scores["target_defect_recall"] == 0.0
    assert res_base.sub_scores["anchor_suite_accuracy"] >= 99.7
    assert res_base.passed_holdout is False

    # 2. Perfect repair
    diff_w6 = bench.clean_weights["layer6.weight"] - spec.base_weights["layer6.weight"]
    g_perfect = NeuralWeightGenome(
        base_model_id=spec.base_model_id,
        sparse_patches={"layer6.weight": diff_w6},
    )
    res_perfect = evaluator.evaluate(g_perfect)
    assert res_perfect.sub_scores["target_defect_recall"] == 100.0
    assert res_perfect.sub_scores["anchor_suite_accuracy"] == 100.0
    assert res_perfect.passed_holdout is True


def test_multiseed_tri_factor_benchmark_and_wilson_ci():
    """Verify multi-seed benchmark execution and Wilson confidence intervals."""
    from evolab.neural.experiment import run_multiseed_tri_factor_benchmark, wilson_score_interval

    # Wilson interval check
    ci_low, ci_high = wilson_score_interval(10, 10)
    assert 70.0 < ci_low <= 75.0
    assert ci_high == 100.0

    # Fast 2-seed execution
    summary = run_multiseed_tri_factor_benchmark(seeds=[1, 2], architecture="shallow")
    assert summary["group_a_blind_ga"]["seeds_tested"] == 2
    assert summary["group_c_darwin_wsp"]["pass_rate_pct"] == 100.0


