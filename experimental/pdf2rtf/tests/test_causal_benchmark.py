"""test_causal_benchmark.py — Unit tests for MAP-Elites vs. Random Search Causal Benchmark."""

from pathlib import Path
import pytest

from experimental.pdf2rtf.benchmark_map_elites_vs_random import (
    compute_cell_coordinate,
    compute_paired_statistics,
    desc_cluster_density,
    desc_table_sensitivity,
    evaluate_archive_state,
    run_comparative_benchmark,
    sample_uniform_genome,
)
from experimental.pdf2rtf.genome import ProfileGenome
import random


def test_compute_cell_coordinate():
    # Bounds clamping tests
    assert compute_cell_coordinate(0.0, 0.0) == (0, 0)
    assert compute_cell_coordinate(1.0, 1.0) == (9, 9)
    assert compute_cell_coordinate(0.55, 0.25) == (5, 2)
    assert compute_cell_coordinate(-0.5, 1.5) == (0, 9)


def test_descriptors_and_sampling():
    rng = random.Random(42)
    g = sample_uniform_genome(rng)
    assert isinstance(g, ProfileGenome)
    d1 = desc_cluster_density(g)
    d2 = desc_table_sensitivity(g)
    assert 0.0 <= d1 <= 1.0
    assert 0.0 <= d2 <= 1.0


def test_compute_paired_statistics():
    me = [90.0, 92.0, 95.0, 91.0]
    rs = [80.0, 82.0, 83.0, 81.0]
    stats = compute_paired_statistics(me, rs)
    assert stats["mean_diff"] > 0
    assert stats["t_stat"] > 0
    assert stats["p_val"] < 0.05
    assert stats["cohens_d"] > 1.0


def test_evaluate_archive_state():
    archive = {(0, 0): 0.95, (1, 1): 0.99, (2, 2): 0.92}
    summary = evaluate_archive_state(archive, total_niches=100)
    assert summary.occupied_niches == 3
    assert summary.total_niches == 100
    assert summary.coverage_pct == 3.0
    assert summary.max_fitness == 0.99
    assert abs(summary.qd_score - 2.86) < 1e-4


def test_causal_benchmark_mini_run(tmp_path: Path):
    """Executes mini benchmark with tmp_path to verify end-to-end integration without touching production reports."""
    report_file = tmp_path / "mini_causal_report.json"
    report = run_comparative_benchmark(
        budget=100,
        seeds=[42],
        pop_size=50,
        parallel=False,
        save_report_path=report_file,
    )
    assert report_file.exists()
    assert "comparative_summary" in report
    assert "illumination_trajectory" in report
    assert "seed_level_runs" in report
    assert report["preregistered_protocol"]["budget_evaluations"] == 100
