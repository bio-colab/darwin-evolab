"""tests/test_self_evolution.py — Tests for Closed Self-Improvement Loop & Meta-Evolution."""

from __future__ import annotations

import pytest

from evolab.self_evolution import MetaProposal, SelfEvolutionEngine


def test_generate_proposals_on_premature_convergence():
    """Verify that diagnosed premature convergence triggers diversity injection proposal."""
    engine = SelfEvolutionEngine(base_config={"immigrant_fraction": 0.0})
    diagnosis = {
        "premature_convergence": True,
        "stagnation": True,
        "overfit_risk": False,
    }

    proposals = engine.generate_proposals(diagnosis)
    assert len(proposals) >= 1
    categories = [p.category for p in proposals]
    assert "diversity_injection" in categories

    div_prop = next(p for p in proposals if p.category == "diversity_injection")
    assert div_prop.parameters["immigrant_fraction"] > 0.0


def test_self_evolution_accepts_and_atomically_commits_improvement():
    """Verify that statistically significant improvements are accepted and update configuration."""
    engine = SelfEvolutionEngine(base_config={"immigrant_fraction": 0.0, "mutation_rate": 0.15})
    proposal = MetaProposal(
        proposal_id=1,
        category="diversity_injection",
        parameters={"immigrant_fraction": 0.15},
        rationale="Inject immigrants to break premature convergence",
    )

    # Significant improvement across 7 seeds
    b_scores = [70.0, 71.0, 69.5, 70.5, 71.2, 70.8, 69.9]
    c_scores = [78.0, 79.0, 77.5, 78.5, 79.2, 78.8, 77.9]

    outcome = engine.step(proposal, b_scores, c_scores, regressions=0)
    assert outcome["action"] == "CONFIG_UPDATED"
    assert engine.config["immigrant_fraction"] == 0.15
    assert proposal.status == "ACCEPT"


def test_self_evolution_rejects_noisy_proposal_and_protects_config():
    """Verify that noisy/insignificant candidate modifications are rejected without config corruption."""
    original_config = {"immigrant_fraction": 0.0, "mutation_rate": 0.15}
    engine = SelfEvolutionEngine(base_config=original_config)
    proposal = MetaProposal(
        proposal_id=2,
        category="random_tweak",
        parameters={"mutation_rate": 0.35},
        rationale="Aggressive mutation",
    )

    # Noisy, identical distributions
    b_scores = [80.0, 81.0, 80.0, 79.0, 81.0, 80.0, 80.5]
    c_scores = [80.05, 81.0, 80.0, 79.0, 81.05, 80.0, 80.5]

    outcome = engine.step(proposal, b_scores, c_scores, regressions=0)
    assert outcome["action"] == "PROPOSAL_REJECTED"
    # Config must remain pristine
    assert engine.config["mutation_rate"] == 0.15
    assert proposal.status == "REJECT"
    assert "mutation_rate:0.35" in engine.dead_ends
