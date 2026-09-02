"""555-class astable: ngspice .tran → measure_transient → fitness → short GA."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from evolab.engine import EvolutionEngine
from evolab.genome import FloatGenome, Individual

from experimental.electronics.evaluators.tran_evaluator import Timer555TrueTransientEvaluator as Timer555TranEvaluator

_EMBEDDED_NGSPICE = Path(__file__).resolve().parents[1] / "tools" / "ngspice"
NGSPICE = (
    os.environ.get("EVOLAB_NGSPICE")
    or shutil.which("ngspice")
    or (str(_EMBEDDED_NGSPICE) if _EMBEDDED_NGSPICE.exists() else "/usr/bin/ngspice")
)


def main() -> dict:
    if not Path(NGSPICE).exists():
        return {"ok": False, "error": "ngspice_binary_missing", "path": NGSPICE}
    
    # Target specs: 100Hz, 3Vpp (relaxed for BJT astable)
    ev = Timer555TranEvaluator(NGSPICE, target_freq_hz=100.0, target_vpp=3.0)
    
    # Diverse initial population covering wide parameter range
    rng_pop = [
        Individual(FloatGenome([10000.0, 10000.0, 1e-7]), species="spec_electronics"),
        Individual(FloatGenome([8000.0, 12000.0, 8e-8]), species="spec_electronics"),
        Individual(FloatGenome([15000.0, 7000.0, 1.2e-7]), species="spec_electronics"),
        Individual(FloatGenome([5000.0, 5000.0, 5e-8]), species="spec_electronics"),
    ]
    
    engine = EvolutionEngine(
        fitness_fn=ev,
        population_size=4,
        seed=1,
        genome_size=3,
        early_stop_fitness=99.0,
        stagnation_patience=6,
        sharing_mode="off",
    )
    
    report = engine.run(15, initial_population=rng_pop)
    best = report.get("best_individual") or {}
    
    return {
        "ok": True,
        "pipeline": "astable → ngspice-44.2 .tran → TransientArtifact → measure_transient → fitness → GA",
        "evals": report.get("candidates") or report.get("n_evaluations"),
        "best_fitness": best.get("fitness"),
        "holdout": report.get("holdout"),
        "best_artifacts": (best.get("artifacts") if isinstance(best, dict) else None),
        "generations_run": 15,
        "pop": 4,
        "ngspice_version": "ngspice-44.2",
        "ngspice_path": NGSPICE,
        "target_specs": {"freq_hz": 100.0, "vpp": 3.0},
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
