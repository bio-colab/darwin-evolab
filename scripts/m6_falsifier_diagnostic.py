#!/usr/bin/env python3
"""M6 falsifier diagnostic (descriptive only — no decision role): find the
exact (candidates, weights) pairs where the gate stays silent on the lru
teacher store, i.e. where v2's <=1.10:1 envelope claim does NOT hold for a
NARROWED candidate subset.
"""
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab import EngineConfig, EvolutionEngine
from evolab.code_fixtures import SCENARIO_REGISTRY, make_code_population
from evolab.experience import (
    ExperienceMutationPrior,
    ExperienceStore,
    attach_experience_recorder,
    problem_fingerprint,
)

SCEN = sys.argv[1] if len(sys.argv) > 1 else "lru_cache_logic"
scenario = SCENARIO_REGISTRY[SCEN]()
fp = problem_fingerprint(scenario.sources, scenario.target_file, scenario.func_name)

td = tempfile.mkdtemp(prefix="m6_diag_")
teacher_db = Path(td) / "teacher.db"
os.environ.pop("EVOLAB_EXPERIENCE", None)
os.environ.pop("EVOLAB_EXPERIENCE_DB", None)
for i in range(3):
    ev = scenario.create_evaluator()
    wired = attach_experience_recorder(
        ev, scenario.sources, scenario.target_file, scenario.func_name,
        db_path=teacher_db, run_id=f"teacher_t{i}", prior_enabled=False,
    )
    engine = EvolutionEngine(
        fitness_fn=wired,
        config=EngineConfig(generations=8, population_size=12, elite_count=1,
                            genome_size=2, seed=9901 + i),
    )
    initial = make_code_population(scenario, 12, random.Random(9901 + i))
    engine.run(generations=8, initial_population=initial)

store = ExperienceStore(teacher_db)
stats = store.stats(fp)
print(f"total_experiences={stats['total_experiences']}")
per = stats.get("per_edit_kind") or {}
for kind, slot in sorted(per.items()):
    n = slot.get("n", 0)
    s = slot.get("holdout_success", 0)
    rate = (s + 1) / (n + 2) if n else 0.5
    print(f"  {kind:24s} n={n:4d} s={s:4d} rate={rate:.4f}")

# Now the gate's view: which candidate-kind SUBSETS stay silent / fire?
prior = ExperienceMutationPrior(store, fp, strength=0.5, cache_ttl=1e9)
kinds = sorted(per.keys())
print("\npairwise gate check (which pairs keep the prior armed):")
from itertools import combinations
for a, b in combinations(kinds, 2):
    w = prior.kind_weights([a, b])
    if w is None:
        state = "GATED (None)"
    else:
        nu = prior._near_uniform(w)
        ratio = max(w.values()) / min(w.values())
        state = f"SILENT ratio={ratio:.3f} weights={ {k: round(v,3) for k,v in w.items()} }"
    print(f"  {a:22s} + {b:22s} -> {state}")
store.close()
shutil.rmtree(td, ignore_errors=True)
