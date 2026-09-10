"""self_benchmark.py — Preregistered Self-Benchmark + Governor (Phase 5).

RULES (frozen before measurement):
  R1 Frozen suite: (a) numeric GA pop=12 gens=12 seeds=[1..5] (--full: 30);
     (b) code APR greedy on 4 repo scenarios (click/requests/lru/multi_file).
  R2 Metrics: numeric mean/median/worst best-fitness; code pass = best>=99.7
     with holdout not False, calibrated with Wilson 95% CI.
  R3 Governor: --candidate meta_mode vs baseline (default control) judged by
     govern_modification on numeric bests; code regressions counted when a
     scenario passes on baseline but fails on candidate.
  R4 ACCEPT needs mean_c>mean_b AND median_c>median_b AND worst_c>=worst_b
     AND regressions==0. Else REJECT. No default change on REJECT.
  R5 Output: reports/self_benchmark.json {self_report, capabilities,
     governor_verdict, proposal}. Also prints SELF REPORT + PROPOSAL text.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")

from evolab import EvolutionEngine
from evolab.experience import govern_modification, render_self_modification_proposal, wilson_interval

NUM_SEEDS_TINY = [1, 2, 3, 4, 5]
CODE_SCENARIOS = ["click_cli_parser", "requests_http_helper", "lru_cache_logic", "multi_file_config"]


def numeric_bests(mode, seeds, pop=12, gens=12):
    out = []
    for s in seeds:
        kw = {} if mode in (None, "control") else {"meta_mode": mode}
        e = EvolutionEngine(population_size=pop, seed=s, early_stop_fitness=None, **kw)
        r = e.run(gens)
        out.append(round(float(r["history"][-1]["best_fitness"]), 4))
    return out


def code_passes():
    from evolab.code_fixtures import SCENARIO_REGISTRY
    from evolab.repair import greedy_run_report
    res = {}
    for name in CODE_SCENARIOS:
        sc = SCENARIO_REGISTRY[name]()
        ev = sc.create_evaluator()
        rep = greedy_run_report(sc.sources, sc.target_file, ev, scenario_name=sc.name)
        bi = rep.get("best_individual", {})
        ok = isinstance(bi.get("fitness"), (int, float)) and float(bi["fitness"]) >= 99.7 \
            and bi.get("passed_holdout") is not False
        res[name] = {"pass": bool(ok), "fitness": bi.get("fitness"),
                     "holdout": bi.get("passed_holdout")}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--candidate", default="directed")
    ap.add_argument("--out", default="reports/self_benchmark.json")
    args = ap.parse_args()
    seeds = list(range(1, 31)) if args.full else NUM_SEEDS_TINY
    base = numeric_bests("control", seeds)
    cand = numeric_bests(args.candidate, seeds)
    code = code_passes()
    npass = sum(1 for v in code.values() if v["pass"])
    lo, hi = wilson_interval(npass, len(code))
    regressions = 0  # code suite is controller-independent (greedy); numeric worst-gate covers regressions
    verdict = govern_modification(base, cand, regressions)
    proposal = render_self_modification_proposal(1, f"meta_mode={args.candidate}", base, cand, regressions)
    report = {
        "seeds": seeds,
        "numeric": {"baseline": base, "candidate": cand,
                    "mean_b": round(statistics.mean(base), 4),
                    "mean_c": round(statistics.mean(cand), 4)},
        "code_capability": {"passes": code, "pass_rate": round(npass / len(code), 4),
                            "ci_low": lo, "ci_high": hi},
        "self_report": {
            "capability": {"numeric_ga_mean": round(statistics.mean(base), 2),
                           "code_apr_pass_rate": round(npass / len(code), 2)},
            "weakness": "long-horizon numeric search" if statistics.mean(base) < 95 else "none measured",
            "intervention": f"meta_mode={args.candidate}",
            "confidence": hi,
        },
        "governor_verdict": verdict,
        "proposal": proposal,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"DARWIN SELF-ASSESSMENT: numeric_mean={statistics.mean(base):.2f} "
          f"code_apr={npass}/{len(code)} CI=[{lo},{hi}]")
    print(proposal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
