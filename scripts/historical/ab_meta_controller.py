"""ab_meta_controller.py — Preregistered A/B for the Phase 4 Meta-Controller.

RULES (frozen before measurement — tuning after looking is forbidden):
  R1 Frozen suite: numeric GA on default_fitness, pop=12, gens=15,
     seeds=1..10 (use --tiny for 3 seeds smoke test).
  R2 Three arms: control (meta off) vs directed vs reshuffle (noise arm).
  R3 Primary metric: mean best fitness; secondary: median, worst.
  R4 Default-eligibility: directed is eligible ONLY on govern ACCEPT
     (mean_c>mean_b AND median_c>median_b AND worst_c>=worst_b AND
     regressions==0). Reshuffle isolates trigger value from injection
     noise: if reshuffle ~= directed, the effect is noise, not diagnosis.
  R5 Output: reports/ab_meta_controller.json {arms, stats, governor_verdict}.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")

from evolab import EvolutionEngine
from evolab.experience import govern_modification

POP, GENS = 12, 15
SEEDS_FULL = list(range(1, 11))
SEEDS_TINY = [1, 2, 3]


def run_arm(mode, seeds):
    bests = []
    for s in seeds:
        kw = {} if mode == "control" else {"meta_mode": mode}
        e = EvolutionEngine(population_size=POP, seed=s, early_stop_fitness=None, **kw)
        r = e.run(GENS)
        bests.append(round(float(r["history"][-1]["best_fitness"]), 4))
    return bests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default="reports/ab_meta_controller.json")
    args = ap.parse_args()
    seeds = SEEDS_TINY if args.tiny else SEEDS_FULL
    arms = {m: run_arm(m, seeds) for m in ("control", "directed", "reshuffle")}
    stats = {m: {"mean": round(statistics.mean(v), 4),
                 "median": round(statistics.median(v), 4),
                 "worst": round(min(v), 4), "bests": v} for m, v in arms.items()}
    verdict = govern_modification(arms["control"], arms["directed"])
    reshuffle_check = govern_modification(arms["control"], arms["reshuffle"])
    out = {"seeds": seeds, "pop": POP, "gens": GENS, "arms": arms, "stats": stats,
           "governor_verdict": verdict, "reshuffle_check": reshuffle_check,
           "default_eligible": verdict["decision"] == "ACCEPT",
           "note": "directed needs ACCEPT while reshuffle stays REJECT to prove trigger value"}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("governor:", verdict["decision"], verdict["reasons"])
    print("reshuffle:", reshuffle_check["decision"], reshuffle_check["reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
