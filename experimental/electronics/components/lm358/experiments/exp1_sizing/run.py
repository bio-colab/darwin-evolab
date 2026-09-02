"""Run LM358 Exp1 - reproducible via Oracle."""
from pathlib import Path
import hashlib, json, time
from experimental.electronics.components.lm358.evaluator import get_evaluator
from experimental.electronics.benchmarks.comparative_runner import ComparativeSizingBenchmark
from evolab.genome import FloatGenome

def run():
    ev = get_evaluator()
    bench = ComparativeSizingBenchmark(budget_evals=60, evaluator=ev)
    # also run single default point
    res_default = ev.evaluate(FloatGenome(ev.defaults))
    # Provenance is COMPUTED from the actual source file at run time —
    # never hardcoded. The original TI SLOS068AB PDF is not bundled; what is
    # hashed is the transcription text file.
    source_path = Path(__file__).resolve().parent.parent.parent / "source" / "SLOS068AB_LM358B.pdf.txt"
    report = {
        "experiment": "lm358_exp1_sizing",
        "source": "SLOS068AB_LM358B.pdf.txt",
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest().upper(),
        "sha256_of": "transcription text file; original TI SLOS068AB PDF not bundled in repository",
        "timestamp": int(time.time()),
        "tool_used": res_default.artifacts.get("tool_used"),
        "oracle_agreement": res_default.artifacts.get("oracle_agreement"),
        "default_score": res_default.score,
        "default_passed": res_default.passed_holdout,
        "specs": ev.specs,
        "benchmark": [r.__dict__ for r in bench.run_all()],
    }
    out = Path(__file__).parent.parent.parent / "results" / "exp1_sizing_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    # also global reports
    Path("experimental/electronics/reports/lm358_exp1.json").write_text(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    import json as _j
    print(_j.dumps(run(), indent=2))
