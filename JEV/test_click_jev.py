import time
from evolab.code_fixtures import scenario_click_parser
from evolab.repair import RepairGenome, catalog_sources, _score
from JEV.jev_client import JevClient

client = JevClient()
scenario = scenario_click_parser()
evaluator = scenario.create_evaluator()

sources = dict(scenario.sources)
target_file = scenario.target_file

current = RepairGenome(sources=sources, target_file=target_file, edits=[])
best_score, best_hold = _score(evaluator, current)
evaluations = 1
taken = set(current.edit_keys())
gen = 1
history = [{"generation": 1, "best_fitness": best_score, "edits": 0}]

t0 = time.perf_counter()
catalog = catalog_sources(sources)

improved = True
while improved and best_score < 100.0:
    improved = False
    current_res = evaluator.evaluate(current)
    failures = current_res.artifacts.get("failures", [])
    failure_text = failures[0] if failures else "Tests failing"
    current_code = current.to_code()

    available_edits = [e for e in catalog if e.locus() not in taken]
    available_kinds = sorted(list(set(e.kind for e in available_edits)))

    # Ask Jev for operator guidance
    probs = client.prioritize_operators(current_code, failure_text, available_kinds)
    print(f"\n--- Generation {gen} ---")
    print(f"Current Failure: {failure_text}")
    print(f"JEV Kind Probabilities: {probs}")

    # Sort available edits by Jev probability
    def _jev_key(e):
        p = probs.get(e.kind, 0.0)
        return (-p, e.file, e.lineno, e.col_offset)

    sorted_edits = sorted(available_edits, key=_jev_key)

    best_trial = None
    best_trial_score = best_score
    best_trial_hold = best_hold
    best_edit = None

    for edit in sorted_edits:
        trial = RepairGenome(
            sources=dict(sources),
            target_file=target_file,
            edits=current.edits + [edit],
        )
        score, hold = _score(evaluator, trial)
        evaluations += 1

        if score <= best_score:
            continue
        if hold is False and best_hold is True:
            continue

        if score > best_trial_score:
            best_trial = trial
            best_trial_score = score
            best_trial_hold = hold
            best_edit = edit
            print(f"  Candidate {edit.kind} L{edit.lineno}: score {best_score} -> {score} (Eval {evaluations})")
            # In first-ascent greedy, we take the first improving candidate
            break

    if best_trial is not None and best_edit is not None:
        current = best_trial
        best_score = best_trial_score
        best_hold = best_trial_hold
        taken.add(best_edit.locus())
        improved = True
        history.append({
            "generation": gen + 1,
            "best_fitness": best_score,
            "edits": len(current.edits),
            "added": best_edit.kind,
            "lineno": best_edit.lineno,
        })
        gen += 1

duration = time.perf_counter() - t0
print("\n" + "=" * 60)
print(f"JEV-GUIDED REPAIR COMPLETED:")
print(f"Final Score: {best_score} / 100.0 (Holdout Passed: {best_hold})")
print(f"Total Evals Consumed: {evaluations}")
print(f"Total Wall Time: {duration:.2f}s (Jev API time: {client.total_api_time_seconds:.2f}s, Engine time: {duration - client.total_api_time_seconds:.3f}s)")
print(f"Total Jev Calls: {client.total_calls}, Tokens: {client.total_input_tokens} in / {client.total_output_tokens} out")
print("Edits Applied:")
for e in current.edits:
    print(f"  - {e.kind} at line {e.lineno}:{e.col_offset}")
