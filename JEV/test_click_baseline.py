import time
from evolab.code_fixtures import scenario_click_parser
from evolab.repair import greedy_repair

scenario = scenario_click_parser()
evaluator = scenario.create_evaluator()

t0 = time.perf_counter()
winning_genome, history, n_evals = greedy_repair(
    sources=scenario.sources,
    target_file=scenario.target_file,
    evaluator=evaluator,
    max_evals=250,
    prioritize_by_suspicion=True,
)
duration = time.perf_counter() - t0
final_score = evaluator.evaluate(winning_genome).score

print(f"BASELINE: Final Score={final_score}, Evals={n_evals}, Time={duration*1000:.1f}ms")
print("History:", history)
print("Edits:", [e.serialize() for e in winning_genome.edits])
