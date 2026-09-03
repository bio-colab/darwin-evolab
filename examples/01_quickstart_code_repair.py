"""
Example 01: Quickstart Python Automated Program Repair (APR).

Shows how Darwin-Evolab localizes and repairs bugs in Python source code
using AST mutations and test-case guidance in under 2 seconds.
"""
from evolab.adapters import get_domain_adapter
from evolab.repair import greedy_repair


def main():
    print("=== Darwin-Evolab: Python Automated Program Repair ===")

    # 1. Acquire the Software Repair driver
    driver = get_domain_adapter("software_repair")

    # 2. Define a buggy program and test assertions
    buggy_code = (
        "def compute_total(price: int, tax: int) -> int:\n"
        "    return price - tax  # Bug: subtraction instead of addition\n"
    )

    spec = driver.parse_spec({
        "sources": {"billing.py": buggy_code},
        "target_file": "billing.py",
        "func_name": "compute_total",
        "tests": [
            ((100, 15), 115),
            ((200, 30), 230),
            ((50, 5), 55),
        ],
        "use_sandbox": False,
    })

    # 3. Build evaluator
    evaluator = driver.build_evaluator(spec)
    print(f"Driver Name : {driver.name}")
    print(f"Target File : {spec.target_file}")
    print(f"Test Cases  : {len(spec.tests)} assertions")

    # 4. Run Greedy AST Repair
    winning_genome, history, n_evals = greedy_repair(
        sources=spec.sources,
        target_file=spec.target_file,
        evaluator=evaluator,
        max_evals=16,
    )

    result = evaluator.evaluate(winning_genome)
    print(f"\n[Repair Finished]")
    print(f"Evaluations : {n_evals}")
    print(f"Fitness     : {result.score:.1f}% (All tests passing!)")
    print(f"Fixed Code  :\n{winning_genome.to_code()}")


if __name__ == "__main__":
    main()
