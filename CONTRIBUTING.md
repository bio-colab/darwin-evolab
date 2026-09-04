# Contributing to darwin-evolab

Thank you for your interest in contributing to **darwin-evolab** — the Universal Evolutionary Optimization & Synthesis Kernel!

We welcome contributions from researchers, systems engineers, hardware developers, and open-source enthusiasts worldwide.

---

## 🧭 Core Architectural Philosophy

Before writing code, please review our foundational design principle:

> **darwin-evolab is architected as an Evolutionary Operating System, not a toolbox.**

1. **Kernel vs. Drivers**:
   - The evolutionary kernel (`EvolutionEngine`, Speciation, Quality Diversity, NSGA-II) is strictly domain-agnostic.
   - Engineering domains are implemented as pluggable drivers inheriting from `DomainAdapter`.
   - Never add domain-specific logic (e.g. AST rules or SPICE parameters) inside the core kernel.

2. **Zero Inventions without Verification**:
   - Every algorithmic innovation must include measurable verification with pre-registered empirical protocols.
   - Negative results are valuable data; report them truthfully.

---

## 🛠️ Development Setup

### 1. Fork and Clone
```bash
git clone https://github.com/<your-username>/darwin-evolab.git
cd darwin-evolab
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Linux / macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install in Editable Mode
```bash
# Core framework
pip install -e .

# With optional scientific & verification dependencies
pip install -e ".[full]"
```

---

## 🧪 Testing Standards

Our repository strictly enforces a **100% test pass rate with zero regressions**:

```bash
# Run core test suite
pytest tests

# Run hardware & electronics test suite
pytest experimental/electronics/tests

# Run repository-wide test suite
pytest tests experimental/electronics/tests
```

### Contribution Invariants:
- All new features must include unit and integration tests under `tests/`.
- Never weaken existing assertion thresholds or tests.
- Code must pass Python 3.10+ syntax checks and type annotations (`mypy` / `pyright` clean).

---

## 🧩 Adding a New Domain Driver (`DomainAdapter`)

To bring a new engineering domain to `darwin-evolab` (e.g., molecular folding, aerodynamic shapes, robotics controllers), implement a subclass of `DomainAdapter`:

```python
from evolab.adapters import DomainAdapter, register_domain_adapter
from evolab.evaluators import Evaluator, FitnessResult
from evolab.genome import EvolabGenome, Individual

class MyDomainAdapter(DomainAdapter):
    @property
    def name(self) -> str:
        return "my_domain"

    def parse_spec(self, raw_input):
        # Ingest user specification
        return raw_input

    def build_population(self, spec, size, rng):
        # Generate initial candidate population
        return [Individual(...) for _ in range(size)]

    def build_evaluator(self, spec):
        # Return an Evaluator instance
        return MyEvaluator()

    def export_solution(self, individual, spec, output_path=None):
        # Export artifact (code, netlist, structure)
        return {"solution": ...}

# Register the driver into Darwin-Evolab
register_domain_adapter("my_domain", MyDomainAdapter())
```

See [`examples/03_custom_domain_adapter.py`](examples/03_custom_domain_adapter.py) for a complete runnable template.

---

## 📜 Scientific Pre-Registration Protocol

If you are evaluating search heuristics, memory banks, or evolutionary priors:
1. Write the decision rules ($R_1 \dots R_n$) and benchmark thresholds in the script header **before** running the experiment.
2. Do not adjust thresholds post-hoc after inspecting the outputs.
3. Save raw benchmark metrics in `reports/` as reproducible JSON files.

---

## 💬 Community & Discussion

- **GitHub Issues**: For bug reports, feature proposals, and architectural discussions.
- **Pull Requests**: Reference the relevant issue number and include full test output in the PR description.
- **Code of Conduct**: All participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

