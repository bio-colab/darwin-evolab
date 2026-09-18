# Genesis & Foundation Model Evolutionary Bridge (`experimental/genesis_bridge/`)

> **Status: FROZEN / EXPLORATORY RESEARCH TRACK (`physical_claim: false`)**  
> *This exploratory module is preserved in a frozen state under `experimental/` for theoretical investigation and academic reproducibility. Core development is focused on Pillar 1 (Software APR) and Pillar 2 (Digital CGP).*

---

## 1. Overview

The **Genesis Bridge** connects Darwin-Evolab's domain-agnostic evolutionary engine with external foundational environments:
- **Genesis Physics Simulator**: High-speed, vectorized physical simulator (`genesis-world`) or remote cluster endpoints.
- **Foundation Models & GNNs**: Tensor serialization of code ASTs, silicon CGP DAGs, and continuous vectors.
- **Multi-Channel Reward Streaming**: Decouples scalar evaluation into structured, multi-objective Pareto reward vectors.
- **Mock Fallback**: High-speed, zero-dependency `MockGenesisSimulator` ensuring offline reproducibility and headless CI execution.

---

## 2. Architecture & Protocol

```
                        🧬 Darwin-Evolab Kernel
                                  │
                       [EvolabGenome / Individual]
                                  │
                                  ▼
                     serialize_for_foundation_model()
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
        [Graph / GNN Tensor]          [Continuous Vector Tensor]
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                        GenesisEnvironment
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
  MockGenesisSimulator   NativeGenesisEnvironment  RemoteGenesisEnvironment
   (Headless / Mock)       (genesis-world local)      (REST / Cluster)
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
                         GenesisRewardVector
             (primary_fitness, channel_rewards, latency)
```

---

## 3. Usage Example

```python
from experimental.genesis_bridge import (
    GenesisBridge,
    MockGenesisSimulator,
    serialize_for_foundation_model,
)
from evolab.engine import EvolutionEngine

# 1. Initialize bridge with zero-dependency mock simulator
simulator = MockGenesisSimulator(domain="physics_and_silicon")
bridge = GenesisBridge(environment=simulator)

# 2. Attach to evolutionary engine
engine = EvolutionEngine(
    population_size=16,
    genome_size=3,
    fitness_fn=bridge.attach_to_engine(None, objective_channel="stability"),
)

# 3. Serialize genome to GNN tensor for external foundation models
# tensor_repr = serialize_for_foundation_model(best_individual)
```

---

## 4. Integrity & Provenance Notice

This track carries `physical_claim: false`. In environments where `genesis-world` is not installed, the bridge defaults cleanly to `MockGenesisSimulator` with explicit logging to prevent false assertions of physical realism.
