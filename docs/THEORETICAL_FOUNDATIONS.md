# Theoretical Foundations & Classical References

> **Formal Academic Treatise on the Evolutionary and Algorithmic Foundations of Darwin-Evolab**  
> *Author:* Bio-Colab Research  
> *Document Identifier:* `EVOLAB-TR-2026-01`  
> *Status:* Reference Architecture & Theoretical Canon

---

## 1. Abstract & Epistemic Scope

`darwin-evolab` is formulated as a domain-agnostic, discrete-continuous evolutionary optimization kernel. Rather than relying on heuristic or ad-hoc stochastic search strategies, the kernel and its domain drivers instantiate the mathematical and computational principles established in classical evolutionary computation literature. 

This document articulates the formal theoretical grounding of the three operational tracks within `darwin-evolab`:
1. **Digital Logic & Circuit Netlist Synthesis**, grounded in Cartesian Genetic Programming (Miller, 1999, 2011).
2. **Software Automated Program Repair & SPICE Synthesis**, grounded in Tree-based Genetic Programming, Parsimony Pressure, and Circuit Embryology (Koza, 1992, 1999).
3. **Adaptive Sampling, Causal Credit Assignment, & Trial Allocation**, grounded in Schema Theory and Adaptive Systems (Holland, 1975).

---

## 2. Cartesian Genetic Programming (CGP) & Neutral Drift

### 2.1 Theoretical Formulation (Miller, 1999, 2011)
Cartesian Genetic Programming represents computational graphs as a two-dimensional grid of $N_r 	imes N_c$ nodes, where each node $n_i$ computes a function $f \in \mathcal{F}$ over inputs selected from previous columns (strictly feed-forward, acyclic directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$).

The defining mathematical property of CGP is the distinction between:
- **Genotype ($\mathcal{G}_{	ext{geno}}$)**: The complete array of encoded nodes and interconnects, of fixed cardinality $|\mathcal{V}|$.
- **Phenotype ($\mathcal{G}_{	ext{pheno}}$)**: The active sub-DAG obtained via backward reachability analysis from primary outputs $\mathcal{O}$:
  $$\mathcal{V}_{	ext{active}} = \{ v \in \mathcal{V} \mid \exists w \in \mathcal{O} 	ext{ such that } v ightsquigarrow w \}$$

### 2.2 Neutral Genetic Drift & Escape from Local Extrema
In standard Genetic Algorithms, search plateaus ($
abla f = 0$) induce stagnation. Miller demonstrated that the non-coding nodes ($\mathcal{V}_{	ext{neutral}} = \mathcal{V} \setminus \mathcal{V}_{	ext{active}}$) act as an unconstrained reservoir for neutral genetic drift.

Under Miller's canonical $(1 + \lambda)$ Evolutionary Strategy:
$$\mathcal{P}_{t+1} = egin{cases} 
\mathcal{C}^* & 	ext{if } f(\mathcal{C}^*) \ge f(\mathcal{P}_t) \
\mathcal{P}_t & 	ext{otherwise}
\end{cases}$$
where $\mathcal{C}^* = rg\max_{c \in \{\mathcal{C}_1, \dots, \mathcal{C}_\lambda\}} f(c)$.

The strict weak inequality ($\ge$) permits the parent to be replaced by an offspring of identical phenotypic fitness ($\Delta f = 0$), allowing the genotype to traverse neutral networks in genotype space until discovering an escape portal to higher fitness basins.

### 2.3 Implementation in `darwin-evolab`
- **Module**: `src/evolab/cgp_logic.py`
- **Active Subgraph Extraction**: `CGPGenome.get_active_nodes()` executes linear-time backward depth-first search from primary output indices, isolating non-coding gates prior to CMOS transistor accounting and logic verification.
- **Topological Integrity**: The gate netlist ensures 100% formal truth-table compliance without feedback loops.

---

## 3. Program Synthesis, AST Manifolds, & Parsimony Pressure

### 3.1 Theoretical Formulation (Koza, 1992, 1999)
Koza formalized computer program synthesis as natural selection over executable Abstract Syntax Trees (ASTs). The search space $\mathcal{S}$ is defined by:
1. **Terminal Set $\mathcal{T}$**: Constants, variables, and zero-arity functions.
2. **Function Set $\mathcal{F}$**: Syntactic operators and control-flow expressions.
3. **Closure**: Every function $f \in \mathcal{F}$ must accept as arguments any data type emitted by any terminal $t \in \mathcal{T}$ or function $f' \in \mathcal{F}$.
4. **Sufficiency**: The union $\mathcal{T} \cup \mathcal{F}$ must span a manifold capable of expressing a solution to the target problem.

### 3.2 Parsimony Pressure & Code Bloat Mitigation
In program evolution, genotypes tend to grow monotonically with unexecuted or functionally redundant code fragments (syntactic introns or "bloat"). Koza established parsimony pressure to penalize excessive structural complexity:
$$f_{	ext{parsimonious}}(P) = f_{	ext{empirical}}(P) - lpha \cdot \Omega(P)$$
where $\Omega(P)$ denotes a structural complexity measure (e.g., node count or edit distance), and $lpha > 0$ governs parsimony regularization.

### 3.3 Analog Synthesis via SPICE-in-the-Loop
In *Genetic Programming III* (Koza et al., 1999), Koza demonstrated the automated synthesis of analog circuits (such as operational amplifiers and elliptic filters) using embryonic electrical nodes modified by developmental transforms and evaluated via SPICE circuit simulations.

### 3.4 Implementation in `darwin-evolab`
- **Module**: `src/evolab/repair.py` & `src/evolab/swe_bench.py`
- **AST Edit Grammar**: Restricts mutation to atomic, syntax-preserving AST operations (`DeleteStatement`, `InsertGuard`, `SwapCondition`, `ReplaceConstant`, `CallWrap`), guaranteeing syntactic closure without invalid syntax generation.
- **Spectrum-Based Fault Localization (SBFL)**: Incorporates Ochiai suspicion coefficients to restrict the functional terminal set to high-entropy code loci.
- **Analog Silicon Track**: `experimental/electronics/models/ngspice_bridge.py` and `src/evolab/silicon/opamp_benchmark.py` embody Koza's SPICE-in-the-loop paradigm for SkyWater 130nm CMOS operational amplifier optimization.

---

## 4. Complex Adaptive Systems, Schema Theory, & Trial Allocation

### 4.1 Theoretical Formulation (Holland, 1975)
Holland provided the foundational mathematical framework for adaptation in natural and artificial systems. Central to this framework is the **Schema Theorem**:
$$\mathbb{E}[m(H, t+1)] \ge m(H, t) \cdot rac{f(H)}{ar{f}(t)} \left[ 1 - p_c rac{\delta(H)}{l - 1} - o(H) p_m ight]$$
where:
- $m(H, t)$ is the representation count of schema $H$ at generation $t$,
- $f(H)$ is the mean fitness of individuals sampling schema $H$,
- $ar{f}(t)$ is the population mean fitness,
- $\delta(H)$ is the defining length of the schema,
- $o(H)$ is the schema order,
- $p_c, p_m$ denote crossover and mutation probabilities.

The theorem proves that short, low-order, above-average schemata (building blocks) receive exponentially increasing trials over generations.

### 4.2 Multi-Armed Bandit Allocation & Credit Assignment
Holland formulated the trade-off between exploration (sampling unknown regions) and exploitation (refining known high-yield regions) as an optimal allocation of trials under a $k$-armed bandit framework. The optimal policy allocates trials exponentially to the observed highest-payoff operator while retaining log-bounded exploration of alternate operators.

Furthermore, Holland's Bucket Brigade algorithm established the formal mechanism of credit assignment, propagating fitness rewards backward through causal chains to credit antecedent state-transition operators.

### 4.3 Implementation in `darwin-evolab`
- **Module**: `src/evolab/engine.py`, `src/evolab/causal.py`, & `src/evolab/priors.py`
- **Causal Delta Tracking**: `_causal_events` logs the empirical fitness transition $\Delta f = f_{	ext{child}} - ar{f}_{	ext{parents}}$ attributable to specific mutation classes.
- **Empirical Experience Prior**: `ExperienceMutationPrior` computes Laplace-smoothed sampling weights:
  $$w_k = (1 - \lambda) + \lambda \cdot rac{s_k + lpha}{n_k + 2lpha}$$
  instantiating Holland's adaptive trial allocation over AST mutation operators with explicit zero-signal gating ($M_6$).

---

## 5. Formal Scholarly Bibliography

```bibtex
@book{holland1975adaptation,
  author    = {Holland, John H.},
  title     = {Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence},
  publisher = {University of Michigan Press},
  address   = {Ann Arbor, MI},
  year      = {1975},
  isbn      = {978-0262581110}
}

@book{koza1992genetic,
  author    = {Koza, John R.},
  title     = {Genetic Programming: On the Programming of Computers by Means of Natural Selection},
  publisher = {MIT Press},
  address   = {Cambridge, MA},
  year      = {1992},
  isbn      = {978-0262111706}
}

@book{koza1999genetic3,
  author    = {Koza, John R. and Bennett, Forrest H. and Andre, David and Keane, Martin A.},
  title     = {Genetic Programming III: Darwinian Invention and Problem Solving},
  publisher = {Morgan Kaufmann},
  address   = {San Francisco, CA},
  year      = {1999},
  isbn      = {978-1558605435}
}

@inproceedings{miller1999cgp,
  author    = {Miller, Julian F. and Thomson, Peter},
  title     = {An Empirical Investigation of the Efficiency of Learning Boolean Functions Using Cartesian Genetic Programming},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  pages     = {1161--1168},
  year      = {1999}
}

@book{miller2011cartesian,
  author    = {Miller, Julian F.},
  title     = {Cartesian Genetic Programming},
  series    = {Natural Computing Series},
  publisher = {Springer-Verlag},
  address   = {Berlin, Heidelberg},
  year      = {2011},
  doi       = {10.1007/978-3-642-17310-3},
  isbn      = {978-3642173097}
}
```

---

*This document serves as the governing theoretical specification for `darwin-evolab`. Any algorithmic modifications to the core kernel must preserve compatibility with these classical formulations.*
