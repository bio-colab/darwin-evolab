"""transfer.py — Low-level Memory Harness and Cross-Target Behavioral Transfer."""
from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .invariants import Observation, OracleVerdict, ViolationSeverity
from .taxonomy import FaultCategory


class MemoryArena:
    """Low-level memory allocation arena with prologue/epilogue guard canaries.
    Simulates authentic C-style buffer allocations and overflow detection."""

    CANARY_PROLOGUE = b"\xde\xad\xbe\xef\xca\xfe\xba\xbe"
    CANARY_EPILOGUE = b"\xba\xad\xf0\x0d\xfe\xed\xfa\xce"

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._raw = bytearray(self.CANARY_PROLOGUE + (b"\x00" * capacity) + self.CANARY_EPILOGUE)
        self._data_start = len(self.CANARY_PROLOGUE)
        self._data_end = self._data_start + capacity

    def write(self, data: bytes | bytearray | str, offset: int = 0) -> None:
        """Raw memory copy without bounds check, allowing real buffer overflow.
        Overruns within the arena smash the epilogue canary. Writes past the arena
        raise a hardware-level MemoryError (Segmentation Fault / Access Violation)."""
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")
        target_start = self._data_start + offset
        target_end = target_start + len(data)

        if target_start < 0 or target_end > len(self._raw):
            raise MemoryError(f"Hardware-Level Fault: write past mapped memory arena (offset={offset}, len={len(data)})")

        self._raw[target_start:target_end] = data

    def read(self, offset: int = 0, length: int = 16) -> bytes:
        target_start = self._data_start + offset
        target_end = target_start + length
        if target_start < 0 or target_end > len(self._raw):
            raise MemoryError(f"Hardware-Level Fault: read past mapped memory arena (offset={offset})")
        return bytes(self._raw[target_start:target_end])

    def check_canaries(self) -> tuple[bool, str]:
        """Verifies guard canary integrity."""
        prologue = bytes(self._raw[:len(self.CANARY_PROLOGUE)])
        epilogue = bytes(self._raw[-len(self.CANARY_EPILOGUE):])
        if prologue != self.CANARY_PROLOGUE:
            return False, "PROLOGUE_CANARY_OVERWRITTEN: Heap Underflow / Negative Offset Clashing"
        if epilogue != self.CANARY_EPILOGUE:
            return False, "EPILOGUE_CANARY_OVERWRITTEN: Heap Buffer Overflow / Epilogue Smashing"
        return True, "INTEGRITY_VERIFIED"


# ===========================================================================
# Pillar F: Cross-Target Transfer Memory (Generalization Across Targets)
# ===========================================================================

@dataclass
class TransferablePrimitive:
    """Represents an invariant-violating primitive learned from a specific target."""
    name: str
    feature_motif: list[float]
    discovered_fault: FaultCategory
    severity: ViolationSeverity
    source_target: str
    frequency: int = 1
    invariant_descriptor: list[float] = field(default_factory=list)

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_motif": self.feature_motif,
            "discovered_fault": self.discovered_fault.value,
            "severity": self.severity.name,
            "source_target": self.source_target,
            "frequency": self.frequency,
            "invariant_descriptor": self.invariant_descriptor,
        }


class PermutationSymmetricInputSummary:
    """Computes coordinate permutation-symmetric statistical summaries of input vectors.

    NOTE ON SCOPE AND METHODOLOGICAL LIMITATIONS:
    This summary is invariant ONLY under pure coordinate permutations and uniform isotropic shifts.
    It is NOT invariant under general or dimension-specific affine transformations (s_j != s_k or b_j != b_k),
    nor does it represent program execution behavior. True cross-target behavioral transfer requires
    runtime execution traces (e.g. branch transition entropy, differential state deltas, memory canary
    violations) rather than input-space coordinate moments.
    """

    @staticmethod
    def extract(values: Sequence[float]) -> tuple[float, float, float, float]:
        if not values:
            return (0.0, 0.0, 0.0, 0.0)
        n = len(values)
        mean = sum(values) / n
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        centered = sorted(v - mean for v in values)
        c0 = round(centered[0], 4) if n > 0 else 0.0
        c1 = round(centered[1], 4) if n > 1 else c0
        return (round(mean, 4), round(spread, 4), c0, c1)

    @staticmethod
    def distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# Backward compatibility alias
InvariantBehavioralDescriptor = PermutationSymmetricInputSummary


# ===========================================================================
# Pillar F: Pure Behavioral Security Transfer (Genome-Blind Architecture)
# ===========================================================================

@dataclass
class BehavioralSecurityDescriptor:
    """A representation-agnostic, genome-blind software security motif.

    Constructed strictly from runtime execution telemetry and oracle verdicts:
    1. fault_transitions: Transition bigrams between FaultCategory values (e.g. NORMAL_SUCCESS -> STATE_CORRUPTION).
    2. violated_invariants: Set/list of security invariant classes breached.
    3. max_severity: Peak severity rating observed across the execution trajectory.
    4. severity_progression: Sequence of severity levels per step.
    5. state_diff_keys: Canonical keys of mutated state subsystems (e.g. 'heap_canary', 'isolated_state').
    6. state_diff_cardinality: Total count/cardinality of corrupted or modified state elements.
    7. temporal_expansion_ratio: Execution time dilation (duration / baseline_duration).
    8. duration_ms: Average step execution time in milliseconds.
    """
    fault_transitions: list[tuple[str, str]] = field(default_factory=list)
    violated_invariants: list[str] = field(default_factory=list)
    max_severity: ViolationSeverity = ViolationSeverity.NONE
    severity_progression: list[int] = field(default_factory=list)
    state_diff_keys: list[str] = field(default_factory=list)
    state_diff_cardinality: int = 0
    temporal_expansion_ratio: float = 1.0
    duration_ms: float = 0.0
    output_digest: str = ""

    @classmethod
    def from_trajectory(
        cls,
        observations: Sequence[Observation],
        verdicts: Sequence[OracleVerdict] | None = None,
        baseline_duration_ms: float = 1.0,
    ) -> BehavioralSecurityDescriptor:
        """Constructs a behavioral descriptor from an execution trajectory without reading genome values."""
        if not observations:
            return cls()

        # 1. Fault transitions
        faults = [obs.fault_category.name for obs in observations]
        transitions: list[tuple[str, str]] = []
        if len(faults) == 1:
            transitions.append(("START", faults[0]))
        else:
            for i in range(len(faults) - 1):
                transitions.append((faults[i], faults[i + 1]))

        # 2. Invariants & severities
        violated_set: set[str] = set()
        severities: list[int] = []
        max_sev = ViolationSeverity.NONE

        if verdicts:
            for v in verdicts:
                severities.append(int(v.max_severity))
                if int(v.max_severity) > int(max_sev):
                    max_sev = v.max_severity
                for viol in v.violations:
                    violated_set.add(viol.invariant_name)
        else:
            # Direct extraction from observations
            for obs in observations:
                if obs.fault_category == FaultCategory.STATE_CORRUPTION:
                    violated_set.add("NoStateCorruptionInvariant")
                    severities.append(int(ViolationSeverity.HIGH))
                    if int(ViolationSeverity.HIGH) > int(max_sev):
                        max_sev = ViolationSeverity.HIGH
                elif obs.fault_category == FaultCategory.PRIVILEGE_BOUNDARY_VIOLATION:
                    violated_set.add("NoPrivilegeBoundaryViolationInvariant")
                    severities.append(int(ViolationSeverity.CRITICAL))
                    max_sev = ViolationSeverity.CRITICAL
                elif obs.fault_category == FaultCategory.RESOURCE_EXHAUSTION:
                    violated_set.add("NoResourceExhaustionInvariant")
                    severities.append(int(ViolationSeverity.HIGH))
                    if int(ViolationSeverity.HIGH) > int(max_sev):
                        max_sev = ViolationSeverity.HIGH
                elif obs.fault_category == FaultCategory.LOGIC_DEVIATION:
                    severities.append(int(ViolationSeverity.MEDIUM))
                    if int(ViolationSeverity.MEDIUM) > int(max_sev):
                        max_sev = ViolationSeverity.MEDIUM
                else:
                    severities.append(int(ViolationSeverity.NONE))

        # 3. State diff footprint
        all_keys: set[str] = set()
        cardinality = 0
        for obs in observations:
            if obs.state_diff:
                for k, v in obs.state_diff.items():
                    all_keys.add(str(k))
                    if isinstance(v, (list, tuple, set, dict)):
                        cardinality += len(v)
                    else:
                        cardinality += 1

        # 4. Temporal signature
        total_dur = sum(obs.duration_ms for obs in observations)
        avg_dur = total_dur / len(observations)
        expansion = round(avg_dur / max(0.001, baseline_duration_ms), 4)

        # 5. Semantic Output Digest / Deterministic Fingerprint
        rets = [str(obs.return_value) for obs in observations if obs.return_value is not None]
        out_digest = hashlib.sha256("||".join(rets).encode("utf-8")).hexdigest()[:12] if rets else ""

        return cls(
            fault_transitions=transitions,
            violated_invariants=sorted(violated_set),
            max_severity=max_sev,
            severity_progression=severities,
            state_diff_keys=sorted(all_keys),
            state_diff_cardinality=cardinality,
            temporal_expansion_ratio=expansion,
            duration_ms=round(avg_dur, 3),
            output_digest=out_digest,
        )

    def distance(self, other: BehavioralSecurityDescriptor, check_functional_digest: bool = False) -> float:
        """Computes representation-blind metric distance between two behavioral profiles in [0, 1]."""
        # 1. Invariant overlap distance (Jaccard)
        s1 = set(self.violated_invariants)
        s2 = set(other.violated_invariants)
        if not s1 and not s2:
            jaccard_dist = 0.0
        elif not s1 or not s2:
            jaccard_dist = 1.0
        else:
            jaccard_dist = 1.0 - len(s1 & s2) / len(s1 | s2)

        # 2. Fault transition distance
        t1 = set(self.fault_transitions)
        t2 = set(other.fault_transitions)
        if not t1 and not t2:
            trans_dist = 0.0
        elif not t1 or not t2:
            trans_dist = 1.0
        else:
            trans_dist = 1.0 - len(t1 & t2) / len(t1 | t2)

        # 3. State diff keys overlap
        k1 = set(self.state_diff_keys)
        k2 = set(other.state_diff_keys)
        if not k1 and not k2:
            key_dist = 0.0
        elif not k1 or not k2:
            key_dist = 0.8
        else:
            key_dist = 1.0 - len(k1 & k2) / len(k1 | k2)

        # 4. Severity discrepancy
        sev_dist = abs(int(self.max_severity) - int(other.max_severity)) / 5.0

        # 5. Temporal expansion divergence
        temp_dist = min(2.0, abs(self.temporal_expansion_ratio - other.temporal_expansion_ratio)) / 2.0

        # 6. Functional output consistency check
        out_dist = 0.0
        if check_functional_digest and self.output_digest and other.output_digest:
            out_dist = 0.0 if self.output_digest == other.output_digest else 1.0

        if check_functional_digest and (self.output_digest or other.output_digest):
            composite = (
                0.25 * jaccard_dist +
                0.20 * trans_dist +
                0.15 * key_dist +
                0.15 * sev_dist +
                0.10 * temp_dist +
                0.15 * out_dist
            )
        else:
            composite = (
                0.30 * jaccard_dist +
                0.25 * trans_dist +
                0.20 * key_dist +
                0.15 * sev_dist +
                0.10 * temp_dist
            )
        return round(composite, 4)

    def serialize(self) -> dict[str, Any]:
        return {
            "fault_transitions": [list(t) for t in self.fault_transitions],
            "violated_invariants": self.violated_invariants,
            "max_severity": self.max_severity.name,
            "severity_progression": self.severity_progression,
            "state_diff_keys": self.state_diff_keys,
            "state_diff_cardinality": self.state_diff_cardinality,
            "temporal_expansion_ratio": self.temporal_expansion_ratio,
            "duration_ms": self.duration_ms,
            "output_digest": self.output_digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehavioralSecurityDescriptor:
        return cls(
            fault_transitions=[tuple(t) for t in data.get("fault_transitions", [])],
            violated_invariants=list(data.get("violated_invariants", [])),
            max_severity=ViolationSeverity[data.get("max_severity", "NONE")],
            severity_progression=list(data.get("severity_progression", [])),
            state_diff_keys=list(data.get("state_diff_keys", [])),
            state_diff_cardinality=int(data.get("state_diff_cardinality", 0)),
            temporal_expansion_ratio=float(data.get("temporal_expansion_ratio", 1.0)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            output_digest=str(data.get("output_digest", "")),
        )


@dataclass
class TransferableBehavioralMotif:
    """Represents a learned security defect motif completely decoupled from genome representations."""
    name: str
    descriptor: BehavioralSecurityDescriptor
    source_target: str
    discovered_fault: FaultCategory
    severity: ViolationSeverity
    frequency: int = 1

    def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "descriptor": self.descriptor.serialize(),
            "source_target": self.source_target,
            "discovered_fault": self.discovered_fault.value,
            "severity": self.severity.name,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferableBehavioralMotif:
        return cls(
            name=data["name"],
            descriptor=BehavioralSecurityDescriptor.from_dict(data["descriptor"]),
            source_target=data["source_target"],
            discovered_fault=FaultCategory(data["discovered_fault"]),
            severity=ViolationSeverity[data["severity"]],
            frequency=data.get("frequency", 1),
        )


class CrossTargetTransferMemory:
    """Retains evolutionary exploit motifs across multiple distinct software targets,
    enabling transfer learning and few-shot cross-target generalization."""

    def __init__(self):
        self.archive: list[TransferablePrimitive] = []
        self.behavioral_motifs: list[TransferableBehavioralMotif] = []

    def record_behavioral_motif(
        self,
        name: str,
        trajectory: Sequence[Observation],
        verdicts: Sequence[OracleVerdict] | None = None,
        source_target: str = "target",
    ) -> BehavioralSecurityDescriptor | None:
        """Stores a pure behavioral security motif learned from execution traces without inspecting genomes."""
        desc = BehavioralSecurityDescriptor.from_trajectory(trajectory, verdicts=verdicts)
        if int(desc.max_severity) < int(ViolationSeverity.MEDIUM):
            return None

        for motif in self.behavioral_motifs:
            if motif.name == name and motif.source_target == source_target:
                motif.frequency += 1
                if int(desc.max_severity) > int(motif.severity):
                    motif.descriptor = desc
                    motif.severity = desc.max_severity
                return desc

        primary_fault = FaultCategory.NORMAL_SUCCESS
        for obs in trajectory:
            if obs.fault_category != FaultCategory.NORMAL_SUCCESS:
                primary_fault = obs.fault_category
                break

        self.behavioral_motifs.append(TransferableBehavioralMotif(
            name=name,
            descriptor=desc,
            source_target=source_target,
            discovered_fault=primary_fault,
            severity=desc.max_severity,
        ))
        return desc

    def compute_behavioral_affinity(
        self,
        trajectory: Sequence[Observation],
        verdicts: Sequence[OracleVerdict] | None = None,
        max_bonus: float = 12.0,
    ) -> float:
        """Computes behavioral transfer guidance bonus without EVER inspecting candidate genomes."""
        if not self.behavioral_motifs:
            return 0.0

        cand_desc = BehavioralSecurityDescriptor.from_trajectory(trajectory, verdicts=verdicts)
        best_bonus = 0.0

        for motif in self.behavioral_motifs:
            dist = cand_desc.distance(motif.descriptor)
            bonus = max(0.0, max_bonus * (1.0 - 1.25 * dist))
            best_bonus = max(best_bonus, bonus)

        return round(best_bonus, 4)

    def record_exploit_motif(
        self,
        name: str,
        genome_values: Sequence[float],
        fault: FaultCategory,
        severity: ViolationSeverity,
        source_target: str,
    ) -> None:
        """Stores a vulnerability motif in the legacy transfer archive."""
        if int(severity) < int(ViolationSeverity.MEDIUM):
            return

        inv_desc = list(PermutationSymmetricInputSummary.extract(genome_values))

        for prim in self.archive:
            if prim.name == name and prim.source_target == source_target:
                prim.frequency += 1
                if not prim.invariant_descriptor:
                    prim.invariant_descriptor = inv_desc
                return

        self.archive.append(TransferablePrimitive(
            name=name,
            feature_motif=list(genome_values),
            discovered_fault=fault,
            severity=severity,
            source_target=source_target,
            invariant_descriptor=inv_desc,
        ))

    def compute_transfer_affinity(
        self,
        candidate_values: Sequence[float],
        max_bonus: float = 8.0,
        decay: float = 12.0,
    ) -> float:
        """Legacy coordinate-based input shape affinity."""
        if not self.archive:
            return 0.0
        cand_desc = PermutationSymmetricInputSummary.extract(candidate_values)
        best_bonus = 0.0
        for prim in self.archive:
            prim_desc = prim.invariant_descriptor or PermutationSymmetricInputSummary.extract(prim.feature_motif)
            d = PermutationSymmetricInputSummary.distance(cand_desc, prim_desc)
            bonus = max(0.0, max_bonus - decay * d)
            best_bonus = max(best_bonus, bonus)
        return round(best_bonus, 4)

    def seed_target_population(
        self,
        target_engine: Any,
        target_name: str,
        max_injections: int = 4,
    ) -> int:
        """Injects cross-target pioneer individuals into the target engine's population."""
        if not self.archive:
            return 0

        if not getattr(target_engine, "population", None):
            pop_size = getattr(target_engine, "population_size", 16)
            g_size = getattr(target_engine, "genome_size", 16)
            from ..genome import FloatGenome, Individual
            rng = getattr(target_engine, "rng", random)
            target_engine.population = [
                Individual(genome=FloatGenome(values=[rng.uniform(-5.0, 5.0) for _ in range(g_size)]), species="spec_default")
                for _ in range(pop_size)
            ]

        pop = target_engine.population
        pop_len = len(pop)
        if pop_len == 0:
            return 0

        injected = 0
        sorted_prims = sorted(self.archive, key=lambda p: (int(p.severity), p.frequency), reverse=True)

        for i, prim in enumerate(sorted_prims[:max_injections]):
            idx = pop_len - 1 - i
            if idx < 0:
                break
            ind = pop[idx]
            if hasattr(ind, "genome") and hasattr(ind.genome, "values"):
                g_len = len(ind.genome.values)
                vals = list(ind.genome.values)
                for j in range(min(g_len, len(prim.feature_motif))):
                    vals[j] = prim.feature_motif[j]
                ind.genome.values = vals
                ind.species = f"spec_transfer_{prim.source_target}"
                injected += 1

        return injected

    def export_to_dict(self) -> dict[str, Any]:
        return {
            "primitives": [p.serialize() for p in self.archive],
            "behavioral_motifs": [m.serialize() for m in self.behavioral_motifs],
        }

    @classmethod
    def load_from_dict(cls, data: dict[str, Any]) -> CrossTargetTransferMemory:
        inst = cls()
        for p in data.get("primitives", []):
            inst.archive.append(TransferablePrimitive(
                name=p["name"],
                feature_motif=p["feature_motif"],
                discovered_fault=FaultCategory(p["discovered_fault"]),
                severity=ViolationSeverity[p["severity"]],
                source_target=p["source_target"],
                frequency=p.get("frequency", 1),
                invariant_descriptor=p.get("invariant_descriptor", []),
            ))
        for m in data.get("behavioral_motifs", []):
            inst.behavioral_motifs.append(TransferableBehavioralMotif.from_dict(m))
        return inst


