"""
motifs.py — Canonical biological neural motifs from the Drosophila Connectome.

Implements ground-truth behavioral models and benchmark time-series datasets:
1. Hassenstein-Reichardt Elementary Motion Detector (EMD):
   Models the T4/T5 directional selectivity circuit in the Drosophila visual system.
   Detects directional motion (Right vs. Left) via asymmetric temporal correlation:
   Rightward: R1(t - tau) AND R2(t)
   Leftward:  R2(t - tau) AND R1(t)

2. Olfactory Antennal Lobe Lateral Inhibition:
   Models local interneuron (LN) lateral inhibition between glomeruli to sharpen
   odor recognition contrast before projecting to Kenyon cells.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MotifDataset:
    """Container holding sensory input sequences and target neural responses."""

    motif_name: str
    inputs: list[list[int]]         # Timesteps x NumInputs (binary spike train)
    expected_outputs: list[list[int]]  # Timesteps x NumOutputs (expected decision spikes)
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return len(self.inputs)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "motif_name": self.motif_name,
            "length": self.length,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "metadata": self.metadata,
        }


def generate_emd_dataset(
    length: int = 32,
    seed: int = 42,
) -> MotifDataset:
    """
    Generates a visual motion detection sequence for two adjacent photoreceptors (R1, R2).

    Simulates dark/light edges moving across the visual field:
    - Left-to-Right edge: R1 fires at step t, R2 fires at step t+1 -> Right detector fires at t+1.
    - Right-to-Left edge: R2 fires at step t, R1 fires at step t+1 -> Left detector fires at t+1.
    - Static or random noise: Neither motion detector should fire.
    """
    rng = random.Random(seed)
    inputs: list[list[int]] = [[0, 0] for _ in range(length)]
    expected: list[list[int]] = [[0, 0] for _ in range(length)]

    t = 1
    events = []
    while t < length - 2:
        motion_type = rng.choice(["right", "left", "static", "flicker"])
        if motion_type == "right":
            # R1 activates at t, R2 activates at t+1
            inputs[t][0] = 1
            inputs[t + 1][1] = 1
            expected[t + 1][0] = 1  # Right motion detected
            events.append({"step": t, "type": "motion_right"})
            t += 3
        elif motion_type == "left":
            # R2 activates at t, R1 activates at t+1
            inputs[t][1] = 1
            inputs[t + 1][0] = 1
            expected[t + 1][1] = 1  # Left motion detected
            events.append({"step": t, "type": "motion_left"})
            t += 3
        elif motion_type == "flicker":
            # Both activate simultaneously (no direction)
            inputs[t][0] = 1
            inputs[t][1] = 1
            events.append({"step": t, "type": "flicker_simultaneous"})
            t += 2
        else:
            # Silence
            t += 1

    return MotifDataset(
        motif_name="hassenstein_reichardt_emd",
        inputs=inputs,
        expected_outputs=expected,
        metadata={
            "motif": "hassenstein_reichardt_emd",
            "total_timesteps": length,
            "events": len(events),
            "provenance": "Drosophila_T4_T5_EMD",
        },
    )


def generate_olfactory_dataset(
    length: int = 32,
    seed: int = 42,
) -> MotifDataset:
    """
    Generates an olfactory glomerular activation sequence with lateral inhibition.

    Inputs represent 3 odor receptors [O1, O2, O3].
    Lateral inhibition contract:
    - When strong primary odor O1 is active, it inhibits secondary weaker odors O2 and O3.
    - Dominant receptor wins and emits a clean projection spike.
    """
    rng = random.Random(seed)
    inputs: list[list[int]] = []
    expected: list[list[int]] = []

    for _ in range(length):
        r1 = rng.choice([0, 1])
        r2 = rng.choice([0, 1])
        r3 = rng.choice([0, 1])
        inputs.append([r1, r2, r3])

        # Glomerulus 1 has strongest lateral inhibition priority
        o1 = r1
        o2 = r2 if not r1 else 0
        o3 = r3 if (not r1 and not r2) else 0
        expected.append([o1, o2, o3])

    return MotifDataset(
        motif_name="olfactory_lateral_inhibition",
        inputs=inputs,
        expected_outputs=expected,
        metadata={
            "motif": "olfactory_lateral_inhibition",
            "total_timesteps": length,
            "provenance": "Drosophila_AntennalLobe_LN",
        },
    )
