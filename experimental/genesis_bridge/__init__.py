"""
experimental/genesis_bridge — Genesis physics simulator & foundation model evolutionary bridge.

Status: FROZEN / EXPLORATORY RESEARCH TRACK (physical_claim: false)
This exploratory research track bridges the universal evolutionary kernel with external physical
simulation environments (Genesis) and foundation models via tensor and GNN representations.
Preserved in a frozen state under experimental/ for academic reproducibility.
"""
from __future__ import annotations

from .bridge import (
    FoundationModelPrior,
    GenesisBridge,
    GenesisEnvironment,
    GenesisRewardVector,
    MockGenesisSimulator,
    NativeGenesisEnvironment,
    RemoteGenesisEnvironment,
    deserialize_from_tensor,
    serialize_for_foundation_model,
)

__all__ = [
    "GenesisBridge",
    "GenesisRewardVector",
    "GenesisEnvironment",
    "MockGenesisSimulator",
    "RemoteGenesisEnvironment",
    "NativeGenesisEnvironment",
    "serialize_for_foundation_model",
    "deserialize_from_tensor",
    "FoundationModelPrior",
]
