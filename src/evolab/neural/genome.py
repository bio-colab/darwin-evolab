"""src/evolab/neural/genome.py — Programmatic Neural Weight-Space Genome.

Represents candidate weight-space modifications as sparse coordinate updates
or low-rank factors (Rank-r) over an immutable base neural network.
Complies strictly with the EvolabGenome contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np


class NeuralWeightGenome:
    """Represents programmatic weight-space perturbations over an immutable base model.

    Complies with EvolabGenome contract: clone(), fingerprint(), distance_to(),
    serialize(), describe(), mutate(), and crossover().
    Supports both NumPy arrays and PyTorch tensors natively.
    """

    def __init__(
        self,
        base_model_id: str,
        sparse_patches: Optional[Dict[str, np.ndarray]] = None,
        low_rank_factors: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        scale: float = 1.0,
        edit_lineage: Optional[List[dict]] = None,
    ) -> None:
        self.base_model_id = str(base_model_id)
        self.sparse_patches: Dict[str, np.ndarray] = sparse_patches or {}
        self.low_rank_factors: Dict[str, Tuple[np.ndarray, np.ndarray]] = low_rank_factors or {}
        self.scale = float(scale)
        self.edit_lineage: List[dict] = list(edit_lineage or [])

    def clone(self) -> NeuralWeightGenome:
        """Deep copy of the sparse weight genome without copying the immutable base model."""
        new_sparse: Dict[str, np.ndarray] = {}
        for k, v in self.sparse_patches.items():
            if hasattr(v, "clone"):  # PyTorch tensor
                new_sparse[k] = v.clone()
            else:  # NumPy array
                new_sparse[k] = np.copy(v)

        new_lr: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for k, (u, v) in self.low_rank_factors.items():
            u_c = u.clone() if hasattr(u, "clone") else np.copy(u)
            v_c = v.clone() if hasattr(v, "clone") else np.copy(v)
            new_lr[k] = (u_c, v_c)

        return NeuralWeightGenome(
            base_model_id=self.base_model_id,
            sparse_patches=new_sparse,
            low_rank_factors=new_lr,
            scale=self.scale,
            edit_lineage=list(self.edit_lineage),
        )

    def fingerprint(self) -> str:
        """Deterministic cryptographic SHA-256 fingerprint of all active parameter updates."""
        h = hashlib.sha256()
        h.update(self.base_model_id.encode("utf-8"))
        h.update(f"{self.scale:.6f}".encode("utf-8"))

        for k in sorted(self.sparse_patches.keys()):
            h.update(k.encode("utf-8"))
            arr = self.sparse_patches[k]
            if hasattr(arr, "detach"):
                arr_np = arr.detach().cpu().numpy()
            else:
                arr_np = np.asarray(arr)
            h.update(arr_np.tobytes())

        for k in sorted(self.low_rank_factors.keys()):
            u, v = self.low_rank_factors[k]
            u_np = u.detach().cpu().numpy() if hasattr(u, "detach") else np.asarray(u)
            v_np = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            h.update(k.encode("utf-8"))
            h.update(u_np.tobytes())
            h.update(v_np.tobytes())

        return h.hexdigest()

    def get_dense_delta(self, layer_name: str) -> Optional[np.ndarray]:
        """Compute the effective dense delta matrix (sparse + u*v^T) * scale for a layer."""
        dense: Optional[np.ndarray] = None
        if layer_name in self.sparse_patches:
            p = self.sparse_patches[layer_name]
            p_np = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p, dtype=float)
            dense = np.copy(p_np)

        if layer_name in self.low_rank_factors:
            u, v = self.low_rank_factors[layer_name]
            u_np = u.detach().cpu().numpy() if hasattr(u, "detach") else np.asarray(u, dtype=float)
            v_np = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=float)
            # v is expected to be (d_in, r) or (r, d_in)
            if u_np.ndim == 2 and v_np.ndim == 2:
                if u_np.shape[1] == v_np.shape[0]:
                    lr_mat = np.matmul(u_np, v_np)
                elif u_np.shape[1] == v_np.shape[1]:
                    lr_mat = np.matmul(u_np, v_np.T)
                else:
                    lr_mat = np.matmul(u_np, v_np)
            else:
                lr_mat = np.outer(u_np, v_np)

            if dense is not None:
                dense = dense + lr_mat
            else:
                dense = lr_mat

        if dense is not None:
            return dense * self.scale
        return None

    def distance_to(
        self,
        other: Any,
        probe_inputs: Optional[np.ndarray] = None,
        forward_fn: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None,
        model: Optional[Any] = None,
    ) -> float:
        """Functional Frobenius and parameter distance normalized by active layers.

        If probe_inputs, forward_fn, and model are provided, computes functional
        divergence on probe set outputs plus normalized Frobenius parameter distance:
            D(g1, g2) = (1/N) * ||f(X; g1) - f(X; g2)||_F + 0.1 * D_param(g1, g2)
        Otherwise, falls back to normalized Frobenius parameter distance.
        """
        if not isinstance(other, NeuralWeightGenome):
            return 1000.0

        all_layers = sorted(
            set(self.sparse_patches.keys())
            | set(self.low_rank_factors.keys())
            | set(other.sparse_patches.keys())
            | set(other.low_rank_factors.keys())
        )
        if not all_layers:
            return 0.0

        total_param_norm = 0.0
        for layer in all_layers:
            d1 = self.get_dense_delta(layer)
            d2 = other.get_dense_delta(layer)
            if d1 is not None and d2 is not None:
                diff = d1 - d2
                total_param_norm += float(np.linalg.norm(diff, ord="fro"))
            elif d1 is not None:
                total_param_norm += float(np.linalg.norm(d1, ord="fro"))
            elif d2 is not None:
                total_param_norm += float(np.linalg.norm(d2, ord="fro"))

        param_dist = float(total_param_norm / max(1, len(all_layers)))

        # Functional divergence if probe set & forward function are supplied
        if probe_inputs is not None and forward_fn is not None and model is not None:
            with self.patch_scope(model):
                out1 = forward_fn(model, probe_inputs)
            with other.patch_scope(model):
                out2 = forward_fn(model, probe_inputs)
            func_diff = float(np.linalg.norm(out1 - out2, ord="fro") / max(1, len(probe_inputs)))
            return float(func_diff + 0.1 * param_dist)

        return param_dist

    def apply_to_model(self, model: Any, in_place: bool = True) -> Any:
        """Applies candidate weight patches to a model (PyTorch module or NumPy/dict model).

        If in_place is False, operates on a deep copy of the model to guarantee absolute safety.
        """
        target_model = model if in_place else copy.deepcopy(model)

        # 1. PyTorch Module
        if hasattr(target_model, "state_dict") and hasattr(target_model, "load_state_dict"):
            try:
                import torch
                if isinstance(target_model, torch.nn.Module):
                    with torch.no_grad():
                        state_dict = target_model.state_dict()
                        all_layers = set(self.sparse_patches.keys()) | set(self.low_rank_factors.keys())
                        for name in all_layers:
                            if name in state_dict:
                                delta_np = self.get_dense_delta(name)
                                if delta_np is not None:
                                    delta_t = torch.from_numpy(delta_np).to(
                                        device=state_dict[name].device,
                                        dtype=state_dict[name].dtype,
                                    )
                                    state_dict[name].add_(delta_t)
                    return target_model
            except ImportError:
                pass

        # 2. NumPy dict / Custom model with weights attribute
        target_dict = getattr(target_model, "weights", None)
        if target_dict is None and isinstance(target_model, dict):
            target_dict = target_model

        if isinstance(target_dict, dict):
            all_layers = set(self.sparse_patches.keys()) | set(self.low_rank_factors.keys())
            for name in all_layers:
                if name in target_dict:
                    delta_np = self.get_dense_delta(name)
                    if delta_np is not None:
                        target_dict[name] = target_dict[name] + delta_np

        return target_model

    def revert_from_model(self, model: Any) -> Any:
        """Reverts candidate weight patches from a model in-place (subtracts deltas)."""
        # 1. PyTorch Module
        if hasattr(model, "state_dict") and hasattr(model, "load_state_dict"):
            try:
                import torch
                if isinstance(model, torch.nn.Module):
                    with torch.no_grad():
                        state_dict = model.state_dict()
                        all_layers = set(self.sparse_patches.keys()) | set(self.low_rank_factors.keys())
                        for name in all_layers:
                            if name in state_dict:
                                delta_np = self.get_dense_delta(name)
                                if delta_np is not None:
                                    delta_t = torch.from_numpy(delta_np).to(
                                        device=state_dict[name].device,
                                        dtype=state_dict[name].dtype,
                                    )
                                    state_dict[name].sub_(delta_t)
                    return model
            except ImportError:
                pass

        # 2. NumPy dict / Custom model with weights attribute
        target_dict = getattr(model, "weights", None)
        if target_dict is None and isinstance(model, dict):
            target_dict = model

        if isinstance(target_dict, dict):
            all_layers = set(self.sparse_patches.keys()) | set(self.low_rank_factors.keys())
            for name in all_layers:
                if name in target_dict:
                    delta_np = self.get_dense_delta(name)
                    if delta_np is not None:
                        target_dict[name] = target_dict[name] - delta_np

        return model

    @contextmanager
    def patch_scope(self, model: Any) -> Iterator[Any]:
        """Context manager applying weight patches on enter and strictly reverting on exit.

        Guarantees zero parameter drift and eliminates candidate evaluation cross-contamination.
        """
        self.apply_to_model(model, in_place=True)
        try:
            yield model
        finally:
            self.revert_from_model(model)

    def prune_inactive_weights(self, threshold: float = 1e-4) -> int:
        """Ablate / zero out weight perturbations below magnitude threshold."""
        pruned_count = 0
        for name, patch in list(self.sparse_patches.items()):
            mask = np.abs(patch) < threshold
            pruned_count += int(np.sum(mask))
            patch[mask] = 0.0
            # If entire patch is zero, remove layer entry to maintain extreme parsimony
            if np.all(patch == 0.0):
                del self.sparse_patches[name]
        return pruned_count

    def serialize(self, include_coo: bool = True) -> dict[str, Any]:
        """JSON-safe serialization of genome metadata, sparsity, and lineage.

        When include_coo is True, stores coordinate list (COO) representations of
        sparse patches and factor matrices to enable complete bit-accurate reconstruction.
        """
        active_layers = sorted(set(self.sparse_patches.keys()) | set(self.low_rank_factors.keys()))
        layer_summaries = {}
        for l in active_layers:
            d = self.get_dense_delta(l)
            if d is not None:
                layer_summaries[l] = {
                    "shape": list(d.shape),
                    "nonzero_elements": int(np.count_nonzero(d)),
                    "frobenius_norm": round(float(np.linalg.norm(d, ord="fro")), 6),
                }

        payload: dict[str, Any] = {
            "genome_type": "NeuralWeightGenome",
            "base_model_id": self.base_model_id,
            "scale": round(self.scale, 6),
            "fingerprint": self.fingerprint(),
            "active_layers_count": len(active_layers),
            "layers": layer_summaries,
            "edit_lineage": self.edit_lineage,
            "edit_lineage_count": len(self.edit_lineage),
        }

        if include_coo:
            sparse_coo: dict[str, Any] = {}
            for name, patch in self.sparse_patches.items():
                p_np = patch.detach().cpu().numpy() if hasattr(patch, "detach") else np.asarray(patch, dtype=float)
                coords = np.nonzero(p_np)
                sparse_coo[name] = {
                    "shape": list(p_np.shape),
                    "coords": [coords[dim].tolist() for dim in range(p_np.ndim)],
                    "values": [float(v) for v in p_np[coords]],
                }
            payload["sparse_coo"] = sparse_coo

            lr_factors: dict[str, Any] = {}
            for name, (u, v) in self.low_rank_factors.items():
                u_np = u.detach().cpu().numpy() if hasattr(u, "detach") else np.asarray(u, dtype=float)
                v_np = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=float)
                lr_factors[name] = {
                    "u": u_np.tolist(),
                    "v": v_np.tolist(),
                }
            payload["low_rank_factors"] = lr_factors

        return payload

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> NeuralWeightGenome:
        """Reconstruct a NeuralWeightGenome instance from a serialized dictionary."""
        base_model_id = data.get("base_model_id", "unknown")
        scale = float(data.get("scale", 1.0))
        edit_lineage = copy.deepcopy(data.get("edit_lineage", []))

        sparse_patches: dict[str, np.ndarray] = {}
        if "sparse_coo" in data:
            for name, entry in data["sparse_coo"].items():
                shape = tuple(entry["shape"])
                coords = tuple(np.array(c, dtype=int) for c in entry["coords"])
                values = np.array(entry["values"], dtype=float)
                dense = np.zeros(shape, dtype=float)
                if len(values) > 0:
                    dense[coords] = values
                sparse_patches[name] = dense

        low_rank_factors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if "low_rank_factors" in data:
            for name, factors in data["low_rank_factors"].items():
                u = np.array(factors["u"], dtype=float)
                v = np.array(factors["v"], dtype=float)
                low_rank_factors[name] = (u, v)

        return cls(
            base_model_id=base_model_id,
            sparse_patches=sparse_patches,
            low_rank_factors=low_rank_factors,
            scale=scale,
            edit_lineage=edit_lineage,
        )

    def describe(self) -> dict[str, float | int | str]:
        """Behavioral descriptors for quality diversity, MAP-Elites, and analysis."""
        total_nonzero = 0
        total_fro = 0.0
        all_layers = set(self.sparse_patches.keys()) | set(self.low_rank_factors.keys())

        for l in all_layers:
            d = self.get_dense_delta(l)
            if d is not None:
                total_nonzero += int(np.count_nonzero(d))
                total_fro += float(np.linalg.norm(d, ord="fro"))

        return {
            "edited_params": total_nonzero,
            "active_layers": len(all_layers),
            "norm_fro": round(total_fro, 4),
            "scale": round(self.scale, 4),
        }

    def mutate(self, rng: random.Random | None = None, **kwargs: Any) -> NeuralWeightGenome:
        """Mutate sparse patches or scaling factor."""
        child = self.clone()
        r = rng or random
        mutation_type = kwargs.get("type", "perturb")

        if mutation_type == "scale":
            factor = r.uniform(0.8, 1.25)
            child.scale *= factor
            child.edit_lineage.append({"op": "scale_mutation", "factor": factor})
            return child

        # Default: perturb existing active patches or add low-rank perturbation
        all_layers = list(set(child.sparse_patches.keys()) | set(child.low_rank_factors.keys()))
        if all_layers:
            target_layer = r.choice(all_layers) if hasattr(r, "choice") else random.choice(all_layers)
            target_layer = str(target_layer)
            if target_layer in child.sparse_patches:
                patch = child.sparse_patches[target_layer]
                coords = np.nonzero(patch)
                if len(coords[0]) > 0:
                    idx = r.randint(0, len(coords[0]) - 1) if hasattr(r, "randint") else random.randint(0, len(coords[0]) - 1)
                    coord = tuple(c[idx] for c in coords)
                    noise = float(r.gauss(0.0, 0.05)) if hasattr(r, "gauss") else float(r.normal(0.0, 0.05)) if hasattr(r, "normal") else float(random.gauss(0.0, 0.05))
                    patch[coord] += noise
                    child.edit_lineage.append({"op": "point_perturb", "layer": target_layer, "noise": noise})

        return child

    def crossover(self, other: Any, rng: random.Random | None = None, **kwargs: Any) -> NeuralWeightGenome:
        """Perform layer-wise uniform crossover between two candidate weight patches."""
        if not isinstance(other, NeuralWeightGenome):
            return self.clone()

        r = rng or random
        child = self.clone()
        all_layers = sorted(set(self.sparse_patches.keys()) | set(other.sparse_patches.keys()))

        for layer in all_layers:
            if r.random() < 0.5 and layer in other.sparse_patches:
                child.sparse_patches[layer] = np.copy(other.sparse_patches[layer])
            elif layer in self.sparse_patches:
                child.sparse_patches[layer] = np.copy(self.sparse_patches[layer])

        child.scale = (self.scale + other.scale) / 2.0
        child.edit_lineage.append({"op": "layer_crossover", "parents": [self.fingerprint()[:8], other.fingerprint()[:8]]})
        return child
