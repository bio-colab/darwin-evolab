"""src/evolab/neural/profiler.py — General Activation Profiler & Subspace Inspector.

Empirical System-1 diagnostic engine that inspects representations across arbitrary
neural architectures (both NumPy dictionaries and PyTorch nn.Modules).
Computes activation discrepancy, representation drift, and anchor null-space bases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import svd


@dataclass
class LayerActivationStats:
    """Activation statistics for an individual neural network layer."""
    layer_name: str
    fail_mean: np.ndarray
    fail_std: np.ndarray
    anchor_mean: Optional[np.ndarray] = None
    anchor_std: Optional[np.ndarray] = None
    discrepancy: Optional[np.ndarray] = None
    contrast_ratio: Optional[np.ndarray] = None
    active_indices: List[int] = field(default_factory=list)
    anchor_activations: Optional[np.ndarray] = None


@dataclass
class ActivationProfile:
    """Structured telemetry output from the ActivationProfiler."""
    layers: Dict[str, LayerActivationStats]
    top_discrepancy_layer: str
    active_indices: Dict[str, List[int]]
    summary: str
    anchor_activations: Dict[str, np.ndarray] = field(default_factory=dict)


class ActivationProfiler:
    """Architecture-agnostic activation telemetry inspector for neural models."""

    def __init__(self, discrepancy_threshold: float = 1.0, top_k_units: int = 4) -> None:
        self.discrepancy_threshold = float(discrepancy_threshold)
        self.top_k_units = int(top_k_units)

    def forward_trace_numpy(
        self,
        weights: Dict[str, np.ndarray],
        inputs: np.ndarray,
        activation_fn: Callable[[np.ndarray], np.ndarray] = lambda x: np.maximum(0, x),
    ) -> Dict[str, np.ndarray]:
        """Compute layer-wise activations for a NumPy feedforward network.

        Identifies sequential layers by prefix (e.g. 'layer1.weight', 'layer2.weight').
        """
        activations: Dict[str, np.ndarray] = {"inputs": np.asarray(inputs, dtype=float)}
        current_x = np.asarray(inputs, dtype=float)

        # Detect linear layers sorted by name
        weight_keys = sorted([k for k in weights.keys() if k.endswith(".weight") or (k.startswith("layer") and not k.endswith(".bias"))])

        for idx, w_key in enumerate(weight_keys):
            w = weights[w_key]
            # Match corresponding bias
            b_key = w_key.replace(".weight", ".bias") if ".weight" in w_key else f"{w_key}.bias"
            b = weights.get(b_key, np.zeros(w.shape[0]))

            # Linear transformation
            z = current_x @ w.T + b

            # Store pre/post activation
            layer_id = w_key.replace(".weight", "")
            # Apply activation for all except potentially the final output layer
            if idx < len(weight_keys) - 1:
                current_x = activation_fn(z)
            else:
                current_x = z
            activations[layer_id] = current_x

        return activations

    def forward_trace_torch(
        self,
        module: Any,
        inputs: Any,
    ) -> Dict[str, np.ndarray]:
        """Compute layer-wise activations for a PyTorch nn.Module using forward hooks."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is not installed in the environment.")

        activations: Dict[str, np.ndarray] = {}
        hooks = []

        def get_hook(name: str):
            def hook(mod, inp, out):
                val = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
                activations[name] = val
            return hook

        for name, submod in module.named_modules():
            if len(list(submod.children())) == 0 and name != "":
                hooks.append(submod.register_forward_hook(get_hook(name)))

        try:
            if not isinstance(inputs, torch.Tensor):
                inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            else:
                inputs_tensor = inputs

            with torch.no_grad():
                module(inputs_tensor)
        finally:
            for h in hooks:
                h.remove()

        return activations

    def profile(
        self,
        model: Union[Dict[str, np.ndarray], Any],
        failing_inputs: np.ndarray,
        anchor_inputs: Optional[np.ndarray] = None,
    ) -> ActivationProfile:
        """Inspect activations on failing and anchor inputs to diagnose defect representations."""
        f_in = np.asarray(failing_inputs, dtype=float)
        a_in = np.asarray(anchor_inputs, dtype=float) if anchor_inputs is not None else None

        # Execute forward traces
        if isinstance(model, dict):
            fail_acts = self.forward_trace_numpy(model, f_in)
            anchor_acts = self.forward_trace_numpy(model, a_in) if a_in is not None else None
        else:
            fail_acts = self.forward_trace_torch(model, f_in)
            anchor_acts = self.forward_trace_torch(model, a_in) if a_in is not None else None

        layer_stats: Dict[str, LayerActivationStats] = {}
        active_indices_map: Dict[str, List[int]] = {}
        max_disc_score = -1.0
        top_layer = ""

        # Analyze representations across layers (excluding raw input)
        analyzable_layers = [k for k in fail_acts.keys() if k != "inputs"]

        for layer in analyzable_layers:
            f_act = fail_acts[layer]
            f_mean = np.mean(f_act, axis=0)
            f_std = np.std(f_act, axis=0)

            a_mean, a_std, disc, contrast = None, None, None, None
            anchor_act_mat = None

            if anchor_acts is not None and layer in anchor_acts:
                a_act = anchor_acts[layer]
                anchor_act_mat = a_act
                a_mean = np.mean(a_act, axis=0)
                a_std = np.std(a_act, axis=0)
                disc = f_mean - a_mean
                contrast = np.abs(disc) / (f_std + a_std + 1e-6)
                score = float(np.max(np.abs(disc)))
            else:
                score = float(np.max(np.abs(f_mean)))

            # Identify active unit indices
            if disc is not None:
                pos_idx = np.where(disc > self.discrepancy_threshold)[0]
                if len(pos_idx) > 0:
                    chosen_idx = pos_idx.tolist()
                else:
                    chosen_idx = np.argsort(disc)[-self.top_k_units:].tolist()
            else:
                chosen_idx = np.argsort(f_mean)[-self.top_k_units:].tolist()

            layer_stats[layer] = LayerActivationStats(
                layer_name=layer,
                fail_mean=f_mean,
                fail_std=f_std,
                anchor_mean=a_mean,
                anchor_std=a_std,
                discrepancy=disc,
                contrast_ratio=contrast,
                active_indices=chosen_idx,
                anchor_activations=anchor_act_mat,
            )
            active_indices_map[layer] = chosen_idx

            if score > max_disc_score:
                max_disc_score = score
                top_layer = layer

        if not top_layer and analyzable_layers:
            top_layer = analyzable_layers[-1]

        summary = (
            f"Profiled {len(analyzable_layers)} layers. "
            f"Identified primary defect subspace in '{top_layer}' with active units {active_indices_map.get(top_layer, [])}."
        )

        anchor_map = {k: v.anchor_activations for k, v in layer_stats.items() if v.anchor_activations is not None}

        return ActivationProfile(
            layers=layer_stats,
            top_discrepancy_layer=top_layer,
            active_indices=active_indices_map,
            summary=summary,
            anchor_activations=anchor_map,
        )

    def compute_null_space_basis(
        self,
        activations: np.ndarray,
        tol: float = 1e-4,
        center: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the orthogonal null-space projection matrix for anchor activations.

        Given anchor activations X in R^(N x d_in), computes P_null in R^(d_in x d_in)
        such that X @ P_null ≈ 0 (or zero-variance if center=True).
        Returns:
            P_null: (d_in, d_in) projection matrix.
            singular_values: (min(N, d_in),) singular values of X.
        """
        X = np.asarray(activations, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        N, d_in = X.shape
        X_proc = (X - np.mean(X, axis=0, keepdims=True)) if center and N > 1 else X

        # SVD: X = U Sigma V^T
        U, s, Vt = svd(X_proc, full_matrices=True)
        # Numerical rank
        cutoff = tol * (s[0] if len(s) > 0 else 1.0)
        rank = int(np.sum(s > cutoff))

        # Vt has shape (d_in, d_in). Rows r to d_in-1 are the null space vectors!
        # The null space basis V_null = Vt[rank:, :].T  (shape d_in x (d_in - rank))
        if rank < d_in:
            V_null = Vt[rank:, :].T
            P_null = V_null @ V_null.T
        else:
            # When full rank, construct least-interfering regularized subspace
            # using bottom singular vectors
            k = max(1, d_in // 4)
            V_least = Vt[-k:, :].T
            P_null = V_least @ V_least.T

        return P_null, s
