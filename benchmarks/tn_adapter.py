from __future__ import annotations

import math
import random
from typing import Any, cast

import numpy as np
import torch

from benchmarks.instance_generators import normalize_for_tn
from benchmarks.metrics import has_finite_edges, is_valid_tsp_route, route_cost
from tsp_module import tn_tsp_solver


def _to_numpy_distances(distances: np.ndarray | torch.Tensor | Any) -> np.ndarray:
    detach = getattr(distances, "detach", None)
    if callable(detach):
        tensor_like = cast(Any, detach())
        return cast(np.ndarray, tensor_like.cpu().numpy()).astype(float)
    return np.asarray(distances, dtype=float)


def _route_to_list(route: Any) -> list[int]:
    detach = getattr(route, "detach", None)
    if callable(detach):
        tensor_like = cast(Any, detach())
        return [
            int(node)
            for node in cast(np.ndarray, tensor_like.cpu().numpy()).reshape(-1).tolist()
        ]
    return [int(node) for node in np.asarray(route).reshape(-1).tolist()]


def _set_all_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def solve_tsp_tn(
    distances: np.ndarray | torch.Tensor | Any,
    tau: float = 1.0,
    n_layers: int | None = None,
    seed: int | None = None,
    normalize: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the repository TN solver and return a benchmark-friendly result."""
    original_distances = _to_numpy_distances(distances)
    solver_name = "tn_exact" if n_layers is None else f"tn_n_layers_{n_layers}"
    normalization_factor = 1.0
    solver_distances = original_distances
    if normalize:
        solver_distances, normalization_factor = normalize_for_tn(
            original_distances, method="max"
        )

    metadata: dict[str, Any] = {
        "tau": tau,
        "n_layers": n_layers,
        "normalize": normalize,
        "normalization_factor": normalization_factor,
        "seed": seed,
    }
    _set_all_seeds(seed)

    try:
        tensor_distances = torch.tensor(solver_distances, dtype=torch.float32)
        raw_route = tn_tsp_solver(
            tensor_distances,
            tau=tau,
            verbose=verbose,
            n_layers=n_layers,
        )
        route = _route_to_list(raw_route)
    except Exception as exc:
        metadata["error"] = str(exc)
        return {
            "solver": solver_name,
            "route": None,
            "cost": None,
            "status": "error",
            "metadata": metadata,
        }

    valid_route = is_valid_tsp_route(route, original_distances.shape[0])
    finite_edges = has_finite_edges(route, original_distances) if valid_route else False
    cost = route_cost(route, original_distances) if valid_route else None
    metadata.update({"valid_route": valid_route, "finite_edges": finite_edges})

    if not valid_route:
        metadata["numeric_warning"] = (
            "Invalid route. This can happen with too few restriction layers or "
            "numerical underflow/overflow from tau."
        )
        status = "invalid_route"
    elif not finite_edges:
        metadata["numeric_warning"] = (
            "Route uses at least one forbidden or infinite edge."
        )
        status = "invalid_edges"
    elif cost is None or not math.isfinite(float(cost)):
        metadata["numeric_warning"] = "Route cost is not finite."
        status = "non_finite_cost"
    else:
        status = "ok"

    return {
        "solver": solver_name,
        "route": route,
        "cost": float(cost)
        if cost is not None and math.isfinite(float(cost))
        else cost,
        "status": status,
        "metadata": metadata,
    }
