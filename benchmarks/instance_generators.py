from __future__ import annotations

from typing import Any

import numpy as np


def generate_euclidean_tsp(
    n: int,
    seed: int,
    scale: float = 1000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a symmetric Euclidean TSP instance from random 2D points."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, scale, size=(n, 2))
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, 0.0)
    metadata = {
        "instance_type": "euclidean",
        "n": n,
        "seed": seed,
        "scale": scale,
        "points": points.tolist(),
        "symmetric": True,
        "complete": True,
    }
    return distances.astype(float), metadata


def generate_random_complete_tsp(
    n: int,
    seed: int,
    low: float = 1.0,
    high: float = 100.0,
    symmetric: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a complete TSP instance with random finite edge costs."""
    rng = np.random.default_rng(seed)
    distances = rng.uniform(low, high, size=(n, n))
    if symmetric:
        distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)
    metadata = {
        "instance_type": "random_symmetric" if symmetric else "random_asymmetric",
        "n": n,
        "seed": seed,
        "low": low,
        "high": high,
        "symmetric": symmetric,
        "complete": True,
    }
    return distances.astype(float), metadata


def generate_sparse_hamiltonian_tsp(
    n: int,
    seed: int,
    edge_probability: float,
    low: float = 1.0,
    high: float = 100.0,
    symmetric: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a TSP matrix with forbidden edges and a guaranteed finite tour."""
    rng = np.random.default_rng(seed)
    distances = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    cycle = rng.permutation(n).tolist()

    for index, origin in enumerate(cycle):
        destination = cycle[(index + 1) % n]
        cost = float(rng.uniform(low, high))
        distances[origin, destination] = cost
        if symmetric:
            distances[destination, origin] = cost

    if symmetric:
        for origin in range(n):
            for destination in range(origin + 1, n):
                if np.isfinite(distances[origin, destination]):
                    continue
                if rng.random() < edge_probability:
                    cost = float(rng.uniform(low, high))
                    distances[origin, destination] = cost
                    distances[destination, origin] = cost
    else:
        for origin in range(n):
            for destination in range(n):
                if origin == destination or np.isfinite(distances[origin, destination]):
                    continue
                if rng.random() < edge_probability:
                    distances[origin, destination] = float(rng.uniform(low, high))

    metadata = {
        "instance_type": "sparse_hamiltonian_symmetric"
        if symmetric
        else "sparse_hamiltonian_asymmetric",
        "n": n,
        "seed": seed,
        "edge_probability": edge_probability,
        "low": low,
        "high": high,
        "symmetric": symmetric,
        "complete": False,
        "guaranteed_cycle": cycle,
    }
    return distances, metadata


def normalize_for_tn(
    distances: np.ndarray,
    method: str | None = "max",
) -> tuple[np.ndarray, float]:
    """Normalize finite costs for the exponential TN weights and preserve infinities."""
    matrix = np.asarray(distances, dtype=float).copy()
    finite_mask = np.isfinite(matrix)
    finite_values = matrix[finite_mask]
    if method is None or method == "none" or finite_values.size == 0:
        return matrix, 1.0

    if method == "max":
        factor = float(np.max(np.abs(finite_values)))
    elif method == "mean":
        factor = float(np.mean(np.abs(finite_values)))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if factor <= 0.0 or not np.isfinite(factor):
        factor = 1.0
    matrix[finite_mask] = matrix[finite_mask] / factor
    return matrix, factor
