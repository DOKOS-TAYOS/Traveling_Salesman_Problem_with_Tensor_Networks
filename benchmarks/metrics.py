from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import numpy as np

ResultT = TypeVar("ResultT")


def _to_numpy_array(value: Any) -> np.ndarray:
    """Convert numpy-like and torch-like values to a NumPy array."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        tensor_like = cast(Any, detach())
        return cast(np.ndarray, tensor_like.cpu().numpy())
    return np.asarray(value)


def _route_to_list(route: Sequence[int] | np.ndarray | Any) -> list[int]:
    array = _to_numpy_array(route).reshape(-1)
    return [int(node) for node in array.tolist()]


def _edge_pairs(route: list[int], cycle: bool) -> list[tuple[int, int]]:
    if len(route) < 2:
        return []
    if cycle and route[0] == route[-1]:
        return list(zip(route[:-1], route[1:], strict=False))
    pairs = list(zip(route[:-1], route[1:], strict=False))
    if cycle:
        pairs.append((route[-1], route[0]))
    return pairs


def route_cost(
    route: Sequence[int] | np.ndarray | Any,
    distances: np.ndarray | Any,
    cycle: bool = True,
) -> float:
    """Return the cost of a TSP route, closing the cycle by default."""
    matrix = _to_numpy_array(distances)
    route_list = _route_to_list(route)
    n = matrix.shape[0]
    total = 0.0
    for origin, destination in _edge_pairs(route_list, cycle=cycle):
        if origin < 0 or destination < 0 or origin >= n or destination >= n:
            return float("inf")
        edge_cost = float(matrix[origin, destination])
        if not math.isfinite(edge_cost):
            return float("inf")
        total += edge_cost
    return float(total)


def is_valid_tsp_route(
    route: Sequence[int] | np.ndarray | Any,
    n: int,
    cycle: bool = True,
    allow_start_repeat: bool = True,
) -> bool:
    """Check that a route visits every node exactly once."""
    del cycle
    route_list = _route_to_list(route)
    if (
        allow_start_repeat
        and len(route_list) == n + 1
        and route_list[0] == route_list[-1]
    ):
        route_list = route_list[:-1]
    if len(route_list) != n:
        return False
    if any(node < 0 or node >= n for node in route_list):
        return False
    return len(set(route_list)) == n


def has_finite_edges(
    route: Sequence[int] | np.ndarray | Any,
    distances: np.ndarray | Any,
    cycle: bool = True,
) -> bool:
    """Return whether every edge used by the route has finite cost."""
    matrix = _to_numpy_array(distances)
    route_list = _route_to_list(route)
    n = matrix.shape[0]
    for origin, destination in _edge_pairs(route_list, cycle=cycle):
        if origin < 0 or destination < 0 or origin >= n or destination >= n:
            return False
        if not math.isfinite(float(matrix[origin, destination])):
            return False
    return True


def relative_gap(cost: float | None, optimum: float | None) -> float:
    """Return the relative gap to an optimum, or NaN for invalid inputs."""
    if cost is None or optimum is None:
        return float(np.nan)
    cost_float = float(cost)
    optimum_float = float(optimum)
    if not math.isfinite(cost_float) or not math.isfinite(optimum_float):
        return float(np.nan)
    return float((cost_float - optimum_float) / max(abs(optimum_float), 1e-12))


def canonicalize_cycle(
    route: Sequence[int] | np.ndarray | Any,
    start: int | None = None,
    symmetric: bool = False,
) -> list[int]:
    """Remove duplicate closure and rotate a cycle to a stable representation."""
    route_list = _route_to_list(route)
    if len(route_list) > 1 and route_list[0] == route_list[-1]:
        route_list = route_list[:-1]
    if not route_list:
        return []

    def rotate(values: list[int], first_node: int | None) -> list[int]:
        if first_node is None or first_node not in values:
            return values.copy()
        index = values.index(first_node)
        return values[index:] + values[:index]

    forward = rotate(route_list, start)
    if not symmetric:
        return forward
    backward = rotate(list(reversed(route_list)), start)
    return min(forward, backward)


def measure_runtime_and_memory(
    fn: Callable[..., ResultT],
    *args: Any,
    **kwargs: Any,
) -> tuple[ResultT, float, float, float, float]:
    """Run a callable and return result, wall-clock seconds, and RSS memory."""
    try:
        import psutil
    except ImportError:
        process = None
    else:
        process = psutil.Process()

    rss_before_mb = (
        process.memory_info().rss / (1024 * 1024)
        if process is not None
        else float(np.nan)
    )
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    runtime_s = time.perf_counter() - start
    rss_after_mb = (
        process.memory_info().rss / (1024 * 1024)
        if process is not None
        else float(np.nan)
    )
    rss_delta_mb = rss_after_mb - rss_before_mb
    return result, runtime_s, rss_before_mb, rss_after_mb, rss_delta_mb
