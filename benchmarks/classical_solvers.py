from __future__ import annotations

import importlib.util
import itertools
import math
from typing import Any, cast

import numpy as np

from benchmarks.metrics import (
    canonicalize_cycle,
    has_finite_edges,
    is_valid_tsp_route,
    route_cost,
)

SolverResult = dict[str, Any]


def _to_numpy_distances(distances: np.ndarray | Any) -> np.ndarray:
    detach = getattr(distances, "detach", None)
    if callable(detach):
        tensor_like = cast(Any, detach())
        return cast(np.ndarray, tensor_like.cpu().numpy()).astype(float)
    return np.asarray(distances, dtype=float)


def _default_start(n: int, start: int | None) -> int:
    return n - 1 if start is None else int(start)


def _result(
    solver: str,
    route: list[int] | None,
    cost: float | None,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> SolverResult:
    return {
        "solver": solver,
        "route": route,
        "cost": cost,
        "status": status,
        "metadata": metadata or {},
    }


def solve_tsp_bruteforce(
    distances: np.ndarray | Any,
    start: int | None = None,
    max_n: int = 10,
) -> SolverResult:
    """Solve small TSP instances exactly by enumerating all tours."""
    matrix = _to_numpy_distances(distances)
    n = matrix.shape[0]
    start_node = _default_start(n, start)
    metadata = {"start": start_node, "max_n": max_n}
    if n > max_n:
        return _result("bruteforce", None, None, "skipped", metadata)
    if n == 0:
        return _result("bruteforce", None, None, "no_solution", metadata)
    if n == 1:
        return _result("bruteforce", [start_node], 0.0, "ok", metadata)

    other_nodes = [node for node in range(n) if node != start_node]
    best_route: list[int] | None = None
    best_cost = float("inf")
    checked = 0
    for permutation in itertools.permutations(other_nodes):
        route = [start_node, *permutation]
        cost = route_cost(route, matrix)
        checked += 1
        if cost < best_cost:
            best_cost = cost
            best_route = route

    metadata["permutations_checked"] = checked
    if best_route is None or not math.isfinite(best_cost):
        return _result("bruteforce", None, None, "no_solution", metadata)
    return _result("bruteforce", best_route, float(best_cost), "ok", metadata)


def solve_tsp_held_karp(
    distances: np.ndarray | Any,
    start: int | None = None,
    max_n: int = 18,
) -> SolverResult:
    """Solve TSP exactly with the classical Held-Karp dynamic program."""
    matrix = _to_numpy_distances(distances)
    n = matrix.shape[0]
    start_node = _default_start(n, start)
    metadata = {"start": start_node, "max_n": max_n}
    if n > max_n:
        return _result("held_karp", None, None, "skipped", metadata)
    if n == 0:
        return _result("held_karp", None, None, "no_solution", metadata)
    if n == 1:
        return _result("held_karp", [start_node], 0.0, "ok", metadata)

    nodes = [node for node in range(n) if node != start_node]
    bit_for_node = {node: index for index, node in enumerate(nodes)}
    all_mask = (1 << len(nodes)) - 1
    dp: dict[tuple[int, int], tuple[float, int | None]] = {}

    for node in nodes:
        edge_cost = float(matrix[start_node, node])
        if math.isfinite(edge_cost):
            dp[(1 << bit_for_node[node], node)] = (edge_cost, None)

    for mask in range(1, all_mask + 1):
        for node in nodes:
            node_bit = 1 << bit_for_node[node]
            if not mask & node_bit or mask == node_bit:
                continue
            previous_mask = mask ^ node_bit
            best_cost = float("inf")
            best_previous: int | None = None
            for previous in nodes:
                previous_bit = 1 << bit_for_node[previous]
                if not previous_mask & previous_bit:
                    continue
                previous_state = dp.get((previous_mask, previous))
                edge_cost = float(matrix[previous, node])
                if previous_state is None or not math.isfinite(edge_cost):
                    continue
                candidate = previous_state[0] + edge_cost
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = previous
            if best_previous is not None:
                dp[(mask, node)] = (best_cost, best_previous)

    best_total = float("inf")
    best_end: int | None = None
    for node in nodes:
        state = dp.get((all_mask, node))
        return_cost = float(matrix[node, start_node])
        if state is None or not math.isfinite(return_cost):
            continue
        candidate = state[0] + return_cost
        if candidate < best_total:
            best_total = candidate
            best_end = node

    if best_end is None or not math.isfinite(best_total):
        return _result("held_karp", None, None, "no_solution", metadata)

    reverse_route: list[int] = []
    current = best_end
    mask = all_mask
    while current is not None:
        reverse_route.append(current)
        state = dp[(mask, current)]
        mask ^= 1 << bit_for_node[current]
        current = state[1]
    route = [start_node, *reversed(reverse_route)]
    return _result("held_karp", route, float(best_total), "ok", metadata)


def _integerize_distances(matrix: np.ndarray) -> tuple[list[list[int]], int, int]:
    finite_values = matrix[np.isfinite(matrix)]
    finite_values = finite_values[finite_values >= 0.0]
    max_finite = float(np.max(finite_values)) if finite_values.size else 1.0
    scale = 1_000_000 if max_finite <= 1_000_000 else max(1, int(10**12 / max_finite))
    converted = np.zeros(matrix.shape, dtype=object)
    finite_mask = np.isfinite(matrix)
    converted[finite_mask] = np.rint(matrix[finite_mask] * scale).astype(object)
    positive_finite = finite_mask & (matrix > 0.0)
    converted[positive_finite] = np.maximum(converted[positive_finite], 1)
    max_integer = max(
        int(np.max(converted[finite_mask])) if finite_values.size else 1, 1
    )
    penalty = min(max(max_integer * matrix.shape[0] * 1000, 10**9), 2**60)
    converted[~finite_mask] = penalty
    np.fill_diagonal(converted, 0)
    return converted.astype(int).tolist(), int(scale), int(penalty)


def solve_tsp_ortools(
    distances: np.ndarray | Any,
    start: int | None = None,
    time_limit_s: int | float = 5,
) -> SolverResult:
    """Solve TSP with OR-Tools Routing when the optional dependency is installed."""
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # pyright: ignore[reportMissingImports]
    except ImportError:
        return _result(
            "ortools",
            None,
            None,
            "skipped_missing_dependency",
            {"time_limit_s": time_limit_s},
        )

    matrix = _to_numpy_distances(distances)
    n = matrix.shape[0]
    start_node = _default_start(n, start)
    integer_matrix, scale, penalty = _integerize_distances(matrix)
    metadata = {
        "start": start_node,
        "time_limit_s": time_limit_s,
        "integer_scale": scale,
        "infinity_penalty": penalty,
    }

    manager = pywrapcp.RoutingIndexManager(n, 1, start_node)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return integer_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    whole_seconds = max(int(time_limit_s), 0)
    fractional_seconds = max(float(time_limit_s) - whole_seconds, 0.0)
    search_parameters.time_limit.seconds = whole_seconds
    search_parameters.time_limit.nanos = int(fractional_seconds * 1_000_000_000)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return _result("ortools", None, None, "no_solution", metadata)

    route: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    valid_route = is_valid_tsp_route(route, n)
    finite_edges = has_finite_edges(route, matrix) if valid_route else False
    cost = route_cost(route, matrix) if valid_route else None
    metadata["valid_route"] = valid_route
    metadata["finite_edges"] = finite_edges

    if not valid_route:
        return _result("ortools", route, cost, "invalid_route", metadata)
    if not finite_edges:
        return _result("ortools", route, cost, "invalid_edges", metadata)
    assert cost is not None
    return _result("ortools", route, float(cost), "ok", metadata)


def _is_complete_finite(matrix: np.ndarray) -> bool:
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    return bool(np.all(np.isfinite(matrix[off_diagonal])))


def _is_symmetric(matrix: np.ndarray, tolerance: float = 1e-9) -> bool:
    return bool(
        np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0, equal_nan=False)
    )


def _is_metric(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if matrix[i, k] > matrix[i, j] + matrix[j, k] + tolerance:
                    return False
    return True


def _networkx_missing_result(solver: str) -> SolverResult:
    return _result(solver, None, None, "skipped_missing_dependency")


def _build_networkx_graph(matrix: np.ndarray, directed: bool) -> Any:
    import networkx as nx

    graph = nx.DiGraph() if directed else nx.Graph()
    n = matrix.shape[0]
    graph.add_nodes_from(range(n))
    for origin in range(n):
        for destination in range(n):
            if origin == destination:
                continue
            if not directed and destination < origin:
                continue
            graph.add_edge(
                origin, destination, weight=float(matrix[origin, destination])
            )
    return graph


def _networkx_route_result(
    solver: str,
    route: list[int],
    matrix: np.ndarray,
    symmetric: bool,
    metadata: dict[str, Any] | None = None,
) -> SolverResult:
    canonical_route = canonicalize_cycle(
        route, start=matrix.shape[0] - 1, symmetric=symmetric
    )
    valid_route = is_valid_tsp_route(canonical_route, matrix.shape[0])
    finite_edges = has_finite_edges(canonical_route, matrix) if valid_route else False
    cost = route_cost(canonical_route, matrix) if valid_route else None
    result_metadata = metadata or {}
    result_metadata.update({"valid_route": valid_route, "finite_edges": finite_edges})
    if not valid_route:
        return _result(solver, canonical_route, cost, "invalid_route", result_metadata)
    if not finite_edges:
        return _result(solver, canonical_route, cost, "invalid_edges", result_metadata)
    assert cost is not None
    return _result(solver, canonical_route, float(cost), "ok", result_metadata)


def solve_tsp_networkx_christofides(distances: np.ndarray | Any) -> SolverResult:
    """Run NetworkX Christofides when the instance is symmetric, finite, and metric."""
    if importlib.util.find_spec("networkx") is None:
        return _networkx_missing_result("networkx_christofides")
    try:
        import networkx as nx
    except ImportError:
        return _networkx_missing_result("networkx_christofides")

    matrix = _to_numpy_distances(distances)
    metadata = {"requires_metric_symmetric_complete": True}
    if (
        not _is_complete_finite(matrix)
        or not _is_symmetric(matrix)
        or not _is_metric(matrix)
    ):
        return _result(
            "networkx_christofides",
            None,
            None,
            "skipped_not_applicable",
            metadata,
        )
    graph = _build_networkx_graph(matrix, directed=False)
    try:
        route = nx.approximation.christofides(graph, weight="weight")
    except Exception as exc:  # pragma: no cover - defensive wrapper for optional solver
        return _result(
            "networkx_christofides", None, None, "error", {"error": str(exc)}
        )
    return _networkx_route_result(
        "networkx_christofides", route, matrix, True, metadata
    )


def solve_tsp_networkx_greedy(distances: np.ndarray | Any) -> SolverResult:
    """Run NetworkX greedy TSP for complete finite instances."""
    try:
        import networkx as nx
    except ImportError:
        return _networkx_missing_result("networkx_greedy")

    matrix = _to_numpy_distances(distances)
    if not _is_complete_finite(matrix):
        return _result("networkx_greedy", None, None, "skipped_not_applicable")
    symmetric = _is_symmetric(matrix)
    graph = _build_networkx_graph(matrix, directed=not symmetric)
    try:
        route = nx.approximation.greedy_tsp(
            graph,
            weight="weight",
            source=matrix.shape[0] - 1,
        )
    except Exception as exc:  # pragma: no cover - optional NetworkX behavior
        return _result("networkx_greedy", None, None, "error", {"error": str(exc)})
    return _networkx_route_result("networkx_greedy", route, matrix, symmetric)


def solve_tsp_networkx_simulated_annealing(
    distances: np.ndarray | Any,
    seed: int | None = None,
) -> SolverResult:
    """Run NetworkX simulated annealing TSP for complete finite instances."""
    try:
        import networkx as nx
    except ImportError:
        return _networkx_missing_result("networkx_simulated_annealing")

    matrix = _to_numpy_distances(distances)
    if not _is_complete_finite(matrix):
        return _result(
            "networkx_simulated_annealing", None, None, "skipped_not_applicable"
        )
    symmetric = _is_symmetric(matrix)
    graph = _build_networkx_graph(matrix, directed=not symmetric)
    try:
        route = nx.approximation.simulated_annealing_tsp(
            graph,
            init_cycle="greedy",
            weight="weight",
            source=matrix.shape[0] - 1,
            seed=seed,
        )
    except Exception as exc:  # pragma: no cover - optional NetworkX behavior
        return _result(
            "networkx_simulated_annealing",
            None,
            None,
            "error",
            {"error": str(exc), "seed": seed},
        )
    return _networkx_route_result(
        "networkx_simulated_annealing",
        route,
        matrix,
        symmetric,
        {"seed": seed},
    )


def _python_tsp_missing_result(solver: str) -> SolverResult:
    return _result(solver, None, None, "skipped_missing_dependency")


def solve_tsp_python_tsp_dynamic_programming(
    distances: np.ndarray | Any,
    max_n: int = 18,
) -> SolverResult:
    """Run the optional python-tsp dynamic-programming solver."""
    matrix = _to_numpy_distances(distances)
    if matrix.shape[0] > max_n:
        return _result(
            "python_tsp_dynamic_programming",
            None,
            None,
            "skipped",
            {"max_n": max_n},
        )
    if not _is_complete_finite(matrix):
        return _result(
            "python_tsp_dynamic_programming", None, None, "skipped_not_applicable"
        )
    try:
        from python_tsp.exact import solve_tsp_dynamic_programming  # pyright: ignore[reportMissingImports]
    except ImportError:
        return _python_tsp_missing_result("python_tsp_dynamic_programming")

    try:
        permutation, _distance = solve_tsp_dynamic_programming(matrix)
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        return _result(
            "python_tsp_dynamic_programming",
            None,
            None,
            "error",
            {"error": str(exc)},
        )
    route = canonicalize_cycle(permutation, start=matrix.shape[0] - 1)
    return _networkx_route_result(
        "python_tsp_dynamic_programming", route, matrix, False
    )


def solve_tsp_python_tsp_local_search(
    distances: np.ndarray | Any,
    seed: int | None = None,
) -> SolverResult:
    """Run the optional python-tsp local-search heuristic."""
    matrix = _to_numpy_distances(distances)
    if not _is_complete_finite(matrix):
        return _result("python_tsp_local_search", None, None, "skipped_not_applicable")
    try:
        from python_tsp.heuristics import solve_tsp_local_search  # pyright: ignore[reportMissingImports]
    except ImportError:
        return _python_tsp_missing_result("python_tsp_local_search")

    if seed is not None:
        np.random.seed(seed)
    try:
        permutation, _distance = solve_tsp_local_search(matrix)
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        return _result(
            "python_tsp_local_search", None, None, "error", {"error": str(exc)}
        )
    route = canonicalize_cycle(permutation, start=matrix.shape[0] - 1)
    return _networkx_route_result(
        "python_tsp_local_search",
        route,
        matrix,
        False,
        {"seed": seed},
    )


def solve_tsp_python_tsp_simulated_annealing(
    distances: np.ndarray | Any,
    seed: int | None = None,
) -> SolverResult:
    """Run the optional python-tsp simulated-annealing heuristic."""
    matrix = _to_numpy_distances(distances)
    if not _is_complete_finite(matrix):
        return _result(
            "python_tsp_simulated_annealing", None, None, "skipped_not_applicable"
        )
    try:
        from python_tsp.heuristics import solve_tsp_simulated_annealing  # pyright: ignore[reportMissingImports]
    except ImportError:
        return _python_tsp_missing_result("python_tsp_simulated_annealing")

    if seed is not None:
        np.random.seed(seed)
    try:
        permutation, _distance = solve_tsp_simulated_annealing(matrix)
    except Exception as exc:  # pragma: no cover - optional dependency behavior
        return _result(
            "python_tsp_simulated_annealing",
            None,
            None,
            "error",
            {"error": str(exc)},
        )
    route = canonicalize_cycle(permutation, start=matrix.shape[0] - 1)
    return _networkx_route_result(
        "python_tsp_simulated_annealing",
        route,
        matrix,
        False,
        {"seed": seed},
    )
