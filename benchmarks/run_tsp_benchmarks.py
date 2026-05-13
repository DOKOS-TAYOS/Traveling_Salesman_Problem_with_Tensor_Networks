from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.classical_solvers import (
    solve_tsp_bruteforce,
    solve_tsp_held_karp,
    solve_tsp_networkx_christofides,
    solve_tsp_networkx_greedy,
    solve_tsp_ortools,
    solve_tsp_python_tsp_dynamic_programming,
)
from benchmarks.instance_generators import (
    generate_euclidean_tsp,
    generate_random_complete_tsp,
)
from benchmarks.metrics import (
    has_finite_edges,
    is_valid_tsp_route,
    measure_runtime_and_memory,
    relative_gap,
)
from benchmarks.tau_schedule import load_tau_schedule, tau_for_size
from benchmarks.tn_adapter import solve_tsp_tn

BenchmarkSolver = tuple[str, Callable[[], dict[str, Any]], int | None, float | None]

CSV_COLUMNS = [
    "experiment",
    "instance_id",
    "instance_type",
    "n",
    "seed",
    "solver",
    "status",
    "route",
    "cost",
    "optimum_cost",
    "relative_gap",
    "valid_route",
    "finite_edges",
    "runtime_s",
    "rss_before_mb",
    "rss_after_mb",
    "rss_delta_mb",
    "tau",
    "n_layers",
    "time_limit_s",
    "best_known_cost",
    "gap_to_best_known",
    "exact_optimum_available",
    "metadata_json",
]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _serialize_route(route: list[int] | None) -> str:
    return "" if route is None else json.dumps(route)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _gap_to_best_known(cost: float | None, best_known_cost: float | None) -> float:
    return relative_gap(cost, best_known_cost)


def _instance_specs(
    suite: str,
    sizes: list[int],
    seeds: list[int],
    instance_types: list[str],
) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    specs: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for n in sizes:
        for seed in seeds:
            for instance_type in instance_types:
                if instance_type == "euclidean":
                    distances, metadata = generate_euclidean_tsp(n=n, seed=seed)
                elif instance_type == "random_symmetric":
                    distances, metadata = generate_random_complete_tsp(
                        n=n,
                        seed=seed,
                        symmetric=True,
                    )
                elif instance_type == "random_asymmetric":
                    distances, metadata = generate_random_complete_tsp(
                        n=n,
                        seed=seed,
                        symmetric=False,
                    )
                else:
                    raise ValueError(f"Unknown instance type: {instance_type}")
                instance_id = f"{suite}_{instance_type}_n{n}_seed{seed}"
                specs.append((instance_id, distances, metadata))
    return specs


def _build_solvers(
    suite: str,
    distances: np.ndarray,
    tau: float,
    time_limit_s: float,
    tn_layer_ratios: list[float],
) -> list[BenchmarkSolver]:
    solvers: list[BenchmarkSolver] = [
        (
            "tn_exact",
            lambda: solve_tsp_tn(
                distances,
                tau=tau,
                n_layers=None,
                seed=0,
                normalize=True,
            ),
            None,
            None,
        ),
        ("held_karp", lambda: solve_tsp_held_karp(distances), None, None),
        ("bruteforce", lambda: solve_tsp_bruteforce(distances), None, None),
        (
            "ortools",
            lambda: solve_tsp_ortools(distances, time_limit_s=time_limit_s),
            None,
            time_limit_s,
        ),
        (
            "networkx_christofides",
            lambda: solve_tsp_networkx_christofides(distances),
            None,
            None,
        ),
        ("networkx_greedy", lambda: solve_tsp_networkx_greedy(distances), None, None),
        (
            "python_tsp_dynamic_programming",
            lambda: solve_tsp_python_tsp_dynamic_programming(distances),
            None,
            None,
        ),
    ]

    if suite == "classical_comparison":
        max_layers = max(distances.shape[0] - 2, 0)
        seen_layers: set[int] = set()
        for ratio in tn_layer_ratios:
            n_layers = int(round(ratio * max_layers))
            if n_layers in seen_layers:
                continue
            seen_layers.add(n_layers)
            solvers.insert(
                1,
                (
                    f"tn_n_layers_{n_layers}",
                    lambda layers=n_layers: solve_tsp_tn(
                        distances,
                        tau=tau,
                        n_layers=layers,
                        seed=0,
                        normalize=True,
                    ),
                    n_layers,
                    None,
                ),
            )
    return solvers


def _run_solver(solver: BenchmarkSolver) -> dict[str, Any]:
    solver_name, fn, configured_n_layers, configured_time_limit = solver
    result, runtime_s, rss_before_mb, rss_after_mb, rss_delta_mb = (
        measure_runtime_and_memory(fn)
    )
    result.setdefault("solver", solver_name)
    result["configured_n_layers"] = configured_n_layers
    result["configured_time_limit_s"] = configured_time_limit
    result["runtime_s"] = runtime_s
    result["rss_before_mb"] = rss_before_mb
    result["rss_after_mb"] = rss_after_mb
    result["rss_delta_mb"] = rss_delta_mb
    return result


def _optimum_from_results(results: list[dict[str, Any]]) -> tuple[float | None, bool]:
    for preferred_solver in ("held_karp", "bruteforce"):
        for result in results:
            if result["solver"] == preferred_solver and result["status"] == "ok":
                return float(result["cost"]), True
    return None, False


def _best_known_from_results(results: list[dict[str, Any]]) -> float | None:
    costs = [
        _finite_float(result.get("cost"))
        for result in results
        if result.get("status") == "ok"
    ]
    finite_costs = [cost for cost in costs if cost is not None]
    return min(finite_costs) if finite_costs else None


def _result_to_row(
    suite: str,
    instance_id: str,
    instance_metadata: dict[str, Any],
    distances: np.ndarray,
    result: dict[str, Any],
    optimum_cost: float | None,
    best_known_cost: float | None,
    exact_optimum_available: bool,
    tau: float,
) -> dict[str, Any]:
    route = result.get("route")
    metadata = dict(result.get("metadata", {}))
    metadata["instance_metadata"] = instance_metadata
    valid_route = metadata.get("valid_route")
    finite_edges = metadata.get("finite_edges")
    if route is not None and valid_route is None:
        valid_route = is_valid_tsp_route(route, int(instance_metadata["n"]))
    if route is not None and finite_edges is None and valid_route:
        finite_edges = has_finite_edges(route, distances)

    cost = _finite_float(result.get("cost"))
    return {
        "experiment": suite,
        "instance_id": instance_id,
        "instance_type": instance_metadata["instance_type"],
        "n": instance_metadata["n"],
        "seed": instance_metadata["seed"],
        "solver": result.get("solver"),
        "status": result.get("status"),
        "route": _serialize_route(route),
        "cost": cost if cost is not None else "",
        "optimum_cost": optimum_cost if optimum_cost is not None else "",
        "relative_gap": relative_gap(cost, optimum_cost),
        "valid_route": bool(valid_route) if valid_route is not None else "",
        "finite_edges": bool(finite_edges) if finite_edges is not None else "",
        "runtime_s": result["runtime_s"],
        "rss_before_mb": result["rss_before_mb"],
        "rss_after_mb": result["rss_after_mb"],
        "rss_delta_mb": result["rss_delta_mb"],
        "tau": tau if str(result.get("solver", "")).startswith("tn") else "",
        "n_layers": result.get("configured_n_layers")
        if result.get("configured_n_layers") is not None
        else metadata.get("n_layers", ""),
        "time_limit_s": result.get("configured_time_limit_s") or "",
        "best_known_cost": best_known_cost if best_known_cost is not None else "",
        "gap_to_best_known": _gap_to_best_known(cost, best_known_cost),
        "exact_optimum_available": exact_optimum_available,
        "metadata_json": _json_dumps(metadata),
    }


def run_benchmarks(args: argparse.Namespace) -> list[dict[str, Any]]:
    sizes = [5] if args.dry_run else args.sizes
    seeds = [0] if args.dry_run else args.seeds
    instance_types = ["euclidean"] if args.dry_run else args.instance_types
    tau_schedule = load_tau_schedule(args.tau_by_size)
    specs = _instance_specs(args.suite, sizes, seeds, instance_types)
    rows: list[dict[str, Any]] = []
    for instance_id, distances, metadata in specs:
        tau = tau_for_size(int(metadata["n"]), args.tau, tau_schedule)
        solvers = _build_solvers(
            args.suite,
            distances,
            tau,
            args.time_limit_s,
            args.tn_layer_ratios,
        )
        results = [_run_solver(solver) for solver in solvers]
        optimum_cost, exact_available = _optimum_from_results(results)
        best_known_cost = _best_known_from_results(results)
        for result in results:
            rows.append(
                _result_to_row(
                    args.suite,
                    instance_id,
                    metadata,
                    distances,
                    result,
                    optimum_cost,
                    best_known_cost,
                    exact_available,
                    tau,
                )
            )
    return rows


def write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible TSP benchmark suites."
    )
    parser.add_argument(
        "--suite",
        choices=["small_exact", "classical_comparison"],
        default="small_exact",
    )
    parser.add_argument("--out", type=Path, default=Path("results/tsp_small_exact.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    parser.add_argument(
        "--instance-types",
        nargs="+",
        default=["euclidean", "random_symmetric", "random_asymmetric"],
        choices=["euclidean", "random_symmetric", "random_asymmetric"],
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--tau-by-size",
        type=Path,
        default=None,
        help=(
            "CSV from run_tau_sweep with n and selected_tau columns. "
            "Only exact n values are used; missing n values fall back to --tau."
        ),
    )
    parser.add_argument("--time-limit-s", type=float, default=5.0)
    parser.add_argument(
        "--tn-layer-ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.out == Path("results/tsp_small_exact.csv"):
        args.out = Path("results/tsp_small_exact_dry.csv")
    rows = run_benchmarks(args)
    write_rows(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
