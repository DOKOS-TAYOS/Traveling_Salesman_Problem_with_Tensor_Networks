from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.classical_solvers import solve_tsp_held_karp
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
from benchmarks.tn_adapter import solve_tsp_tn

TAU_SWEEP_COLUMNS = [
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
    "exact_optimum_available",
    "metadata_json",
]

TAU_BY_SIZE_COLUMNS = [
    "n",
    "selected_tau",
    "rows",
    "ok_rate",
    "valid_solution_rate",
    "optimal_rate",
    "median_gap",
    "p95_gap",
    "max_gap",
    "median_runtime_s",
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


def _instance_specs(
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
                instance_id = f"tau_sweep_{instance_type}_n{n}_seed{seed}"
                specs.append((instance_id, distances, metadata))
    return specs


def _exact_optimum(
    distances: np.ndarray,
    max_exact: int,
) -> tuple[float | None, bool, str]:
    result = solve_tsp_held_karp(distances, max_n=max_exact)
    if result["status"] != "ok":
        return None, False, str(result["status"])
    return float(result["cost"]), True, "ok"


def _result_to_row(
    instance_id: str,
    instance_metadata: dict[str, Any],
    distances: np.ndarray,
    tau: float,
    result: dict[str, Any],
    runtime_s: float,
    rss_before_mb: float,
    rss_after_mb: float,
    rss_delta_mb: float,
    optimum_cost: float | None,
    exact_optimum_available: bool,
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
        "experiment": "tau_sweep",
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
        "runtime_s": runtime_s,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_delta_mb": rss_delta_mb,
        "tau": tau,
        "n_layers": metadata.get("n_layers", ""),
        "exact_optimum_available": exact_optimum_available,
        "metadata_json": _json_dumps(metadata),
    }


def summarize_tau_sweep(
    rows: list[dict[str, Any]],
    optimal_tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    """Aggregate raw tau sweep rows by one tau per size n."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["n_numeric"] = pd.to_numeric(frame["n"], errors="coerce")
    frame["tau_numeric"] = pd.to_numeric(frame["tau"], errors="coerce")
    frame["gap_numeric"] = pd.to_numeric(frame["relative_gap"], errors="coerce")
    frame["runtime_numeric"] = pd.to_numeric(frame["runtime_s"], errors="coerce")
    frame["ok_numeric"] = frame["status"].astype(str).eq("ok")
    frame["valid_numeric"] = (
        frame["ok_numeric"]
        & frame["valid_route"].astype(str).str.lower().eq("true")
        & frame["finite_edges"].astype(str).str.lower().eq("true")
        & frame["gap_numeric"].notna()
    )
    frame["optimal_numeric"] = frame["valid_numeric"] & (
        frame["gap_numeric"].abs() <= optimal_tolerance
    )
    valid_frame = frame.copy()
    valid_frame.loc[~valid_frame["valid_numeric"], "gap_numeric"] = np.nan
    grouped = (
        valid_frame.dropna(subset=["n_numeric", "tau_numeric"])
        .groupby(["n_numeric", "tau_numeric"], dropna=False)
        .agg(
            rows=("tau_numeric", "size"),
            ok_rate=("ok_numeric", "mean"),
            valid_solution_rate=("valid_numeric", "mean"),
            optimal_rate=("optimal_numeric", "mean"),
            median_gap=("gap_numeric", "median"),
            p95_gap=("gap_numeric", lambda series: series.quantile(0.95)),
            max_gap=("gap_numeric", "max"),
            median_runtime_s=("runtime_numeric", "median"),
        )
        .reset_index()
        .rename(columns={"n_numeric": "n", "tau_numeric": "tau"})
    )
    grouped["n"] = grouped["n"].astype(int)
    return grouped.to_dict(orient="records")


def select_tau_by_size(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select one tau per n from aggregated tau-sweep rows."""
    if not summary_rows:
        return []
    frame = pd.DataFrame(summary_rows)
    for column in (
        "ok_rate",
        "valid_solution_rate",
        "optimal_rate",
        "median_gap",
        "p95_gap",
        "median_runtime_s",
        "tau",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["median_gap_sort"] = frame["median_gap"].fillna(float("inf"))
    frame["p95_gap_sort"] = frame["p95_gap"].fillna(float("inf"))
    frame["median_runtime_sort"] = frame["median_runtime_s"].fillna(float("inf"))
    sorted_frame = frame.sort_values(
        by=[
            "n",
            "optimal_rate",
            "valid_solution_rate",
            "median_gap_sort",
            "p95_gap_sort",
            "median_runtime_sort",
            "tau",
        ],
        ascending=[True, False, False, True, True, True, True],
    )
    selected = sorted_frame.groupby("n", sort=True).head(1).copy()
    selected["selected_tau"] = selected["tau"]
    selected = selected.drop(
        columns=["median_gap_sort", "p95_gap_sort", "median_runtime_sort"]
    )
    return selected.to_dict(orient="records")


def run_tau_sweep(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sizes = [5] if args.dry_run else args.sizes
    seeds = [0] if args.dry_run else args.seeds
    instance_types = ["euclidean"] if args.dry_run else args.instance_types
    taus = [0.5, 1.0] if args.dry_run else args.taus
    specs = _instance_specs(sizes, seeds, instance_types)
    rows: list[dict[str, Any]] = []
    for instance_id, distances, metadata in specs:
        optimum_cost, exact_available, optimum_status = _exact_optimum(
            distances,
            args.max_exact,
        )
        for tau in taus:
            result, runtime_s, rss_before_mb, rss_after_mb, rss_delta_mb = (
                measure_runtime_and_memory(
                    solve_tsp_tn,
                    distances,
                    tau,
                    None,
                    0,
                    True,
                    False,
                )
            )
            result_metadata = dict(result.get("metadata", {}))
            result_metadata["optimum_status"] = optimum_status
            result["metadata"] = result_metadata
            rows.append(
                _result_to_row(
                    instance_id,
                    metadata,
                    distances,
                    tau,
                    result,
                    runtime_s,
                    rss_before_mb,
                    rss_after_mb,
                    rss_delta_mb,
                    optimum_cost,
                    exact_available,
                )
            )
    summary_rows = summarize_tau_sweep(rows, optimal_tolerance=args.optimal_tolerance)
    selected_rows = select_tau_by_size(summary_rows)
    return rows, selected_rows


def write_rows(
    rows: list[dict[str, Any]],
    out_path: Path,
    fieldnames: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep TN tau values and select one calibrated tau per n."
    )
    parser.add_argument("--out", type=Path, default=Path("results/tau_sweep.csv"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/tau_by_size.csv"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10, 12])
    parser.add_argument(
        "--instance-types",
        nargs="+",
        default=["euclidean", "random_symmetric", "random_asymmetric"],
        choices=["euclidean", "random_symmetric", "random_asymmetric"],
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    )
    parser.add_argument("--max-exact", type=int, default=12)
    parser.add_argument("--optimal-tolerance", type=float, default=1e-9)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        if args.out == Path("results/tau_sweep.csv"):
            args.out = Path("results/tau_sweep_dry.csv")
        if args.summary_out == Path("results/tau_by_size.csv"):
            args.summary_out = Path("results/tau_by_size_dry.csv")
    rows, selected_rows = run_tau_sweep(args)
    write_rows(rows, args.out, TAU_SWEEP_COLUMNS)
    write_rows(selected_rows, args.summary_out, TAU_BY_SIZE_COLUMNS)
    print(
        f"Wrote {len(rows)} rows to {args.out} and "
        f"{len(selected_rows)} rows to {args.summary_out}"
    )


if __name__ == "__main__":
    main()
