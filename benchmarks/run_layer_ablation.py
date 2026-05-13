from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmarks.classical_solvers import solve_tsp_held_karp
from benchmarks.instance_generators import (
    generate_euclidean_tsp,
    generate_random_complete_tsp,
    generate_sparse_hamiltonian_tsp,
)
from benchmarks.metrics import measure_runtime_and_memory, relative_gap
from benchmarks.tau_schedule import load_tau_schedule, tau_for_size
from benchmarks.tn_adapter import solve_tsp_tn


def max_restriction_layers_for_tsp(n: int) -> int:
    """Return the maximum real restriction layers used in the first TN subproblem."""
    return max(n - 2, 0)


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
    edge_probabilities: list[float],
) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    specs: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for n in sizes:
        for seed in seeds:
            distances, metadata = generate_euclidean_tsp(n=n, seed=seed)
            specs.append(
                (f"layer_ablation_euclidean_n{n}_seed{seed}", distances, metadata)
            )

            distances, metadata = generate_random_complete_tsp(
                n=n,
                seed=seed,
                symmetric=False,
            )
            specs.append(
                (
                    f"layer_ablation_random_asymmetric_n{n}_seed{seed}",
                    distances,
                    metadata,
                )
            )

            for edge_probability in edge_probabilities:
                distances, metadata = generate_sparse_hamiltonian_tsp(
                    n=n,
                    seed=seed,
                    edge_probability=edge_probability,
                    symmetric=True,
                )
                instance_id = (
                    "layer_ablation_sparse_hamiltonian_symmetric_"
                    f"n{n}_seed{seed}_p{edge_probability:g}"
                )
                specs.append((instance_id, distances, metadata))
    return specs


def _exact_optimum(
    distances: np.ndarray, n: int, max_exact: int
) -> tuple[float | None, str]:
    if n > max_exact:
        return None, "skipped"
    result = solve_tsp_held_karp(distances, max_n=max_exact)
    if result["status"] != "ok":
        return None, result["status"]
    return float(result["cost"]), "ok"


def _run_single_repetition(
    distances: np.ndarray,
    tau: float,
    n_layers: int,
    run_seed: int,
) -> tuple[dict[str, Any], float, float, float, float]:
    return measure_runtime_and_memory(
        solve_tsp_tn,
        distances,
        tau,
        n_layers,
        run_seed,
        True,
        False,
    )


def _base_row(
    instance_id: str,
    metadata: dict[str, Any],
    optimum_cost: float | None,
    optimum_status: str,
    tau: float,
    max_layers: int,
    layer_ratio: float,
    n_layers: int,
    repeat: int,
    run_seed: int,
    result: dict[str, Any],
    runtime_s: float,
    rss_before_mb: float,
    rss_after_mb: float,
    rss_delta_mb: float,
) -> dict[str, Any]:
    cost = _finite_float(result.get("cost"))
    result_metadata = dict(result.get("metadata", {}))
    result_metadata["instance_metadata"] = metadata
    return {
        "experiment": "layer_ablation",
        "instance_id": instance_id,
        "instance_type": metadata["instance_type"],
        "n": metadata["n"],
        "seed": metadata["seed"],
        "edge_probability": metadata.get("edge_probability", ""),
        "solver": result.get("solver"),
        "status": result.get("status"),
        "route": _serialize_route(result.get("route")),
        "cost": cost if cost is not None else "",
        "optimum_cost": optimum_cost if optimum_cost is not None else "",
        "optimum_status": optimum_status,
        "relative_gap": relative_gap(cost, optimum_cost),
        "valid_route": result_metadata.get("valid_route", ""),
        "finite_edges": result_metadata.get("finite_edges", ""),
        "runtime_s": runtime_s,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_delta_mb": rss_delta_mb,
        "tau": tau,
        "max_layers": max_layers,
        "layer_ratio": layer_ratio,
        "n_layers": n_layers,
        "repeat": repeat,
        "run_seed": run_seed,
        "metadata_json": _json_dumps(result_metadata),
    }


def _add_aggregate_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    frame = pd.DataFrame(rows)
    frame["relative_gap_numeric"] = pd.to_numeric(
        frame["relative_gap"], errors="coerce"
    )
    frame["runtime_s_numeric"] = pd.to_numeric(frame["runtime_s"], errors="coerce")
    frame["valid_numeric"] = frame["valid_route"].astype(str).str.lower().eq("true")
    group_columns = ["instance_id", "layer_ratio", "n_layers"]
    aggregates = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            valid_solution_rate=("valid_numeric", "mean"),
            best_gap=("relative_gap_numeric", "min"),
            median_gap=("relative_gap_numeric", "median"),
            p95_gap=("relative_gap_numeric", lambda series: series.quantile(0.95)),
            mean_runtime_s=("runtime_s_numeric", "mean"),
        )
        .reset_index()
    )

    first_valid_rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_columns, dropna=False):
        key_tuple = (
            cast(tuple[Any, Any, Any], group_key)
            if isinstance(group_key, tuple)
            else (group_key, np.nan, np.nan)
        )
        valid_group = group[group["valid_numeric"]]
        if valid_group.empty:
            first_valid_repeat = np.nan
            runtime_total = group["runtime_s_numeric"].sum()
        else:
            valid_repeats = pd.to_numeric(
                valid_group["repeat"], errors="coerce"
            ).dropna()
            first_valid_repeat = int(float(valid_repeats.min()))
            runtime_total = group[group["repeat"] <= first_valid_repeat][
                "runtime_s_numeric"
            ].sum()
        first_valid_rows.append(
            {
                "instance_id": key_tuple[0],
                "layer_ratio": key_tuple[1],
                "n_layers": key_tuple[2],
                "first_valid_repeat": first_valid_repeat,
                "runtime_total_to_first_valid_s": runtime_total,
            }
        )
    first_valid = pd.DataFrame(first_valid_rows)
    merged = frame.merge(aggregates, on=group_columns, how="left").merge(
        first_valid,
        on=group_columns,
        how="left",
    )
    merged = merged.drop(
        columns=["relative_gap_numeric", "runtime_s_numeric", "valid_numeric"]
    )
    return merged.to_dict(orient="records")


def run_layer_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tau_schedule = load_tau_schedule(args.tau_by_size)
    specs = _instance_specs(args.sizes, args.seeds, args.edge_probabilities)
    for instance_id, distances, metadata in specs:
        n = int(metadata["n"])
        tau = tau_for_size(n, args.tau, tau_schedule)
        optimum_cost, optimum_status = _exact_optimum(distances, n, args.max_exact)
        max_layers = max_restriction_layers_for_tsp(n)
        for layer_ratio in args.layer_ratios:
            n_layers = int(round(layer_ratio * max_layers))
            for repeat in range(args.repeats):
                run_seed = int(metadata["seed"]) * 100_000 + repeat
                (
                    result,
                    runtime_s,
                    rss_before_mb,
                    rss_after_mb,
                    rss_delta_mb,
                ) = _run_single_repetition(distances, tau, n_layers, run_seed)
                rows.append(
                    _base_row(
                        instance_id,
                        metadata,
                        optimum_cost,
                        optimum_status,
                        tau,
                        max_layers,
                        layer_ratio,
                        n_layers,
                        repeat,
                        run_seed,
                        result,
                        runtime_s,
                        rss_before_mb,
                        rss_after_mb,
                        rss_delta_mb,
                    )
                )
    return _add_aggregate_columns(rows)


def write_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TN restriction-layer ablation benchmark."
    )
    parser.add_argument("--out", type=Path, default=Path("results/layer_ablation.csv"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--layer-ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--edge-probabilities",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    parser.add_argument("--repeats", type=int, default=50)
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
    parser.add_argument("--max-exact", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_layer_ablation(args)
    write_rows(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
