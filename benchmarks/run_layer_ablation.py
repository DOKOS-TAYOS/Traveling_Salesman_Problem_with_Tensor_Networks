from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.classical_solvers import solve_tsp_held_karp
from benchmarks.config import (
    DEFAULT_EVALUATION_SEEDS,
    EVALUATION_SPLIT,
    validate_disjoint_splits,
)
from benchmarks.instance_generators import (
    generate_euclidean_tsp,
    generate_random_complete_tsp,
    generate_sparse_hamiltonian_tsp,
)
from benchmarks.metrics import measure_runtime_and_memory, relative_gap
from benchmarks.tau_schedule import (
    load_tau_schedule,
    tau_for_instance,
    tau_source_for_instance,
)
from benchmarks.tn_adapter import solve_tsp_tn

ROW_COLUMNS = [
    "experiment",
    "split",
    "instance_type",
    "n",
    "seed",
    "repeat",
    "layer_seed",
    "layer_ratio",
    "n_layers_requested",
    "n_layers_effective",
    "max_layers",
    "exact_mode",
    "tau",
    "tau_source",
    "route",
    "valid_route",
    "finite_edges",
    "cost",
    "optimum_cost",
    "best_known_cost",
    "relative_gap",
    "gap_to_best_known",
    "exact_optimum_available",
    "runtime_s",
    "status",
    "metadata_json",
]

SUMMARY_COLUMNS = [
    "split",
    "instance_type",
    "n",
    "layer_ratio",
    "number_of_instances",
    "repeats",
    "valid_rate",
    "finite_edge_rate",
    "optimality_rate",
    "mean_gap",
    "median_gap",
    "p95_gap",
    "best_gap",
    "mean_runtime_s",
    "median_runtime_s",
    "first_valid_repeat_median",
]


@dataclass(frozen=True)
class LayerSelection:
    n_layers_requested: int | None
    n_layers_effective: int
    max_layers: int
    exact_mode: bool


def max_restriction_layers_for_tsp(n: int) -> int:
    """Return the maximum restriction layers in the first TN subproblem."""
    return max(n - 2, 0)


def n_layers_for_layer_ratio(n: int, layer_ratio: float) -> LayerSelection:
    """Map a layer ratio to solver arguments; 1.0 uses exact mode."""
    max_layers = max_restriction_layers_for_tsp(n)
    if layer_ratio >= 1.0:
        return LayerSelection(
            n_layers_requested=None,
            n_layers_effective=max_layers,
            max_layers=max_layers,
            exact_mode=True,
        )
    requested = int(round(max(0.0, layer_ratio) * max_layers))
    return LayerSelection(
        n_layers_requested=requested,
        n_layers_effective=requested,
        max_layers=max_layers,
        exact_mode=False,
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _serialize_route(route: list[int] | None) -> str:
    return "" if route is None else json.dumps(route)


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    value_float = float(value)
    return value_float if math.isfinite(value_float) else None


def _instance_specs(
    sizes: list[int],
    seeds: list[int],
    instance_types: list[str],
    edge_probabilities: list[float],
) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    specs: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for n in sizes:
        for seed in seeds:
            for instance_type in instance_types:
                if instance_type == "euclidean":
                    distances, metadata = generate_euclidean_tsp(n=n, seed=seed)
                    specs.append(
                        (
                            f"layer_ablation_euclidean_n{n}_seed{seed}",
                            distances,
                            metadata,
                        )
                    )
                elif instance_type == "random_symmetric":
                    distances, metadata = generate_random_complete_tsp(
                        n=n,
                        seed=seed,
                        symmetric=True,
                    )
                    specs.append(
                        (
                            f"layer_ablation_random_symmetric_n{n}_seed{seed}",
                            distances,
                            metadata,
                        )
                    )
                elif instance_type == "random_asymmetric":
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
                elif instance_type == "sparse_hamiltonian_symmetric":
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
                else:
                    raise ValueError(
                        f"Unknown layer-ablation instance type: {instance_type}"
                    )
    return specs


def _exact_optimum(
    distances: np.ndarray,
    n: int,
    max_exact: int,
) -> tuple[float | None, bool, str]:
    if n > max_exact:
        return None, False, "skipped"
    result = solve_tsp_held_karp(distances, max_n=max_exact)
    if result["status"] != "ok":
        return None, False, str(result["status"])
    return float(result["cost"]), True, "ok"


def _run_single_repetition(
    distances: np.ndarray,
    tau: float,
    n_layers: int | None,
    layer_seed: int,
) -> tuple[dict[str, Any], float, float, float, float]:
    return measure_runtime_and_memory(
        solve_tsp_tn,
        distances,
        tau,
        n_layers,
        layer_seed,
        True,
        False,
    )


def _base_row(
    instance_id: str,
    metadata: dict[str, Any],
    optimum_cost: float | None,
    exact_optimum_available: bool,
    tau: float,
    tau_source: str,
    selection: LayerSelection,
    layer_ratio: float,
    repeat: int,
    layer_seed: int,
    result: dict[str, Any],
    runtime_s: float,
) -> dict[str, Any]:
    cost = _finite_float(result.get("cost"))
    result_metadata = dict(result.get("metadata", {}))
    result_metadata["instance_metadata"] = metadata
    result_metadata["layer_ratio"] = layer_ratio
    result_metadata["n_layers_requested"] = selection.n_layers_requested
    result_metadata["n_layers_effective"] = selection.n_layers_effective
    result_metadata["exact_mode"] = selection.exact_mode
    return {
        "experiment": "layer_ablation",
        "split": EVALUATION_SPLIT,
        "instance_id": instance_id,
        "instance_type": metadata["instance_type"],
        "n": metadata["n"],
        "seed": metadata["seed"],
        "repeat": repeat,
        "layer_seed": layer_seed,
        "layer_ratio": layer_ratio,
        "n_layers_requested": selection.n_layers_requested,
        "n_layers_effective": selection.n_layers_effective,
        "max_layers": selection.max_layers,
        "exact_mode": selection.exact_mode,
        "tau": tau,
        "tau_source": tau_source,
        "route": _serialize_route(result.get("route")),
        "valid_route": result_metadata.get("valid_route", ""),
        "finite_edges": result_metadata.get("finite_edges", ""),
        "cost": cost if cost is not None else "",
        "optimum_cost": optimum_cost if optimum_cost is not None else "",
        "best_known_cost": "",
        "relative_gap": relative_gap(cost, optimum_cost),
        "gap_to_best_known": "",
        "exact_optimum_available": exact_optimum_available,
        "runtime_s": runtime_s,
        "status": result.get("status"),
        "metadata_json": _json_dumps(result_metadata),
    }


def _fill_best_known(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finite_costs = [
        _finite_float(row.get("cost")) for row in rows if row.get("status") == "ok"
    ]
    optimum_values = [_finite_float(row.get("optimum_cost")) for row in rows]
    candidates = [
        value for value in [*finite_costs, *optimum_values] if value is not None
    ]
    best_known_cost = min(candidates) if candidates else None
    for row in rows:
        cost = _finite_float(row.get("cost"))
        row["best_known_cost"] = best_known_cost if best_known_cost is not None else ""
        row["gap_to_best_known"] = relative_gap(cost, best_known_cost)
    return rows


def build_layer_ablation_summary(
    rows: list[dict[str, Any]],
    optimal_tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    """Aggregate layer-ablation rows without dropping invalid routes."""
    if not rows:
        return []

    grouped_rows: dict[tuple[str, str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["split"]),
            str(row["instance_type"]),
            int(row["n"]),
            float(row["layer_ratio"]),
        )
        grouped_rows.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped_rows.items()):
        split, instance_type, n, layer_ratio = key
        valid_flags = [_truthy(row.get("valid_route")) for row in group]
        finite_flags = [_truthy(row.get("finite_edges")) for row in group]
        gaps = [
            _first_finite(row.get("relative_gap"), row.get("gap_to_best_known"))
            for row in group
        ]
        finite_gaps = [gap for gap in gaps if gap is not None]
        runtimes = [_finite_float(row.get("runtime_s")) for row in group]
        finite_runtimes = [runtime for runtime in runtimes if runtime is not None]
        optimal_flags = [
            valid and finite and gap is not None and abs(gap) <= optimal_tolerance
            for valid, finite, gap in zip(valid_flags, finite_flags, gaps, strict=True)
        ]
        first_valid_repeats = _first_valid_repeats_by_instance(group)
        summaries.append(
            {
                "split": split,
                "instance_type": instance_type,
                "n": n,
                "layer_ratio": layer_ratio,
                "number_of_instances": len(
                    {str(row.get("instance_id")) for row in group}
                ),
                "repeats": len({int(row["repeat"]) for row in group}),
                "valid_rate": _mean([float(flag) for flag in valid_flags]),
                "finite_edge_rate": _mean([float(flag) for flag in finite_flags]),
                "optimality_rate": _mean([float(flag) for flag in optimal_flags]),
                "mean_gap": _mean(finite_gaps) if finite_gaps else "",
                "median_gap": _quantile(finite_gaps, 0.5) if finite_gaps else "",
                "p95_gap": _quantile(finite_gaps, 0.95) if finite_gaps else "",
                "best_gap": min(finite_gaps) if finite_gaps else "",
                "mean_runtime_s": _mean(finite_runtimes) if finite_runtimes else "",
                "median_runtime_s": _quantile(finite_runtimes, 0.5)
                if finite_runtimes
                else "",
                "first_valid_repeat_median": float(np.median(first_valid_repeats))
                if first_valid_repeats
                else "",
            }
        )
    return summaries


def _truthy(value: Any) -> bool:
    return str(value).lower() == "true"


def _first_finite(*values: Any) -> float | None:
    for value in values:
        value_float = _finite_float(value)
        if value_float is not None:
            return value_float
    return None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def _first_valid_repeats_by_instance(rows: list[dict[str, Any]]) -> list[int]:
    repeats_by_instance: dict[str, list[int]] = {}
    for row in rows:
        if not _truthy(row.get("valid_route")):
            continue
        instance_id = str(row.get("instance_id"))
        repeats_by_instance.setdefault(instance_id, []).append(int(row["repeat"]))
    return [min(repeats) for repeats in repeats_by_instance.values()]


def run_layer_ablation(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tau_schedule = load_tau_schedule(args.tau_schedule)
    if tau_schedule is not None and tau_schedule.calibration_seeds:
        validate_disjoint_splits(tau_schedule.calibration_seeds, args.seeds)

    all_rows: list[dict[str, Any]] = []
    specs = _instance_specs(
        args.sizes,
        args.seeds,
        args.instance_types,
        args.edge_probabilities,
    )
    for instance_id, distances, metadata in specs:
        n = int(metadata["n"])
        instance_type = str(metadata["instance_type"])
        tau = tau_for_instance(instance_type, n, args.tau, tau_schedule)
        tau_source = tau_source_for_instance(instance_type, n, tau_schedule)
        optimum_cost, exact_available, _optimum_status = _exact_optimum(
            distances,
            n,
            args.max_exact,
        )
        instance_rows: list[dict[str, Any]] = []
        for layer_ratio in args.layer_ratios:
            selection = n_layers_for_layer_ratio(n, layer_ratio)
            for repeat in range(args.repeats):
                layer_seed = int(metadata["seed"]) * 100_000 + repeat
                result, runtime_s, _rss_before_mb, _rss_after_mb, _rss_delta_mb = (
                    _run_single_repetition(
                        distances,
                        tau,
                        selection.n_layers_requested,
                        layer_seed,
                    )
                )
                instance_rows.append(
                    _base_row(
                        instance_id,
                        metadata,
                        optimum_cost,
                        exact_available,
                        tau,
                        tau_source,
                        selection,
                        layer_ratio,
                        repeat,
                        layer_seed,
                        result,
                        runtime_s,
                    )
                )
        all_rows.extend(_fill_best_known(instance_rows))
    summary_rows = build_layer_ablation_summary(
        all_rows,
        optimal_tolerance=args.optimal_tolerance,
    )
    return all_rows, summary_rows


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
        description="Run the TN restriction-layer ablation benchmark."
    )
    parser.add_argument("--out", type=Path, default=Path("results/layer_ablation.csv"))
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/layer_ablation_summary.csv"),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_EVALUATION_SEEDS,
        help="Evaluation seeds only.",
    )
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
    parser.add_argument(
        "--instance-types",
        nargs="+",
        default=["euclidean", "random_asymmetric", "sparse_hamiltonian_symmetric"],
        choices=[
            "euclidean",
            "random_symmetric",
            "random_asymmetric",
            "sparse_hamiltonian_symmetric",
        ],
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--tau-schedule",
        "--tau-by-size",
        dest="tau_schedule",
        type=Path,
        default=None,
        help="CSV produced by benchmarks.calibrate_tau with selected_tau values.",
    )
    parser.add_argument("--max-exact", type=int, default=12)
    parser.add_argument("--optimal-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary_rows = run_layer_ablation(args)
    write_rows(rows, args.out, ROW_COLUMNS)
    write_rows(summary_rows, args.summary_out, SUMMARY_COLUMNS)
    print(
        f"Wrote {len(rows)} rows to {args.out} and "
        f"{len(summary_rows)} summary rows to {args.summary_out}"
    )


if __name__ == "__main__":
    main()
