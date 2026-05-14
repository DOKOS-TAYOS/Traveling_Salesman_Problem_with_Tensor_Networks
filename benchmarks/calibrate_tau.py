from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.classical_solvers import solve_tsp_bruteforce, solve_tsp_held_karp
from benchmarks.config import (
    CALIBRATED_TAU_SOURCE,
    CALIBRATION_SPLIT,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_EVALUATION_SEEDS,
    validate_disjoint_splits,
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
from benchmarks.tn_adapter import solve_tsp_tn

CALIBRATION_COLUMNS = [
    "split",
    "instance_type",
    "n",
    "seed",
    "tau",
    "tau_source",
    "calibration_stage",
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
    "metadata_json",
]

TAU_SCHEDULE_COLUMNS = [
    "split",
    "instance_type",
    "n",
    "selected_tau",
    "criterion",
    "calibration_seeds",
    "tau_values_tested",
    "optimality_rate",
    "median_gap",
    "mean_gap",
    "number_of_instances",
    "first_stage_selected_tau",
    "refinement_step",
    "refinement_neighbors",
    "tau_source",
]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


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
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    specs: list[tuple[np.ndarray, dict[str, Any]]] = []
    for n in sizes:
        for seed in seeds:
            for instance_type in instance_types:
                if instance_type == "euclidean":
                    specs.append(generate_euclidean_tsp(n=n, seed=seed))
                elif instance_type == "random_symmetric":
                    specs.append(
                        generate_random_complete_tsp(n=n, seed=seed, symmetric=True)
                    )
                elif instance_type == "random_asymmetric":
                    specs.append(
                        generate_random_complete_tsp(n=n, seed=seed, symmetric=False)
                    )
                else:
                    raise ValueError(f"Unknown instance type: {instance_type}")
    return specs


def _exact_optimum(
    distances: np.ndarray,
    max_exact: int,
) -> tuple[float | None, str, str]:
    held_karp = solve_tsp_held_karp(distances, max_n=max_exact)
    if held_karp["status"] == "ok":
        return float(held_karp["cost"]), "ok", "held_karp"

    brute = solve_tsp_bruteforce(distances, max_n=min(max_exact, 10))
    if brute["status"] == "ok":
        return float(brute["cost"]), "ok", "bruteforce"

    status = str(held_karp["status"])
    if status == "skipped":
        status = str(brute["status"])
    return None, status, "none"


def _calibration_row(
    distances: np.ndarray,
    metadata: dict[str, Any],
    tau: float,
    result: dict[str, Any],
    runtime_s: float,
    rss_before_mb: float,
    rss_after_mb: float,
    rss_delta_mb: float,
    optimum_cost: float | None,
    optimum_status: str,
    optimum_solver: str,
    tau_source: str,
    calibration_stage: str,
) -> dict[str, Any]:
    route = result.get("route")
    result_metadata = dict(result.get("metadata", {}))
    result_metadata["instance_metadata"] = metadata
    result_metadata["optimum_status"] = optimum_status
    result_metadata["optimum_solver"] = optimum_solver
    valid_route = result_metadata.get("valid_route")
    finite_edges = result_metadata.get("finite_edges")
    if route is not None and valid_route is None:
        valid_route = is_valid_tsp_route(route, int(metadata["n"]))
    if route is not None and finite_edges is None and valid_route:
        finite_edges = has_finite_edges(route, distances)

    cost = _finite_float(result.get("cost"))
    return {
        "split": CALIBRATION_SPLIT,
        "instance_type": metadata["instance_type"],
        "n": metadata["n"],
        "seed": metadata["seed"],
        "tau": tau,
        "tau_source": tau_source,
        "calibration_stage": calibration_stage,
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
        "metadata_json": _json_dumps(result_metadata),
    }


def build_tau_schedule_rows(
    rows: list[dict[str, Any]],
    calibration_seeds: list[int],
    tau_grid: list[float],
    optimal_tolerance: float = 1e-9,
    tau_values_by_n: dict[int, list[float]] | None = None,
    first_stage_selected_by_n: dict[int, float] | None = None,
    refinement_step: float | None = None,
    refinement_neighbors: int | None = None,
) -> list[dict[str, Any]]:
    """Build one selected tau row per size n, aggregating all instance types."""
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame["n_numeric"] = pd.to_numeric(frame["n"], errors="coerce")
    frame["tau_numeric"] = pd.to_numeric(frame["tau"], errors="coerce")
    frame["gap_numeric"] = pd.to_numeric(frame["relative_gap"], errors="coerce")
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
    frame["instance_key"] = (
        frame["instance_type"].astype(str) + "::" + frame["seed"].astype(str)
    )
    stable = frame.copy()
    stable.loc[~stable["valid_numeric"], "gap_numeric"] = np.nan

    grouped = (
        stable.dropna(subset=["n_numeric", "tau_numeric"])
        .groupby(["n_numeric", "tau_numeric"], dropna=False)
        .agg(
            optimality_rate=("optimal_numeric", "mean"),
            median_gap=("gap_numeric", "median"),
            mean_gap=("gap_numeric", "mean"),
            number_of_instances=("instance_key", "nunique"),
            valid_count=("valid_numeric", "sum"),
        )
        .reset_index()
        .rename(columns={"n_numeric": "n", "tau_numeric": "tau"})
    )
    if grouped.empty:
        return []

    grouped["median_gap_sort"] = grouped["median_gap"].fillna(float("inf"))
    grouped["mean_gap_sort"] = grouped["mean_gap"].fillna(float("inf"))
    sorted_frame = grouped.sort_values(
        by=[
            "n",
            "optimality_rate",
            "median_gap_sort",
            "mean_gap_sort",
            "valid_count",
            "tau",
        ],
        ascending=[True, False, True, True, False, True],
    )
    selected = sorted_frame.groupby(["n"], sort=True).head(1)
    schedule_rows: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        n = int(row["n"])
        tested_taus = tau_values_by_n[n] if tau_values_by_n is not None else tau_grid
        schedule_rows.append(
            {
                "split": CALIBRATION_SPLIT,
                "instance_type": "all",
                "n": n,
                "selected_tau": float(row["tau"]),
                "criterion": (
                    "two_stage_unified_by_n_max_optimality_then_median_gap_"
                    "then_lowest_stable_tau"
                ),
                "calibration_seeds": _json_dumps(
                    [int(seed) for seed in calibration_seeds]
                ),
                "tau_values_tested": _json_dumps([float(tau) for tau in tested_taus]),
                "optimality_rate": float(row["optimality_rate"]),
                "median_gap": _nan_to_blank(row["median_gap"]),
                "mean_gap": _nan_to_blank(row["mean_gap"]),
                "number_of_instances": int(row["number_of_instances"]),
                "first_stage_selected_tau": _nan_to_blank(
                    None
                    if first_stage_selected_by_n is None
                    else first_stage_selected_by_n.get(n)
                ),
                "refinement_step": refinement_step
                if refinement_step is not None
                else "",
                "refinement_neighbors": refinement_neighbors
                if refinement_neighbors is not None
                else "",
                "tau_source": CALIBRATED_TAU_SOURCE,
            }
        )
    return schedule_rows


def _nan_to_blank(value: Any) -> float | str:
    value_float = _finite_float(value)
    return value_float if value_float is not None else ""


def build_refined_tau_grid(
    center_tau: float,
    step: float = 0.2,
    neighbors: int = 10,
) -> list[float]:
    """Return positive tau values around a coarse optimum."""
    values = [
        round(float(center_tau) + offset * float(step), 10)
        for offset in range(-int(neighbors), int(neighbors) + 1)
    ]
    positive_values = [value for value in values if value > 0.0]
    return sorted(set(positive_values))


def _run_calibration_stage(
    specs: list[tuple[np.ndarray, dict[str, Any]]],
    tau_values_by_n: dict[int, list[float]],
    max_exact: int,
    stage_label: str,
    tau_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for distances, metadata in specs:
        optimum_cost, optimum_status, optimum_solver = _exact_optimum(
            distances,
            max_exact,
        )
        for tau in tau_values_by_n[int(metadata["n"])]:
            result, runtime_s, rss_before_mb, rss_after_mb, rss_delta_mb = (
                measure_runtime_and_memory(
                    solve_tsp_tn,
                    distances,
                    tau,
                    None,
                    int(metadata["seed"]),
                    True,
                    False,
                )
            )
            rows.append(
                _calibration_row(
                    distances,
                    metadata,
                    tau,
                    result,
                    runtime_s,
                    rss_before_mb,
                    rss_after_mb,
                    rss_delta_mb,
                    optimum_cost,
                    optimum_status,
                    optimum_solver,
                    tau_source,
                    stage_label,
                )
            )
    return rows


def run_calibration(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_disjoint_splits(args.seeds, args.evaluation_seeds)
    specs = _instance_specs(args.sizes, args.seeds, args.instance_types)
    first_stage_grid_by_n = {
        int(n): [float(tau) for tau in args.tau_grid] for n in args.sizes
    }
    first_stage_rows = _run_calibration_stage(
        specs,
        first_stage_grid_by_n,
        args.max_exact,
        "stage_1_integer_grid",
        "calibration_stage_1_integer_grid",
    )
    first_stage_schedule_rows = build_tau_schedule_rows(
        first_stage_rows,
        calibration_seeds=args.seeds,
        tau_grid=args.tau_grid,
        optimal_tolerance=args.optimal_tolerance,
    )
    first_stage_selected_by_n = {
        int(row["n"]): float(row["selected_tau"]) for row in first_stage_schedule_rows
    }
    refined_grid_by_n = {
        n: build_refined_tau_grid(
            center_tau=tau,
            step=args.refinement_step,
            neighbors=args.refinement_neighbors,
        )
        for n, tau in first_stage_selected_by_n.items()
    }
    second_stage_rows = _run_calibration_stage(
        specs,
        refined_grid_by_n,
        args.max_exact,
        "stage_2_local_refinement",
        "calibration_stage_2_local_refinement",
    )
    schedule_rows = build_tau_schedule_rows(
        second_stage_rows,
        calibration_seeds=args.seeds,
        tau_grid=[],
        optimal_tolerance=args.optimal_tolerance,
        tau_values_by_n=refined_grid_by_n,
        first_stage_selected_by_n=first_stage_selected_by_n,
        refinement_step=args.refinement_step,
        refinement_neighbors=args.refinement_neighbors,
    )
    return [*first_stage_rows, *second_stage_rows], schedule_rows


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
        description="Calibrate TN tau on calibration seeds only."
    )
    parser.add_argument("--out", type=Path, default=Path("results/tau_calibration.csv"))
    parser.add_argument(
        "--schedule-out",
        type=Path,
        default=Path("results/tau_schedule.csv"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_CALIBRATION_SEEDS,
        help="Calibration seeds only.",
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_EVALUATION_SEEDS,
        help="Optional evaluation split used only to validate disjoint seeds.",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    parser.add_argument(
        "--instance-types",
        nargs="+",
        default=["euclidean", "random_symmetric", "random_asymmetric"],
        choices=["euclidean", "random_symmetric", "random_asymmetric"],
    )
    parser.add_argument(
        "--tau-grid",
        type=float,
        nargs="+",
        default=[float(tau) for tau in range(1, 41)],
        help="First-stage coarse tau grid.",
    )
    parser.add_argument("--refinement-step", type=float, default=0.2)
    parser.add_argument("--refinement-neighbors", type=int, default=10)
    parser.add_argument("--max-exact", type=int, default=12)
    parser.add_argument("--optimal-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, schedule_rows = run_calibration(args)
    write_rows(rows, args.out, CALIBRATION_COLUMNS)
    write_rows(schedule_rows, args.schedule_out, TAU_SCHEDULE_COLUMNS)
    print(
        f"Wrote {len(rows)} calibration rows to {args.out} and "
        f"{len(schedule_rows)} tau schedule rows to {args.schedule_out}"
    )


if __name__ == "__main__":
    main()
