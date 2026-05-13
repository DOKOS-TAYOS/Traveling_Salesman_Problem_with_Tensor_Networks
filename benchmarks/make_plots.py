from __future__ import annotations

# pyright: reportArgumentType=false

import argparse
import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

BENCHMARK_WARNING = (
    "These benchmarks validate correctness and behavior on small/synthetic TSP "
    "instances; they do not claim computational advantage."
)


def _read_csvs(results_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(results_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        frame["source_csv"] = csv_path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))


def _save_runtime_vs_n(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty or "runtime_s" not in frame or "n" not in frame:
        return
    plot_frame = frame.copy()
    plot_frame["runtime_s"] = _numeric_column(plot_frame, "runtime_s")
    plot_frame["n"] = _numeric_column(plot_frame, "n")
    grouped = (
        plot_frame.dropna(subset=["runtime_s", "n"])
        .groupby(["solver", "n"])["runtime_s"]
        .median()
        .reset_index()
    )
    if grouped.empty:
        return
    plt.figure(figsize=(8, 5))
    for solver, solver_frame in grouped.groupby("solver"):
        solver_frame = solver_frame.sort_values("n")
        plt.plot(solver_frame["n"], solver_frame["runtime_s"], marker="o", label=solver)
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("runtime_s (median, log scale)")
    plt.title("Runtime vs n")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_vs_n.png", dpi=160)
    plt.close()


def _save_gap_vs_n(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty or "n" not in frame:
        return
    plot_frame = frame.copy()
    gap = _numeric_column(plot_frame, "relative_gap")
    if gap.notna().sum() == 0:
        gap = _numeric_column(plot_frame, "gap_to_best_known")
    plot_frame["gap"] = gap
    plot_frame["n"] = _numeric_column(plot_frame, "n")
    grouped = (
        plot_frame.dropna(subset=["gap", "n"])
        .groupby(["solver", "n"])["gap"]
        .median()
        .reset_index()
    )
    if grouped.empty:
        return
    plt.figure(figsize=(8, 5))
    for solver, solver_frame in grouped.groupby("solver"):
        solver_frame = solver_frame.sort_values("n")
        plt.plot(solver_frame["n"], solver_frame["gap"], marker="o", label=solver)
    plt.xlabel("n")
    plt.ylabel("relative_gap or gap_to_best_known")
    plt.title("Gap vs n")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "gap_vs_n.png", dpi=160)
    plt.close()


def _layer_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "layer_ratio" not in frame:
        return pd.DataFrame()
    layer = frame.copy()
    layer["layer_ratio"] = _numeric_column(layer, "layer_ratio")
    layer["valid_route_numeric"] = (
        layer["valid_route"].astype(str).str.lower().eq("true")
    )
    layer["relative_gap"] = _numeric_column(layer, "relative_gap")
    layer["n"] = _numeric_column(layer, "n")
    return layer


def _save_layer_validity(frame: pd.DataFrame, out_dir: Path) -> None:
    layer = _layer_frame(frame)
    if layer.empty:
        return
    grouped = (
        layer.dropna(subset=["layer_ratio", "n"])
        .groupby(["n", "layer_ratio"])["valid_route_numeric"]
        .mean()
        .reset_index()
    )
    if grouped.empty:
        return
    plt.figure(figsize=(8, 5))
    for n, n_frame in grouped.groupby("n"):
        n_frame = n_frame.sort_values("layer_ratio")
        n_label = int(float(cast(object, n)))
        plt.plot(
            n_frame["layer_ratio"],
            n_frame["valid_route_numeric"],
            marker="o",
            label=f"n={n_label}",
        )
    plt.xlabel("layer_ratio")
    plt.ylabel("valid route rate")
    plt.ylim(-0.05, 1.05)
    plt.title("Layer ablation validity")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "layer_ablation_validity.png", dpi=160)
    plt.close()


def _save_layer_gap(frame: pd.DataFrame, out_dir: Path) -> None:
    layer = _layer_frame(frame)
    if layer.empty:
        return
    grouped = (
        layer.dropna(subset=["layer_ratio", "relative_gap"])
        .groupby(["instance_type", "layer_ratio"])["relative_gap"]
        .agg(median_gap="median", p95_gap=lambda series: series.quantile(0.95))
        .reset_index()
    )
    if grouped.empty:
        return
    plt.figure(figsize=(8, 5))
    for instance_type, type_frame in grouped.groupby("instance_type"):
        type_frame = type_frame.sort_values("layer_ratio")
        plt.plot(
            type_frame["layer_ratio"],
            type_frame["median_gap"],
            marker="o",
            label=f"{instance_type} median",
        )
        plt.plot(
            type_frame["layer_ratio"],
            type_frame["p95_gap"],
            linestyle="--",
            label=f"{instance_type} p95",
        )
    plt.xlabel("layer_ratio")
    plt.ylabel("relative_gap")
    plt.title("Layer ablation gap")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "layer_ablation_gap.png", dpi=160)
    plt.close()


def _save_tau_sweep_gap(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty or "experiment" not in frame or "tau" not in frame:
        return
    tau_frame = frame[frame["experiment"].astype(str).eq("tau_sweep")].copy()
    if tau_frame.empty:
        return
    tau_frame["tau"] = _numeric_column(tau_frame, "tau")
    tau_frame["n"] = _numeric_column(tau_frame, "n")
    tau_frame["relative_gap"] = _numeric_column(tau_frame, "relative_gap")
    tau_columns = cast(pd.DataFrame, tau_frame[["tau", "n", "relative_gap"]])
    complete_tau_frame = cast(
        pd.DataFrame,
        tau_frame.loc[tau_columns.notna().all(axis=1)].copy(),
    )
    grouped = (
        complete_tau_frame.groupby(["n", "tau"])["relative_gap"].median().reset_index()
    )
    if grouped.empty:
        return
    plt.figure(figsize=(8, 5))
    for n, n_frame in grouped.groupby("n"):
        n_frame = n_frame.sort_values("tau")
        n_label = int(float(cast(object, n)))
        plt.plot(
            n_frame["tau"],
            n_frame["relative_gap"],
            marker="o",
            label=f"n={n_label}",
        )
    plt.xscale("log")
    plt.xlabel("tau")
    plt.ylabel("median relative_gap")
    plt.title("Tau sweep quality")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "tau_sweep_gap.png", dpi=160)
    plt.close()


def _save_tau_schedule(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty or "selected_tau" not in frame:
        return
    tau_frame = frame.copy()
    tau_frame["n"] = _numeric_column(tau_frame, "n")
    tau_frame["selected_tau"] = _numeric_column(tau_frame, "selected_tau")
    selected_columns = cast(pd.DataFrame, tau_frame[["n", "selected_tau"]])
    tau_frame = cast(
        pd.DataFrame,
        tau_frame.loc[selected_columns.notna().all(axis=1)].copy(),
    )
    if tau_frame.empty:
        return
    tau_frame = tau_frame.sort_values(by="n").drop_duplicates(subset=["n"])
    plt.figure(figsize=(8, 5))
    plt.plot(
        tau_frame["n"],
        tau_frame["selected_tau"],
        marker="o",
        label="selected tau",
    )
    plt.xlabel("n")
    plt.ylabel("tau")
    plt.title("Selected tau by n")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "tau_schedule.png", dpi=160)
    plt.close()


def _dependency_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _write_summary(frame: pd.DataFrame, reports_dir: Path, results_dir: Path) -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Benchmark Summary",
        "",
        f"Generated at: {now}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        "",
        "## Relevant Dependencies",
        "",
    ]
    for package in (
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "networkx",
        "ortools",
        "psutil",
        "python-tsp",
    ):
        lines.append(f"- {package}: {_dependency_version(package)}")

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "- `python -m benchmarks.run_tau_sweep ...`",
            "- `python -m benchmarks.run_tsp_benchmarks --suite small_exact ...`",
            "- `python -m benchmarks.run_layer_ablation ...`",
            "- `python -m benchmarks.make_plots --results-dir results --out-dir reports/figures`",
            "",
            "## Result Files",
            "",
        ]
    )
    for csv_path in sorted(results_dir.glob("*.csv")):
        lines.append(f"- `{csv_path.as_posix()}`")

    lines.extend(["", "## Status Counts", ""])
    if frame.empty or "solver" not in frame or "status" not in frame:
        lines.append("No benchmark rows were found.")
    else:
        counts = frame.groupby(["solver", "status"]).size().reset_index()
        counts = counts.rename(columns={0: "rows"})
        lines.append("```text")
        lines.append(counts.to_string(index=False))
        lines.append("```")

    lines.extend(["", "## Scope Note", "", BENCHMARK_WARNING, ""])
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def make_plots(results_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _read_csvs(results_dir)
    _save_runtime_vs_n(frame, out_dir)
    _save_gap_vs_n(frame, out_dir)
    _save_layer_validity(frame, out_dir)
    _save_layer_gap(frame, out_dir)
    _save_tau_sweep_gap(frame, out_dir)
    _save_tau_schedule(frame, out_dir)
    _write_summary(frame, out_dir.parent, results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create benchmark plots and summary.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plots(args.results_dir, args.out_dir)
    print(f"Wrote plots to {args.out_dir} and summary to {args.out_dir.parent}")


if __name__ == "__main__":
    main()
