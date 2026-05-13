from __future__ import annotations

# pyright: reportArgumentType=false

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DPI = 600
FIGURE_SIZE = (5.4, 3.4)
WIDE_FIGURE_SIZE = (7.0, 3.2)
COMPARISON_MARKER_SIZE = 3.2
SOLVER_LABELS = {
    "held_karp": "Held-Karp",
    "ortools": "OR-Tools",
    "networkx_greedy": "NetworkX greedy",
    "tn_exact": "TN calibrated",
}


def prepare_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add numeric and boolean columns used by the paper aggregations."""
    prepared = frame.copy()
    prepared["gap"] = _numeric_column(prepared, "relative_gap")
    prepared["runtime"] = _numeric_column(prepared, "runtime_s")
    prepared["tau_numeric"] = _numeric_column(prepared, "tau")
    prepared["n_numeric"] = _numeric_column(prepared, "n")
    prepared["valid_solution"] = (
        prepared["status"].astype(str).eq("ok")
        & prepared["valid_route"].astype(str).str.lower().eq("true")
        & prepared["finite_edges"].astype(str).str.lower().eq("true")
        & prepared["gap"].notna()
    )
    prepared["optimal_solution"] = prepared["valid_solution"] & (
        prepared["gap"].abs() <= 1e-9
    )
    prepared["gap_le_1"] = prepared["valid_solution"] & (prepared["gap"] <= 0.01)
    prepared["gap_le_5"] = prepared["valid_solution"] & (prepared["gap"] <= 0.05)
    prepared["gap_le_10"] = prepared["valid_solution"] & (prepared["gap"] <= 0.10)
    return prepared


def tau_optimal_rate_by_size(frame: pd.DataFrame) -> pd.DataFrame:
    """Return optimal-rate and gap summaries by problem size and tau."""
    complete = frame[
        frame[["n_numeric", "tau_numeric", "gap", "runtime"]].notna().all(axis=1)
    ].copy()
    grouped = (
        complete.groupby(["n_numeric", "tau_numeric"], dropna=False)
        .agg(
            rows=("tau_numeric", "size"),
            optimal_rate=("optimal_solution", "mean"),
            median_gap=("gap", "median"),
            p95_gap=("gap", lambda series: series.quantile(0.95)),
            max_gap=("gap", "max"),
            median_runtime_s=("runtime", "median"),
        )
        .reset_index()
        .rename(columns={"n_numeric": "n", "tau_numeric": "tau"})
    )
    grouped["n"] = grouped["n"].astype(int)
    return grouped.sort_values(by=["n", "tau"]).reset_index(drop=True)


def tau_schedule_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return selected tau values sorted by problem size."""
    schedule = frame.copy()
    schedule["n"] = _numeric_column(schedule, "n")
    schedule["selected_tau"] = _numeric_column(schedule, "selected_tau")
    complete = schedule[schedule[["n", "selected_tau"]].notna().all(axis=1)].copy()
    selected = cast(pd.DataFrame, complete[["n", "selected_tau"]])
    return selected.sort_values(by="n").reset_index(drop=True)


def optimal_rate_axis_limits(summary: pd.DataFrame) -> tuple[float, float]:
    """Return a zoomed percentage axis for optimal-rate plots."""
    if summary.empty or "optimal_rate" not in summary:
        return 0.0, 102.0
    rates = _numeric_column(summary, "optimal_rate").dropna()
    if rates.empty:
        return 0.0, 102.0
    min_percent = 100.0 * float(rates.min())
    lower = max(0.0, 5.0 * np.floor((min_percent - 5.0) / 5.0))
    return float(lower), 102.0


def run_quality_summary(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize TN exact quality for named benchmark runs."""
    rows: list[dict[str, float | int | str]] = []
    for run_name, frame in runs.items():
        tn_frame = frame[frame["solver"].astype(str).eq("tn_exact")].copy()
        gap = cast(pd.Series, tn_frame["gap"])
        runtime = cast(pd.Series, tn_frame["runtime"])
        rows.append(
            {
                "run": run_name,
                "rows": len(tn_frame),
                "optimal_rate": float(tn_frame["optimal_solution"].mean()),
                "gap_le_1_rate": float(tn_frame["gap_le_1"].mean()),
                "gap_le_5_rate": float(tn_frame["gap_le_5"].mean()),
                "gap_le_10_rate": float(tn_frame["gap_le_10"].mean()),
                "p95_gap": float(gap.quantile(0.95)),
                "max_gap": float(gap.max()),
                "median_runtime_s": float(runtime.median()),
            }
        )
    return pd.DataFrame(rows)


def solver_comparison_summary(
    frame: pd.DataFrame,
    solvers: Iterable[str],
) -> pd.DataFrame:
    """Summarize paper-facing metrics for selected solvers."""
    rows: list[dict[str, float | int | str]] = []
    for solver in solvers:
        solver_frame = frame[frame["solver"].astype(str).eq(solver)].copy()
        gap = cast(pd.Series, solver_frame["gap"])
        runtime = cast(pd.Series, solver_frame["runtime"])
        rows.append(
            {
                "solver": solver,
                "solver_label": SOLVER_LABELS.get(solver, solver),
                "rows": len(solver_frame),
                "optimal_rate": float(solver_frame["optimal_solution"].mean()),
                "gap_le_1_rate": float(solver_frame["gap_le_1"].mean()),
                "max_gap": float(gap.max()),
                "p95_gap": float(gap.quantile(0.95)),
                "median_runtime_s": float(runtime.median()),
            }
        )
    return pd.DataFrame(rows)


def solver_comparison_by_size_summary(
    frame: pd.DataFrame,
    solvers: Iterable[str],
) -> pd.DataFrame:
    """Summarize selected solver metrics by problem size."""
    rows: list[dict[str, float | int | str]] = []
    for solver_index, solver in enumerate(solvers):
        solver_mask = cast(pd.Series, frame["solver"].astype(str).eq(solver))
        solver_frame = cast(pd.DataFrame, frame.loc[solver_mask].copy())
        n_numeric = cast(pd.Series, solver_frame["n_numeric"])
        complete = cast(pd.DataFrame, solver_frame.loc[n_numeric.notna()].copy())
        if complete.empty:
            continue
        grouped = (
            complete.groupby("n_numeric", dropna=False)
            .agg(
                rows=("n_numeric", "size"),
                optimal_rate=("optimal_solution", "mean"),
                max_gap=("gap", "max"),
                median_runtime_s=("runtime", "median"),
            )
            .reset_index()
            .rename(columns={"n_numeric": "n"})
        )
        grouped["n"] = grouped["n"].astype(int)
        for row in grouped.sort_values(by="n").to_dict(orient="records"):
            rows.append(
                {
                    "solver": solver,
                    "solver_label": SOLVER_LABELS.get(solver, solver),
                    "solver_order": solver_index,
                    "n": int(row["n"]),
                    "rows": int(row["rows"]),
                    "optimal_rate": float(row["optimal_rate"]),
                    "max_gap": float(row["max_gap"]),
                    "median_runtime_s": float(row["median_runtime_s"]),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(by=["solver_order", "n"]).reset_index(drop=True)
    )


def layer_ablation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return quality summaries by restriction-layer ratio."""
    layer = frame.copy()
    layer["layer_ratio_numeric"] = _numeric_column(layer, "layer_ratio")
    complete = layer[layer[["layer_ratio_numeric", "gap"]].notna().all(axis=1)].copy()
    grouped = (
        complete.groupby("layer_ratio_numeric", dropna=False)
        .agg(
            rows=("gap", "size"),
            optimal_rate=("optimal_solution", "mean"),
            gap_le_10_rate=("gap_le_10", "mean"),
            median_gap=("gap", "median"),
            p95_gap=("gap", lambda series: series.quantile(0.95)),
            max_gap=("gap", "max"),
            median_runtime_s=("runtime", "median"),
        )
        .reset_index()
        .rename(columns={"layer_ratio_numeric": "layer_ratio"})
    )
    return grouped.sort_values(by="layer_ratio").reset_index(drop=True)


def layer_ablation_by_size_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return quality summaries by problem size and restriction-layer ratio."""
    layer = frame.copy()
    layer["layer_ratio_numeric"] = _numeric_column(layer, "layer_ratio")
    complete = layer[
        layer[["n_numeric", "layer_ratio_numeric", "gap"]].notna().all(axis=1)
    ].copy()
    grouped = (
        complete.groupby(["n_numeric", "layer_ratio_numeric"], dropna=False)
        .agg(
            rows=("gap", "size"),
            optimal_rate=("optimal_solution", "mean"),
            gap_le_10_rate=("gap_le_10", "mean"),
            median_gap=("gap", "median"),
            p95_gap=("gap", lambda series: series.quantile(0.95)),
            max_gap=("gap", "max"),
            median_runtime_s=("runtime", "median"),
        )
        .reset_index()
        .rename(columns={"n_numeric": "n", "layer_ratio_numeric": "layer_ratio"})
    )
    grouped["n"] = grouped["n"].astype(int)
    return grouped.sort_values(by=["n", "layer_ratio"]).reset_index(drop=True)


def make_paper_figures(
    results_dir: Path,
    out_dir: Path,
) -> list[Path]:
    """Create publication-style figures from the fine-calibrated benchmark CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_by_size = pd.read_csv(results_dir / "tau_by_size_fine.csv")
    tau_sweep = prepare_result_frame(pd.read_csv(results_dir / "tau_sweep_fine.csv"))
    small_tau_1 = prepare_result_frame(pd.read_csv(results_dir / "tsp_small_exact.csv"))
    small_coarse = prepare_result_frame(
        pd.read_csv(results_dir / "tsp_small_exact_tau_calibrated.csv")
    )
    small_fine = prepare_result_frame(
        pd.read_csv(results_dir / "tsp_small_exact_tau_fine.csv")
    )
    classical_fine = prepare_result_frame(
        pd.read_csv(results_dir / "classical_comparison_tau_fine.csv")
    )
    layer_fine = prepare_result_frame(
        pd.read_csv(results_dir / "layer_ablation_tau_fine.csv")
    )

    written: list[Path] = []
    written.extend(_save_tau_schedule(tau_by_size, out_dir))
    written.extend(_save_tau_optimal_rate(tau_sweep, out_dir))
    written.extend(
        _save_quality_comparison(
            {
                r"$\tau=1$": small_tau_1,
                "coarse calibration": small_coarse,
                "fine calibration": small_fine,
            },
            out_dir,
        )
    )
    written.extend(_save_solver_comparison(classical_fine, out_dir))
    written.extend(_save_layer_ablation(layer_fine, out_dir))
    return written


def _save_tau_schedule(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    plot_frame = tau_schedule_summary(frame)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        plot_frame["n"],
        plot_frame["selected_tau"],
        marker="o",
        linewidth=1.8,
    )
    ax.set_xlabel("Number of cities $n$")
    ax.set_ylabel("Calibrated $\\tau$")
    ax.set_xticks(plot_frame["n"].tolist())
    _style_axis(ax)
    return _save_figure(fig, out_dir, "paper_tau_schedule")


def _save_tau_optimal_rate(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    summary = tau_optimal_rate_by_size(frame)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for n, n_frame in summary.groupby("n"):
        n_frame = n_frame.sort_values(by="tau")
        ax.plot(
            n_frame["tau"],
            100.0 * n_frame["optimal_rate"],
            marker="o",
            linewidth=1.5,
            markersize=4.0,
            label=f"$n={int(cast(float, n))}$",
        )
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("Optimal tours found (%)")
    ax.set_ylim(*optimal_rate_axis_limits(summary))
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _style_axis(ax)
    return _save_figure(fig, out_dir, "paper_tau_optimal_rate_by_n")


def _save_quality_comparison(
    runs: dict[str, pd.DataFrame],
    out_dir: Path,
) -> list[Path]:
    summary = run_quality_summary(runs)
    labels = summary["run"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    axes[0].bar(x, 100.0 * summary["optimal_rate"])
    axes[0].set_ylabel("Optimal tours found (%)")
    axes[0].set_ylim(0.0, 105.0)

    axes[1].bar(x, 100.0 * summary["p95_gap"])
    axes[1].set_ylabel("95th percentile gap (%)")

    axes[2].bar(x, 100.0 * summary["max_gap"])
    axes[2].set_ylabel("Worst gap (%)")

    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        _style_axis(ax)
    fig.tight_layout()
    return _save_figure(fig, out_dir, "paper_tau_calibration_quality")


def _save_solver_comparison(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    summary = solver_comparison_by_size_summary(
        frame,
        solvers=["held_karp", "ortools", "networkx_greedy", "tn_exact"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True)
    for _solver, solver_frame in summary.groupby("solver", sort=False):
        solver_frame = solver_frame.sort_values(by="n")
        label = str(solver_frame["solver_label"].iloc[0])
        axes[0].plot(
            solver_frame["n"],
            100.0 * solver_frame["optimal_rate"],
            marker="o",
            linewidth=1.5,
            markersize=COMPARISON_MARKER_SIZE,
            label=label,
        )
        axes[1].plot(
            solver_frame["n"],
            100.0 * solver_frame["max_gap"],
            marker="o",
            linewidth=1.5,
            markersize=COMPARISON_MARKER_SIZE,
            label=label,
        )
        axes[2].plot(
            solver_frame["n"],
            solver_frame["median_runtime_s"],
            marker="o",
            linewidth=1.5,
            markersize=COMPARISON_MARKER_SIZE,
            label=label,
        )

    axes[0].set_ylabel("Optimal tours found (%)")
    axes[0].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("Worst gap (%)")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Median runtime (s)")

    for ax in axes:
        ax.set_xlabel("Number of cities $n$")
        _style_axis(ax)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save_figure(fig, out_dir, "paper_solver_comparison")


def _save_layer_ablation(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    summary = layer_ablation_by_size_summary(frame)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True)
    for n, n_frame in summary.groupby("n", sort=True):
        n_frame = n_frame.sort_values(by="layer_ratio")
        label = f"$n={int(cast(float, n))}$"
        axes[0].plot(
            n_frame["layer_ratio"],
            100.0 * n_frame["optimal_rate"],
            marker="o",
            linewidth=1.5,
            markersize=COMPARISON_MARKER_SIZE,
            label=label,
        )
        axes[1].plot(
            n_frame["layer_ratio"],
            100.0 * n_frame["median_gap"],
            marker="o",
            linewidth=1.5,
            markersize=COMPARISON_MARKER_SIZE,
            label=label,
        )
        axes[2].plot(
            n_frame["layer_ratio"],
            100.0 * n_frame["p95_gap"],
            marker="o",
            linewidth=1.5,
            markersize=COMPARISON_MARKER_SIZE,
            label=label,
        )

    axes[0].set_xlabel("Fraction of restriction layers")
    axes[0].set_ylabel("Optimal tours found (%)")
    axes[0].set_ylim(-3.0, 103.0)
    axes[1].set_xlabel("Fraction of restriction layers")
    axes[1].set_ylabel("Median gap (%)")
    axes[2].set_xlabel("Fraction of restriction layers")
    axes[2].set_ylabel("95th percentile gap (%)")

    for ax in axes:
        _style_axis(ax)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, out_dir, "paper_layer_ablation")


def _save_figure(fig: Figure, out_dir: Path, stem: str) -> list[Path]:
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=PAPER_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _style_axis(ax: Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-style benchmark figures for LaTeX."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper_figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = make_paper_figures(args.results_dir, args.out_dir)
    print(f"Wrote {len(written)} paper figure files to {args.out_dir}")


if __name__ == "__main__":
    main()
