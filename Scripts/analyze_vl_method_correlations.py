#!/usr/bin/env python3
"""Calculate and plot agreement among Praat, FastTrack, and PCA vector lengths."""

from __future__ import annotations

import argparse
import csv
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-vl-correlations-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


METHODS = {
    "Praat fixed": "VL",
    "FastTrack": "ft_VL",
    "Spectral PCA": "pca_VL",
}


def read_rows(path: Path, complete_only: bool) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"vowel", *METHODS.values()}
    if not rows or not required.issubset(rows[0]):
        missing = required - (set(rows[0]) if rows else set())
        raise ValueError(f"{path} is empty or missing columns: {', '.join(sorted(missing))}")
    if complete_only:
        if "complete" not in rows[0]:
            raise ValueError("--complete-only requires a 'complete' column")
        rows = [row for row in rows if row["complete"] == "1"]
    return rows


def paired_values(rows: list[dict], first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = []
    for row in rows:
        try:
            x, y = float(row[first]), float(row[second])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))
    if not pairs:
        return np.empty(0), np.empty(0)
    values = np.asarray(pairs, dtype=float)
    return values[:, 0], values[:, 1]


def correlation_row(rows: list[dict], group: str, first_name: str, second_name: str) -> dict:
    first, second = METHODS[first_name], METHODS[second_name]
    x, y = paired_values(rows, first, second)
    result = {
        "group": group,
        "method_1": first_name,
        "method_2": second_name,
        "column_1": first,
        "column_2": second,
        "n": len(x),
        "pearson_r": "",
        "pearson_p": "",
        "spearman_rho": "",
        "spearman_p": "",
    }
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        result.update({
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pairwise(path: Path, rows: list[dict], statistics: list[dict], percentile: float) -> None:
    pairs = list(combinations(METHODS, 2))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (first_name, second_name) in zip(axes, pairs):
        x, y = paired_values(rows, METHODS[first_name], METHODS[second_name])
        x_limit = np.percentile(x, percentile)
        y_limit = np.percentile(y, percentile)
        visible = (x <= x_limit) & (y <= y_limit)
        image = ax.hexbin(
            x[visible], y[visible], gridsize=55, mincnt=1,
            bins="log", cmap="viridis",
        )
        stats = next(
            row for row in statistics
            if row["group"] == "all" and row["method_1"] == first_name
            and row["method_2"] == second_name
        )
        ax.text(
            0.03, 0.97,
            f'n={stats["n"]:,}\n'
            f'Pearson r={stats["pearson_r"]:+.3f}\n'
            f'Spearman ρ={stats["spearman_rho"]:+.3f}',
            transform=ax.transAxes, va="top", ha="left",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
        )
        ax.set(
            xlabel=f"{first_name} VL" + (" (Hz)" if first_name != "Spectral PCA" else " (PC-score dB)"),
            ylabel=f"{second_name} VL" + (" (Hz)" if second_name != "Spectral PCA" else " (PC-score dB)"),
            xlim=(0, x_limit), ylim=(0, y_limit),
        )
        ax.grid(alpha=0.15)
        fig.colorbar(image, ax=ax, pad=0.01, label="log₁₀ token count")
    fig.suptitle(
        "Token-level vowel trajectory length agreement\n"
        f"Axes shown through the {percentile:g}th percentile; correlations use all finite values"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("Analyses/VL_method_correlations"),
    )
    parser.add_argument("--complete-only", action="store_true")
    parser.add_argument(
        "--plot-percentile", type=float, default=99.5,
        help="Upper percentile displayed on plot axes; statistics always use all values.",
    )
    args = parser.parse_args()
    if not 90 <= args.plot_percentile <= 100:
        parser.error("--plot-percentile must be between 90 and 100")

    rows = read_rows(args.input, args.complete_only)
    pairs = list(combinations(METHODS, 2))
    statistics = [correlation_row(rows, "all", *pair) for pair in pairs]
    for vowel in sorted({row["vowel"] for row in rows}):
        subset = [row for row in rows if row["vowel"] == vowel]
        statistics.extend(correlation_row(subset, vowel, *pair) for pair in pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "vl_method_correlations.csv", statistics)
    plot_pairwise(
        args.output_dir / "vl_method_correlations.png",
        rows, statistics, args.plot_percentile,
    )
    for row in statistics[:3]:
        print(
            f'{row["method_1"]} vs {row["method_2"]}: n={row["n"]}, '
            f'r={row["pearson_r"]:.3f}, rho={row["spearman_rho"]:.3f}'
        )
    print(f"Wrote results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
