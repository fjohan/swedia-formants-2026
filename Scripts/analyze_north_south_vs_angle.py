#!/usr/bin/env python3
"""Relate vowel-space ellipse angles to north-south geographic position."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-angle-geography-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from scipy.stats import pearsonr, spearmanr


REQUIRED = {
    "method", "grouping", "village", "speaker", "fit_basis",
    "geo_x", "geo_y", "ellipse_angle_from_horizontal_signed_deg",
    "axis_ratio", "orientation_reliable",
}


def read_rows(path: Path, include_reference: bool) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    available = set(rows[0]) if rows else set()
    if not rows or not REQUIRED.issubset(available):
        raise ValueError(f"{path} is empty or missing: {', '.join(sorted(REQUIRED - available))}")
    if not include_reference:
        rows = [row for row in rows if row["village"] != "ref"]
    return rows


def statistic(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    function = pearsonr if kind == "pearson" else spearmanr
    return float(function(x, y).statistic)


def clustered_permutation_p(rows: list[dict], kind: str, permutations: int, seed: int) -> float:
    """Shuffle geographic y among villages while retaining speaker clusters."""
    villages = sorted({row["village"] for row in rows})
    if len(villages) < 3 or permutations < 1:
        return math.nan
    village_y = {village: float(next(row["geo_y"] for row in rows if row["village"] == village))
                 for village in villages}
    angles = np.asarray([float(row["ellipse_angle_from_horizontal_signed_deg"]) for row in rows])
    observed_y = np.asarray([village_y[row["village"]] for row in rows])
    observed = abs(statistic(observed_y, angles, kind))
    original = np.asarray([village_y[village] for village in villages])
    indices = np.asarray([villages.index(row["village"]) for row in rows])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        permuted_y = rng.permutation(original)[indices]
        extreme += abs(statistic(permuted_y, angles, kind)) >= observed
    return (extreme + 1) / (permutations + 1)


def calculate_statistics(rows: list[dict], bases: list[str], permutations: int,
                         seed: int) -> list[dict]:
    output = []
    for basis in bases:
        selected = [row for row in rows if row["fit_basis"] == basis
                    and int(row["orientation_reliable"])]
        y = np.asarray([float(row["geo_y"]) for row in selected])
        angles = np.asarray([
            float(row["ellipse_angle_from_horizontal_signed_deg"]) for row in selected
        ])
        if len(selected) >= 3 and np.ptp(y) > 0 and np.ptp(angles) > 0:
            pearson = pearsonr(y, angles)
            spearman = spearmanr(y, angles)
            cluster_pearson = clustered_permutation_p(selected, "pearson", permutations, seed)
            cluster_spearman = clustered_permutation_p(selected, "spearman", permutations, seed + 1)
        else:
            pearson = spearman = None
            cluster_pearson = cluster_spearman = math.nan
        output.append({
            "method": rows[0]["method"], "grouping": rows[0]["grouping"],
            "fit_basis": basis, "n_reliable": len(selected),
            "n_villages": len({row["village"] for row in selected}),
            "pearson_r": float(pearson.statistic) if pearson else math.nan,
            "pearson_p": float(pearson.pvalue) if pearson else math.nan,
            "spearman_rho": float(spearman.statistic) if spearman else math.nan,
            "spearman_p": float(spearman.pvalue) if spearman else math.nan,
            "village_permutation_pearson_p": cluster_pearson,
            "village_permutation_spearman_p": cluster_spearman,
        })
    return output


def jittered_coordinates(rows: list[dict]) -> dict[int, tuple[float, float]]:
    by_village = defaultdict(list)
    for index, row in enumerate(rows):
        by_village[row["village"]].append((index, row))
    result = {}
    for values in by_village.values():
        for position, (index, row) in enumerate(values):
            if len(values) == 1:
                result[index] = (float(row["geo_x"]), float(row["geo_y"]))
            else:
                theta = 2 * math.pi * position / len(values)
                result[index] = (
                    float(row["geo_x"]) + 4 * math.cos(theta),
                    float(row["geo_y"]) + 4 * math.sin(theta),
                )
    return result


def plot_results(path: Path, rows: list[dict], statistics: list[dict], bases: list[str],
                 angle_limit: float) -> None:
    fig, axes = plt.subplots(2, len(bases), figsize=(7 * len(bases), 12), squeeze=False)
    norm = Normalize(-angle_limit, angle_limit)
    cmap = "coolwarm_r"  # negative = warm red; positive = cool blue
    for column, basis in enumerate(bases):
        all_basis = [row for row in rows if row["fit_basis"] == basis]
        reliable = [row for row in all_basis if int(row["orientation_reliable"])]
        unstable = [row for row in all_basis if not int(row["orientation_reliable"])]
        map_ax, scatter_ax = axes[0, column], axes[1, column]

        plotted = jittered_coordinates(all_basis)
        reliable_indices = [index for index, row in enumerate(all_basis)
                            if int(row["orientation_reliable"])]
        map_ax.scatter(
            [plotted[index][0] for index in reliable_indices],
            [plotted[index][1] for index in reliable_indices],
            c=[float(all_basis[index]["ellipse_angle_from_horizontal_signed_deg"])
               for index in reliable_indices],
            cmap=cmap, norm=norm, s=75, edgecolor="#222", linewidth=.5, zorder=3,
        )
        unstable_indices = [index for index, row in enumerate(all_basis)
                            if not int(row["orientation_reliable"])]
        if unstable_indices:
            ux = [plotted[index][0] for index in unstable_indices]
            uy = [plotted[index][1] for index in unstable_indices]
            map_ax.scatter(ux, uy, color="#bbbbbb", s=70, edgecolor="#444", zorder=3)
            map_ax.scatter(ux, uy, color="#333", marker="x", s=32, zorder=4)
        for village in sorted({row["village"] for row in all_basis}):
            row = next(item for item in all_basis if item["village"] == village)
            map_ax.text(float(row["geo_x"]) + 5, float(row["geo_y"]) - 3,
                        village, fontsize=6)
        map_ax.set_ylim(max(float(row["geo_y"]) for row in all_basis) + 20,
                        min(float(row["geo_y"]) for row in all_basis) - 20)
        map_ax.set_aspect("equal", adjustable="datalim")
        map_ax.set(xlabel="geographic x", ylabel="geographic y",
                   title=f"{basis.capitalize()} ellipse angles")
        map_ax.grid(alpha=.2)

        y = np.asarray([float(row["geo_y"]) for row in reliable])
        angles = np.asarray([
            float(row["ellipse_angle_from_horizontal_signed_deg"]) for row in reliable
        ])
        scatter_ax.scatter(y, angles, color="#4c78a8", alpha=.65, s=28)
        if len(y) >= 2 and np.ptp(y) > 0:
            coefficients = np.polyfit(y, angles, 1)
            line_x = np.linspace(y.min(), y.max(), 100)
            scatter_ax.plot(line_x, np.polyval(coefficients, line_x), color="#d1495b", lw=2)
        stats = next(row for row in statistics if row["fit_basis"] == basis)
        p_label = ("village-permutation p" if rows[0]["grouping"] == "speaker"
                   else "permutation p")
        scatter_ax.text(
            .03, .97,
            f'n={stats["n_reliable"]} / {stats["n_villages"]} villages\n'
            f'Pearson r={stats["pearson_r"]:+.3f}, p={stats["pearson_p"]:.3g}\n'
            f'Spearman ρ={stats["spearman_rho"]:+.3f}, p={stats["spearman_p"]:.3g}\n'
            f'{p_label}: {stats["village_permutation_pearson_p"]:.3g} (Pearson)',
            transform=scatter_ax.transAxes, va="top",
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#aaa"},
        )
        scatter_ax.set(xlabel="geographic y (north → south)",
                       ylabel="signed angle from horizontal (degrees)")
        scatter_ax.grid(alpha=.2)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[0, :], shrink=.75,
                 label="Signed angle from horizontal (degrees)")
    fig.suptitle(
        f'{rows[0]["method"]} ellipse orientation versus north–south position '
        f'({rows[0]["grouping"]})'
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("Analyses/Midpoint_vowel_spaces_praat-fixed_village/ellipse_angles.csv"),
    )
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to the input file's directory.")
    parser.add_argument("--fit-basis", choices=("vowels", "midpoints", "both"), default="both")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--angle-limit", type=float, default=50.0)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.permutations < 0:
        parser.error("--permutations cannot be negative")
    bases = ["vowels", "midpoints"] if args.fit_basis == "both" else [args.fit_basis]
    rows = read_rows(args.input, args.include_reference)
    rows = [row for row in rows if row["fit_basis"] in bases]
    if not rows:
        raise SystemExit("No matching ellipse rows")
    output = args.output_dir or args.input.parent
    output.mkdir(parents=True, exist_ok=True)
    statistics = calculate_statistics(rows, bases, args.permutations, args.seed)
    write_csv(output / "north_south_angle_correlations.csv", statistics)
    plot_results(output / "north_south_vs_angle.png", rows, statistics, bases,
                 args.angle_limit)
    for row in statistics:
        print(
            f'{row["fit_basis"]}: n={row["n_reliable"]}, '
            f'r={row["pearson_r"]:+.3f}, rho={row["spearman_rho"]:+.3f}, '
            f'village-permutation p={row["village_permutation_pearson_p"]:.4g}'
        )
    print(f"Wrote results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
