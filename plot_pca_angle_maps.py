#!/usr/bin/env python3
"""Map PCA vowel-space ellipse orientations using SweDia resource coordinates."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-pca-matplotlib")

import matplotlib.pyplot as plt

RESOURCE_ALIASES = {"n_rorum": "nrorum", "st_anna": "stanna", "v_vingaker": "vingaker"}


def read_coordinates(path: Path) -> dict[str, tuple[int, int, str]]:
    coordinates = {}
    for raw in path.read_bytes().decode("latin-1").splitlines():
        line = raw.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        fields = shlex.split(line[1:-1])
        if len(fields) >= 6:
            coordinates[fields[0]] = (int(fields[4]), int(fields[5]), fields[2])
    return coordinates


def read_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "village", "fit_basis", "ellipse_angle_from_vertical_signed_deg",
        "axis_ratio", "orientation_reliable",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} does not contain the current PCA ellipse metrics")
    return rows


def draw_map(ax, rows: list[dict], coordinates: dict, basis: str, angle_limit: float):
    selected = [row for row in rows if row["fit_basis"] == basis]
    coordinate = lambda row: coordinates[RESOURCE_ALIASES.get(row["village"], row["village"])]
    reliable = [row for row in selected if int(row["orientation_reliable"])]
    unstable = [row for row in selected if not int(row["orientation_reliable"])]
    scatter = ax.scatter(
        [coordinate(row)[0] for row in reliable],
        [coordinate(row)[1] for row in reliable],
        c=[float(row["ellipse_angle_from_vertical_signed_deg"]) for row in reliable],
        cmap="coolwarm", vmin=-angle_limit, vmax=angle_limit,
        s=105, edgecolor="#222222", linewidth=0.65, zorder=3,
    )
    if unstable:
        xs = [coordinate(row)[0] for row in unstable]
        ys = [coordinate(row)[1] for row in unstable]
        ax.scatter(xs, ys, color="#b8b8b8", marker="o", s=105, edgecolor="#444444", zorder=3)
        ax.scatter(xs, ys, color="#333333", marker="x", s=48, linewidth=1.4, zorder=4)
    for row in selected:
        x, y, display_name = coordinate(row)
        suffix = "*" if not int(row["orientation_reliable"]) else ""
        ax.text(x + 4, y - 3, display_name + suffix, fontsize=7, zorder=5)
    xs = [coordinate(row)[0] for row in selected]
    ys = [coordinate(row)[1] for row in selected]
    padding = 20
    ax.set_xlim(min(xs) - padding, max(xs) + padding)
    ax.set_ylim(max(ys) + padding, min(ys) - padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="resource x", ylabel="resource y", title=f"Ellipse orientation: {basis}")
    ax.grid(color="#dddddd", linewidth=0.5, zorder=0)
    return scatter, unstable


def save_single(path: Path, rows: list[dict], coordinates: dict, basis: str, angle_limit: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 10), constrained_layout=True)
    scatter, unstable = draw_map(ax, rows, coordinates, basis, angle_limit)
    fig.colorbar(scatter, ax=ax, label="Signed angle from vertical (degrees)", shrink=0.78)
    if unstable:
        ax.text(
            0.01, 0.01, "* neutral gray × = unstable orientation (axis ratio < 1.2)",
            transform=ax.transAxes, fontsize=7, color="#444444",
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_combined(path: Path, rows: list[dict], coordinates: dict, angle_limit: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), constrained_layout=True)
    scatter = None
    for ax, basis in zip(axes, ("vowels", "midpoints")):
        scatter, _ = draw_map(ax, rows, coordinates, basis, angle_limit)
    fig.colorbar(scatter, ax=axes, label="Signed angle from vertical (degrees)", shrink=0.72)
    fig.suptitle("PCA vowel-space orientation by SweDia location")
    fig.text(0.5, 0.015, "Neutral gray × = unstable orientation (axis ratio < 1.2)", ha="center", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("Analyses/BarkPCA_24_villages/village_pca_ellipses.csv"),
    )
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--output", type=Path, default=Path("Analyses/BarkPCA_24_villages"))
    parser.add_argument(
        "--angle-limit", type=float, default=50.0,
        help="Symmetric color limit in degrees; values outside it are clipped.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_metrics(args.metrics)
    coordinates = read_coordinates(args.resource)
    missing = sorted({
        row["village"] for row in rows
        if RESOURCE_ALIASES.get(row["village"], row["village"]) not in coordinates
    })
    if missing:
        raise ValueError("No resource coordinates for: " + ", ".join(missing))
    save_single(args.output / "village_angle_map_vowels.png", rows, coordinates, "vowels", args.angle_limit)
    save_single(args.output / "village_angle_map_midpoints.png", rows, coordinates, "midpoints", args.angle_limit)
    save_combined(args.output / "village_angle_maps.png", rows, coordinates, args.angle_limit)
    print(f"Mapped {len({row['village'] for row in rows})} villages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
