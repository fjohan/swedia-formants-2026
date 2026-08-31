#!/usr/bin/env python3
"""Plot SweDia inventory pass status at resource.txt village coordinates."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import tempfile
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


STATUS_COLORS = {
    "pass": "#2ca02c",
    "partial pass": "#ffcc00",
    "non-pass": "#d62728",
}


def read_resource(path: Path) -> list[dict[str, str | int]]:
    text = path.read_bytes().decode("latin-1")
    rows = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        fields = shlex.split(line[1:-1])
        if len(fields) < 6:
            raise ValueError(f"Malformed resource row at {path}:{line_no}: {raw!r}")
        rows.append({
            "stem": fields[0],
            "abbr": fields[1],
            "name": fields[2],
            "province": fields[3],
            "x": int(fields[4]),
            "y": int(fields[5]),
        })
    return rows


def read_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["abbr"]: row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--inventory", type=Path, default=Path("sweDia_inventory_all_compact.tsv"))
    parser.add_argument("--output", type=Path, default=Path("sweDia_inventory_all_map.png"))
    parser.add_argument("--background", type=Path, help="Optional map image to plot under the points.")
    parser.add_argument("--labels", action="store_true", help="Draw village abbreviations next to points.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    resources = read_resource(args.resource)
    inventory = read_inventory(args.inventory)

    fig, ax = plt.subplots(figsize=(7.2, 10.0))
    if args.background:
        image = plt.imread(args.background)
        height, width = image.shape[:2]
        ax.imshow(image, extent=[0, width, height, 0])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    else:
        xs = [int(row["x"]) for row in resources]
        ys = [int(row["y"]) for row in resources]
        padding = 18
        ax.set_xlim(min(xs) - padding, max(xs) + padding)
        ax.set_ylim(max(ys) + padding, min(ys) - padding)
        ax.set_facecolor("#f7f7f2")

    missing = []
    for row in resources:
        abbr = str(row["abbr"])
        status = inventory.get(abbr, {}).get("status")
        if status not in STATUS_COLORS:
            missing.append(abbr)
            status = "non-pass"
        ax.scatter(
            int(row["x"]),
            int(row["y"]),
            s=42,
            color=STATUS_COLORS[status],
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        if args.labels:
            ax.text(int(row["x"]) + 3, int(row["y"]) - 3, abbr, fontsize=6, zorder=4)

    legend = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color,
               markeredgecolor="black", markersize=7)
        for label, color in STATUS_COLORS.items()
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True)
    ax.set_title("SweDia inventory status")
    ax.set_xlabel("resource x")
    ax.set_ylabel("resource y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Wrote {args.output}")
    if missing:
        print(f"Missing inventory status for {len(missing)} villages: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
