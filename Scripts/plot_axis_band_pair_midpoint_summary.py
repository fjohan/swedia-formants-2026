#!/usr/bin/env python3
"""Tripartite vowel-pair summary using an explicit geographic axis and central band."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-axis-band-summary-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from plot_midpoint_vowel_spaces import read_tokens, speaker_vowels
from plot_regional_pair_midpoint_summary import (
    ALIASES, GROUP_COLORS, centroids, draw, midpoint_rows, resource_provinces, write_csv,
)


DEFAULT_EXCLUDED_REGIONS = {"Gotland", "Åboland", "Nyland", "Österbotten", "Åland"}
GROUP_ORDER = ("North", "Central", "South")


def coordinate_lookup(speakers: list[dict]) -> dict[str, np.ndarray]:
    result = {}
    for row in speakers:
        result.setdefault(row["village"], []).append((row["geo_x"], row["geo_y"]))
    return {village: np.median(np.asarray(values, dtype=float), axis=0)
            for village, values in result.items()}


def axis_band_groups(speakers: list[dict], south_name: str, north_name: str,
                     central_names: list[str], margin: float) -> tuple[dict, list[dict], dict]:
    coordinates = coordinate_lookup(speakers)
    required = [south_name, north_name, *central_names]
    missing = [name for name in required if name not in coordinates]
    if missing:
        raise ValueError("Axis/reference villages absent after filtering: " + ", ".join(missing))
    south, north = coordinates[south_name], coordinates[north_name]
    vector = north - south
    length = float(np.linalg.norm(vector))
    unit = vector / length
    # Signed perpendicular coordinate is retained for checking east-west coverage.
    perpendicular = np.array([-unit[1], unit[0]])
    central_positions = [float((coordinates[name] - south) @ unit) for name in central_names]
    lower = min(central_positions) - margin
    upper = max(central_positions) + margin
    groups, rows = {}, []
    speaker_counts = {(row["village"], row["speaker"]) for row in speakers}
    for village, coordinate in coordinates.items():
        along = float((coordinate - south) @ unit)
        across = float((coordinate - south) @ perpendicular)
        group = "South" if along < lower else "North" if along > upper else "Central"
        groups[village] = group
        rows.append({
            "village": village, "group": group,
            "geo_x": coordinate[0], "geo_y": coordinate[1],
            "axis_position_from_south": along, "cross_axis_position": across,
            "n_speakers": sum(item[0] == village for item in speaker_counts),
        })
    metadata = {
        "south_endpoint": south_name, "south_x": float(south[0]), "south_y": float(south[1]),
        "north_endpoint": north_name, "north_x": float(north[0]), "north_y": float(north[1]),
        "axis_length": length, "axis_unit_x": float(unit[0]), "axis_unit_y": float(unit[1]),
        "central_reference_villages": central_names,
        "central_reference_axis_positions": central_positions,
        "central_margin_map_units": margin,
        "central_lower_axis_position": lower, "central_upper_axis_position": upper,
    }
    return groups, sorted(rows, key=lambda row: row["axis_position_from_south"], reverse=True), metadata


def plot_groups(path: Path, assignments: list[dict], metadata: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 9), constrained_layout=True)
    for group in GROUP_ORDER:
        selected = [row for row in assignments if row["group"] == group]
        ax.scatter([row["geo_x"] for row in selected], [row["geo_y"] for row in selected],
                   color=GROUP_COLORS[group], s=48, edgecolor="#333", linewidth=.4,
                   label=f"{group} ({len(selected)} villages)")
    south = np.array([metadata["south_x"], metadata["south_y"]])
    unit = np.array([metadata["axis_unit_x"], metadata["axis_unit_y"]])
    perpendicular = np.array([-unit[1], unit[0]])
    extent = max(abs(row["cross_axis_position"]) for row in assignments) + 35
    for position, linestyle in ((metadata["central_lower_axis_position"], "--"),
                                (metadata["central_upper_axis_position"], "--")):
        center = south + position * unit
        ends = np.vstack([center - extent * perpendicular, center + extent * perpendicular])
        ax.plot(ends[:, 0], ends[:, 1], color="#111", ls=linestyle, lw=1.4)
    north = np.array([metadata["north_x"], metadata["north_y"]])
    ax.annotate("", xy=north, xytext=south,
                arrowprops={"arrowstyle": "->", "color": "#111", "lw": 2.2})
    ax.annotate(metadata["south_endpoint"], south, xytext=(5, 5),
                textcoords="offset points", fontsize=8)
    ax.annotate(metadata["north_endpoint"], north, xytext=(5, 5),
                textcoords="offset points", fontsize=8)
    for name in metadata["central_reference_villages"]:
        row = next(row for row in assignments if row["village"] == name)
        ax.scatter(row["geo_x"], row["geo_y"], marker="*", s=140,
                   color="#f3c52b", edgecolor="#111", zorder=5)
        ax.annotate(name, (row["geo_x"], row["geo_y"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.invert_yaxis()
    ax.set(title="Axis-and-band tripartite geographic grouping",
           xlabel="geographic x", ylabel="geographic y")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.18); ax.legend()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--method", choices=("praat-fixed", "fasttrack", "pca"), default="pca")
    parser.add_argument("--bark", action="store_true")
    parser.add_argument("--south-endpoint", default="loderup")
    parser.add_argument("--north-endpoint", default="arjeplog",
                        help="Default uses Arjeplog because Arvidsjaur is absent from this corpus.")
    parser.add_argument("--central-villages", nargs="+", default=["tjallmo", "rimforsa"])
    parser.add_argument("--central-margin", type=float, default=35.0,
                        help="Extra map units on each side of the central villages' axis positions.")
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--include-finland-gotland", action="store_true")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.central_margin < 0:
        parser.error("--central-margin must be non-negative")

    tokens = read_tokens(args.input, args.method, args.include_incomplete, args.bark)
    tokens = [row for row in tokens if args.include_reference or row["village"] != "ref"]
    provinces = resource_provinces(args.resource)
    excluded_villages = sorted({
        row["village"] for row in tokens
        if not args.include_finland_gotland
        and provinces.get(ALIASES.get(row["village"], row["village"])) in DEFAULT_EXCLUDED_REGIONS
    })
    tokens = [row for row in tokens if row["village"] not in excluded_villages]
    speakers = speaker_vowels(tokens)
    groups, assignments, metadata = axis_band_groups(
        speakers, args.south_endpoint, args.north_endpoint,
        args.central_villages, args.central_margin,
    )
    anchors = centroids(speakers)
    regional = centroids(speakers, groups)
    rows = midpoint_rows(args.method, anchors, regional, GROUP_ORDER)
    counts = {group: sum(row["n_speakers"] for row in assignments if row["group"] == group)
              for group in GROUP_ORDER}

    space = "pca" if args.method == "pca" else "bark" if args.bark else "hz"
    output_label = args.method if args.method == "pca" else f"{args.method}_{space}"
    output = args.output_dir or Path(f"Analyses/Axis_band_pair_midpoints_{output_label}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "regional_group_assignments.csv", assignments)
    write_csv(output / "regional_pair_midpoints.csv", rows)
    write_csv(output / "excluded_villages.csv", [
        {"village": village,
         "region": provinces[ALIASES.get(village, village)]}
        for village in excluded_villages
    ])
    plot_groups(output / "geographic_group_definition.png", assignments, metadata)
    draw(output / "regional_pair_midpoint_summary.png", args.method, space,
         anchors, rows, GROUP_ORDER, GROUP_COLORS)
    settings = {
        "input": str(args.input), "method": args.method, "space": space,
        **metadata, "group_order": GROUP_ORDER, "group_speaker_counts": counts,
        "group_village_counts": {
            group: sum(row["group"] == group for row in assignments) for group in GROUP_ORDER
        },
        "include_incomplete": args.include_incomplete,
        "excluded_regions": [] if args.include_finland_gotland else sorted(DEFAULT_EXCLUDED_REGIONS),
        "excluded_villages": excluded_villages,
        "anchors": "corpus-wide medians of speaker-vowel medians after geographic exclusions",
        "arrow_rule": "strict monotonic North-Central-South ordering",
    }
    (output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Groups: {counts}; central axis band "
          f"{metadata['central_lower_axis_position']:.1f}--"
          f"{metadata['central_upper_axis_position']:.1f}; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
