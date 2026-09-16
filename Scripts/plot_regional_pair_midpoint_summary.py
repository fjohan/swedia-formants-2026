#!/usr/bin/env python3
"""Summarize vowel-pair midpoints across three geographic regions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-regional-summary-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from plot_midpoint_vowel_spaces import COLORS as VOWEL_COLORS
from plot_midpoint_vowel_spaces import PAIRS, VOWELS, read_tokens, speaker_vowels


GROUP_COLORS = {"North": "#c90000", "Central": "#111111", "South": "#315a9b"}
AXIS_GROUP_COLORS = {"North": "#315fa8", "Central": "#999999", "South": "#c43c39"}
DEFAULT_EXCLUDED_REGIONS = {"Gotland", "Åboland", "Nyland", "Österbotten", "Åland"}
LAND_COLORS = {"Norrland": "#c90000", "Svealand": "#111111", "Götaland": "#315a9b"}
ALIASES = {
    "n_rorum": "nrorum", "pitea": "pite", "s_finnskoga": "sodrafinnskoga",
    "s_mellosa": "stmellosa", "st_anna": "stanna", "v_vingaker": "vingaker",
    "ref": "reference",
}
LAND_PROVINCES = {
    "Götaland": {"Blekinge", "Bohuslän", "Dalsland", "Gotland", "Halland", "Skåne",
                 "Småland", "Västergötland", "Öland", "Östergötland"},
    "Svealand": {"Dalarna", "Närke", "Södermanland", "Uppland", "Värmland", "Västmanland"},
    "Norrland": {"Gästrikland", "Hälsingland", "Härjedalen", "Jämtland", "Lappland",
                 "Medelpad", "Norrbotten", "Västerbotten", "Ångermanland"},
}


def confidence_ellipse(ax, points: np.ndarray, color: str) -> None:
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    order = np.argsort(values)[::-1]; values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    scale = np.sqrt(5.991)
    ax.add_patch(Ellipse(center, 2*scale*np.sqrt(max(values[0], 0)),
                         2*scale*np.sqrt(max(values[1], 0)), angle=angle,
                         facecolor=color, edgecolor=color, alpha=.12, lw=1.2, zorder=2))


def axis_band_groups(speakers: list[dict], south_name: str, north_name: str,
                     central_names: list[str], margin: float) -> tuple[dict, dict]:
    grouped = defaultdict(list)
    for row in speakers:
        grouped[row["village"]].append((row["geo_x"], row["geo_y"]))
    coordinates = {name: np.median(np.asarray(values), axis=0)
                   for name, values in grouped.items()}
    missing = [name for name in [south_name, north_name, *central_names]
               if name not in coordinates]
    if missing:
        raise ValueError("Geographic reference villages absent: " + ", ".join(missing))
    south, north = coordinates[south_name], coordinates[north_name]
    unit = (north - south) / np.linalg.norm(north - south)
    positions = {name: float((point-south) @ unit) for name, point in coordinates.items()}
    central = [positions[name] for name in central_names]
    lower, upper = min(central)-margin, max(central)+margin
    groups = {name: "South" if value < lower else "North" if value > upper else "Central"
              for name, value in positions.items()}
    return groups, {"positions": positions, "lower": lower, "upper": upper,
                    "south": south.tolist(), "north": north.tolist()}


def resource_provinces(path: Path) -> dict[str, str]:
    output = {}
    for raw in path.read_bytes().decode("latin-1").splitlines():
        line = raw.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        fields = shlex.split(line[1:-1])
        if len(fields) >= 6:
            output[fields[0]] = fields[3]
    return output


def land_for(province: str) -> str | None:
    return next((land for land, provinces in LAND_PROVINCES.items()
                 if province in provinces), None)


def equal_speaker_groups(speakers: list[dict]) -> tuple[dict[str, str], list[int]]:
    counts = defaultdict(set)
    y_values = {}
    for row in speakers:
        counts[row["village"]].add(row["speaker"])
        y_values[row["village"]] = row["geo_y"]
    villages = sorted(counts, key=lambda village: y_values[village])
    best = None
    for first_cut in range(1, len(villages) - 1):
        for second_cut in range(first_cut + 1, len(villages)):
            sizes = [
                sum(len(counts[village]) for village in subset)
                for subset in (villages[:first_cut], villages[first_cut:second_cut], villages[second_cut:])
            ]
            score = sum((size - sum(sizes) / 3) ** 2 for size in sizes)
            if best is None or score < best[0]:
                best = score, first_cut, second_cut, sizes
    _, first_cut, second_cut, sizes = best
    groups = {}
    for label, subset in zip(
        ("North", "Central", "South"),
        (villages[:first_cut], villages[first_cut:second_cut], villages[second_cut:]),
    ):
        groups.update({village: label for village in subset})
    return groups, sizes


def land_groups(speakers: list[dict], provinces: dict[str, str]) -> tuple[dict[str, str], list[int]]:
    groups = {}
    for row in speakers:
        key = ALIASES.get(row["village"], row["village"])
        land = land_for(provinces.get(key, ""))
        if land:
            groups[row["village"]] = land
    unique_speakers = {(row["village"], row["speaker"]) for row in speakers}
    order = ("Norrland", "Svealand", "Götaland")
    sizes = [sum(groups.get(village) == group for village, _ in unique_speakers) for group in order]
    return groups, sizes


def centroids(speakers: list[dict], groups: dict[str, str] | None = None) -> dict:
    values = defaultdict(list)
    for row in speakers:
        if groups is not None and row["village"] not in groups:
            continue
        group = groups[row["village"]] if groups is not None else "Corpus"
        values[group, row["vowel"]].append((row["display_x"], row["display_y"]))
    return {key: np.median(np.asarray(points, dtype=float), axis=0)
            for key, points in values.items()}


def village_balanced_anchors(speakers: list[dict]) -> dict:
    by_village = defaultdict(list)
    for row in speakers:
        by_village[row["village"], row["vowel"]].append(
            (row["display_x"], row["display_y"])
        )
    village_positions = defaultdict(list)
    for (_, vowel), values in by_village.items():
        village_positions[vowel].append(np.median(np.asarray(values), axis=0))
    return {("Corpus", vowel): np.median(np.asarray(values), axis=0)
            for vowel, values in village_positions.items()}


def paired_village_midpoints(speakers: list[dict], groups: dict[str, str]) -> list[dict]:
    by_speaker = defaultdict(dict)
    for row in speakers:
        by_speaker[row["village"], row["speaker"]][row["vowel"]] = row
    speaker_pairs = []
    for (village, speaker), vowels in sorted(by_speaker.items()):
        for first, second in PAIRS:
            if first not in vowels or second not in vowels or village not in groups:
                continue
            a, b = vowels[first], vowels[second]
            speaker_pairs.append({
                "village": village, "speaker": speaker, "group": groups[village],
                "pair": f"{first}->{second}", "first_vowel": first,
                "second_vowel": second,
                "midpoint_x": (a["display_x"] + b["display_x"]) / 2,
                "midpoint_y": (a["display_y"] + b["display_y"]) / 2,
                "pair_distance": float(np.hypot(b["display_x"] - a["display_x"],
                                                b["display_y"] - a["display_y"])),
            })
    grouped = defaultdict(list)
    for row in speaker_pairs:
        grouped[row["village"], row["pair"]].append(row)
    output = []
    for (village, pair), values in sorted(grouped.items()):
        output.append({
            "village": village, "group": values[0]["group"], "pair": pair,
            "first_vowel": values[0]["first_vowel"],
            "second_vowel": values[0]["second_vowel"],
            "n_speakers": len(values),
            "midpoint_x": float(np.median([row["midpoint_x"] for row in values])),
            "midpoint_y": float(np.median([row["midpoint_y"] for row in values])),
            "pair_distance": float(np.median([row["pair_distance"] for row in values])),
        })
    return output


def paired_midpoint_rows(method: str, anchors: dict, village_pairs: list[dict],
                         group_order: tuple[str, ...], bootstrap: int,
                         seed: int) -> tuple[list[dict], dict]:
    output, clouds = [], {}
    for pair_index, (first, second) in enumerate(PAIRS):
        anchor_first, anchor_second = anchors["Corpus", first], anchors["Corpus", second]
        vector = anchor_second - anchor_first
        length = float(np.linalg.norm(vector)); unit = vector / length
        normal = np.array([-unit[1], unit[0]])
        anchor_midpoint = (anchor_first + anchor_second) / 2
        pending, scalars = [], {}
        for group_index, group in enumerate(group_order):
            values = [row for row in village_pairs
                      if row["pair"] == f"{first}->{second}" and row["group"] == group]
            if not values:
                continue
            points = np.asarray([[row["midpoint_x"], row["midpoint_y"]] for row in values])
            true_midpoint = np.median(points, axis=0)
            delta = true_midpoint - anchor_midpoint
            scalar, perpendicular = float(delta @ unit), float(delta @ normal)
            projected = anchor_midpoint + scalar * unit
            scalars[group] = scalar
            rng = np.random.default_rng(seed + pair_index * 10 + group_index)
            cloud = np.asarray([
                np.median(points[rng.integers(0, len(points), len(points))], axis=0)
                for _ in range(bootstrap)
            ])
            clouds[(f"{first}->{second}", group)] = cloud
            pending.append({
                "method": method, "pair": f"{first}->{second}", "group": group,
                "first_vowel": first, "second_vowel": second,
                "n_villages": len(values),
                "n_speakers": sum(row["n_speakers"] for row in values),
                "anchor_pair_length": length,
                "midpoint_shift_along_axis": scalar,
                "midpoint_shift_perpendicular": perpendicular,
                "true_midpoint_x": float(true_midpoint[0]),
                "true_midpoint_y": float(true_midpoint[1]),
                "plot_x": float(projected[0]), "plot_y": float(projected[1]),
                "median_pair_distance": float(np.median([row["pair_distance"] for row in values])),
            })
        monotonic = len(scalars) == 3 and (
            scalars[group_order[0]] < scalars[group_order[1]] < scalars[group_order[2]]
            or scalars[group_order[0]] > scalars[group_order[1]] > scalars[group_order[2]]
        )
        for row in pending:
            row["monotonic_north_central_south"] = int(monotonic)
        output.extend(pending)
    return output, clouds


def midpoint_rows(method: str, anchors: dict, regional: dict, group_order: tuple[str, ...]) -> list[dict]:
    output = []
    for first, second in PAIRS:
        anchor_first, anchor_second = anchors["Corpus", first], anchors["Corpus", second]
        vector = anchor_second - anchor_first
        length = float(np.linalg.norm(vector))
        unit = vector / length
        anchor_midpoint = (anchor_first + anchor_second) / 2
        scalars = {}
        pending = []
        for group in group_order:
            if (group, first) not in regional or (group, second) not in regional:
                continue
            observed = (regional[group, first] + regional[group, second]) / 2
            scalar = float((observed - anchor_midpoint) @ unit)
            scalars[group] = scalar
            projected = anchor_midpoint + scalar * unit
            pending.append({
                "method": method, "pair": f"{first}->{second}", "group": group,
                "first_vowel": first, "second_vowel": second,
                "anchor_pair_length": length, "midpoint_shift_along_axis": scalar,
                "plot_x": float(projected[0]), "plot_y": float(projected[1]),
            })
        monotonic = len(scalars) == 3 and (
            scalars[group_order[0]] < scalars[group_order[1]] < scalars[group_order[2]]
            or scalars[group_order[0]] > scalars[group_order[1]] > scalars[group_order[2]]
        )
        for row in pending:
            row["monotonic_north_central_south"] = int(monotonic)
        output.extend(pending)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(path: Path, method: str, space: str, anchors: dict, rows: list[dict],
         group_order: tuple[str, ...], colors: dict[str, str],
         clouds: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for first, second in PAIRS:
        a, b = anchors["Corpus", first], anchors["Corpus", second]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#aeb6b8", lw=3, alpha=.65)
    for vowel in VOWELS:
        point = anchors["Corpus", vowel]
        ax.scatter(*point, facecolor="white", edgecolor=VOWEL_COLORS[vowel], s=48, zorder=4)
        ax.annotate(f"/{vowel}/", point, xytext=(5, 5), textcoords="offset points",
                    color="#555", fontsize=10, fontweight="bold")
    for first, second in PAIRS:
        pair = f"{first}->{second}"
        selected = [row for row in rows if row["pair"] == pair]
        for row in selected:
            if "true_midpoint_x" in row:
                cloud = (clouds or {}).get((pair, row["group"]))
                if cloud is not None:
                    confidence_ellipse(ax, cloud, colors[row["group"]])
                ax.plot([row["true_midpoint_x"], row["plot_x"]],
                        [row["true_midpoint_y"], row["plot_y"]],
                        color=colors[row["group"]], ls=":", lw=1.1, alpha=.75)
                ax.scatter(row["true_midpoint_x"], row["true_midpoint_y"],
                           facecolor="white", edgecolor=colors[row["group"]],
                           s=52, linewidth=1.5, zorder=4)
            ax.scatter(row["plot_x"], row["plot_y"], color=colors[row["group"]],
                       s=68, zorder=5)
        if selected and selected[0]["monotonic_north_central_south"]:
            north = next(row for row in selected if row["group"] == group_order[0])
            south = next(row for row in selected if row["group"] == group_order[2])
            ax.annotate("", xy=(south["plot_x"], south["plot_y"]),
                        xytext=(north["plot_x"], north["plot_y"]),
                        arrowprops={"arrowstyle": "->", "color": "#222", "lw": 1.5,
                                    "shrinkA": 5, "shrinkB": 5})
    for group in group_order:
        ax.scatter([], [], color=colors[group], s=68, label=group)
    if rows and "true_midpoint_x" in rows[0]:
        ax.scatter([], [], facecolor="white", edgecolor="#666", s=52,
                   label="true 2D midpoint")
        ax.plot([], [], color="#666", ls=":", label="projection onto pair axis")
    if method == "pca":
        xlabel, ylabel = "−PC1 (score dB)", "−PC2 (score dB)"
    else:
        unit = "Bark" if space == "bark" else "Hz"
        xlabel, ylabel = f"−F2 ({unit})", f"−F1 ({unit})"
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=f"Corpus-anchored regional vowel-pair midpoints: {method}")
    ax.grid(alpha=.18); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", framealpha=.9)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--method", choices=("praat-fixed", "fasttrack", "pca"),
                        default="praat-fixed")
    parser.add_argument("--bark", action="store_true",
                        help="Use Bark F1/F2 for Praat or FastTrack; PCA is unchanged.")
    parser.add_argument("--grouping", choices=("axis-band", "equal", "lands"),
                        default="axis-band")
    parser.add_argument("--midpoint-aggregation",
                        choices=("paired-village", "legacy-centroids"),
                        default="paired-village")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--south-endpoint", default="loderup")
    parser.add_argument("--north-endpoint", default="arjeplog")
    parser.add_argument("--central-villages", nargs="+", default=["tjallmo", "rimforsa"])
    parser.add_argument("--central-margin", type=float, default=35.0)
    parser.add_argument("--include-finland-gotland", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--exclude-villages", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.central_margin < 0:
        parser.error("--central-margin must be non-negative")
    if args.bootstrap < 20:
        parser.error("--bootstrap must be at least 20")

    tokens = read_tokens(args.input, args.method, args.include_incomplete, args.bark)
    excluded = set(args.exclude_villages)
    if not args.include_reference:
        excluded.add("ref")
    provinces = resource_provinces(args.resource)
    if not args.include_finland_gotland:
        excluded.update({
            row["village"] for row in tokens
            if provinces.get(ALIASES.get(row["village"], row["village"]))
            in DEFAULT_EXCLUDED_REGIONS
        })
    tokens = [row for row in tokens if row["village"] not in excluded]
    speakers = speaker_vowels(tokens)
    axis_metadata = None
    if args.grouping == "axis-band":
        groups, axis_metadata = axis_band_groups(
            speakers, args.south_endpoint, args.north_endpoint,
            args.central_villages, args.central_margin,
        )
        group_order, colors = ("North", "Central", "South"), AXIS_GROUP_COLORS
        unique_speakers = {(row["village"], row["speaker"]) for row in speakers}
        sizes = [sum(groups.get(village) == group for village, _ in unique_speakers)
                 for group in group_order]
    elif args.grouping == "equal":
        groups, sizes = equal_speaker_groups(speakers)
        group_order, colors = ("North", "Central", "South"), GROUP_COLORS
    else:
        groups, sizes = land_groups(speakers, resource_provinces(args.resource))
        group_order, colors = ("Norrland", "Svealand", "Götaland"), LAND_COLORS
    clouds = None
    if args.midpoint_aggregation == "paired-village":
        anchors = village_balanced_anchors(speakers)
        village_pairs = paired_village_midpoints(speakers, groups)
        rows, clouds = paired_midpoint_rows(
            args.method, anchors, village_pairs, group_order,
            args.bootstrap, args.seed,
        )
    else:
        anchors = centroids(speakers)
        regional = centroids(speakers, groups)
        rows = midpoint_rows(args.method, anchors, regional, group_order)

    space = "pca" if args.method == "pca" else "bark" if args.bark else "hz"
    output_label = args.method if args.method == "pca" else f"{args.method}_{space}"
    output = args.output_dir or Path(
        f"Analyses/Regional_pair_midpoints_{output_label}_{args.grouping}_"
        f"{args.midpoint_aggregation}"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "regional_pair_midpoints.csv", rows)
    unique_speakers = {(row["village"], row["speaker"]) for row in speakers}
    assignments = [{
        "village": village, "group": group,
        "geo_x": next(row["geo_x"] for row in speakers if row["village"] == village),
        "geo_y": next(row["geo_y"] for row in speakers if row["village"] == village),
        "axis_position": ((axis_metadata or {}).get("positions", {})).get(village, ""),
        "n_speakers": sum(item[0] == village for item in unique_speakers),
    } for village, group in sorted(groups.items(),
                                   key=lambda item: next(row["geo_y"] for row in speakers
                                                         if row["village"] == item[0]))]
    write_csv(output / "regional_group_assignments.csv", assignments)
    draw(output / "regional_pair_midpoint_summary.png", args.method, space,
         anchors, rows, group_order, colors, clouds)
    (output / "settings.json").write_text(json.dumps({
        "input": str(args.input), "method": args.method, "space": space,
        "grouping": args.grouping, "group_order": group_order,
        "midpoint_aggregation": args.midpoint_aggregation,
        "group_speaker_counts": dict(zip(group_order, sizes)),
        "include_incomplete": args.include_incomplete,
        "excluded_regions": ([] if args.include_finland_gotland
                             else sorted(DEFAULT_EXCLUDED_REGIONS)),
        "excluded_villages": sorted(excluded),
        "axis": axis_metadata,
        "axis_endpoints": [args.south_endpoint, args.north_endpoint],
        "central_villages": args.central_villages,
        "central_margin": args.central_margin,
        "anchors": ("village-balanced medians of speaker-vowel medians"
                    if args.midpoint_aggregation == "paired-village"
                    else "corpus-wide medians of speaker-vowel medians"),
        "bootstrap_iterations": args.bootstrap,
        "arrow_rule": "strict monotonic ordering across the three geographic groups",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Groups contain {sizes} speakers; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
