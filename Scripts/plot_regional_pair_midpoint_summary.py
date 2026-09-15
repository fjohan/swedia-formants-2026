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
import numpy as np

from plot_midpoint_vowel_spaces import COLORS as VOWEL_COLORS
from plot_midpoint_vowel_spaces import PAIRS, VOWELS, read_tokens, speaker_vowels


GROUP_COLORS = {"North": "#c90000", "Central": "#111111", "South": "#315a9b"}
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
         group_order: tuple[str, ...], colors: dict[str, str]) -> None:
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
    parser.add_argument("--grouping", choices=("equal", "lands"), default="equal")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--exclude-villages", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    tokens = read_tokens(args.input, args.method, args.include_incomplete, args.bark)
    excluded = set(args.exclude_villages)
    if not args.include_reference:
        excluded.add("ref")
    tokens = [row for row in tokens if row["village"] not in excluded]
    speakers = speaker_vowels(tokens)
    if args.grouping == "equal":
        groups, sizes = equal_speaker_groups(speakers)
        group_order, colors = ("North", "Central", "South"), GROUP_COLORS
    else:
        groups, sizes = land_groups(speakers, resource_provinces(args.resource))
        group_order, colors = ("Norrland", "Svealand", "Götaland"), LAND_COLORS
    anchors = centroids(speakers)
    regional = centroids(speakers, groups)
    rows = midpoint_rows(args.method, anchors, regional, group_order)

    space = "pca" if args.method == "pca" else "bark" if args.bark else "hz"
    output = args.output_dir or Path(
        f"Analyses/Regional_pair_midpoints_{args.method}_{space}_{args.grouping}"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "regional_pair_midpoints.csv", rows)
    unique_speakers = {(row["village"], row["speaker"]) for row in speakers}
    assignments = [{
        "village": village, "group": group,
        "geo_y": next(row["geo_y"] for row in speakers if row["village"] == village),
        "n_speakers": sum(item[0] == village for item in unique_speakers),
    } for village, group in sorted(groups.items(),
                                   key=lambda item: next(row["geo_y"] for row in speakers
                                                         if row["village"] == item[0]))]
    write_csv(output / "regional_group_assignments.csv", assignments)
    draw(output / "regional_pair_midpoint_summary.png", args.method, space,
         anchors, rows, group_order, colors)
    (output / "settings.json").write_text(json.dumps({
        "input": str(args.input), "method": args.method, "space": space,
        "grouping": args.grouping, "group_order": group_order,
        "group_speaker_counts": dict(zip(group_order, sizes)),
        "include_incomplete": args.include_incomplete,
        "excluded_villages": sorted(excluded),
        "anchors": "corpus-wide medians of speaker-vowel medians",
        "arrow_rule": "strict monotonic ordering across the three geographic groups",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Groups contain {sizes} speakers; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
