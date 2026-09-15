#!/usr/bin/env python3
"""Aggregate and optionally plot midpoint vowel spaces with ellipse angles."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-midpoint-spaces-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


VOWELS = ("uː", "oː", "ɑː", "æː", "eː", "yː", "ʉ̟ː", "øː")
PAIRS = (
    ("uː", "oː"), ("oː", "ɑː"), ("ɑː", "æː"),
    ("æː", "eː"), ("yː", "ʉ̟ː"), ("ʉ̟ː", "øː"),
)
METHOD_COLUMNS = {
    "praat-fixed": ("f2_50", "f1_50"),
    "fasttrack": ("ft_f2_50", "ft_f1_50"),
    "pca": ("pca_pc1_50", "pca_pc2_50"),
}
COLORS = dict(zip(VOWELS, plt.get_cmap("tab10").colors))


def finite_median(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def method_columns(method: str, bark: bool) -> tuple[str, str]:
    if bark and method == "praat-fixed":
        return "bark_f2_50", "bark_f1_50"
    if bark and method == "fasttrack":
        return "ft_bark_f2_50", "ft_bark_f1_50"
    return METHOD_COLUMNS[method]


def read_tokens(path: Path, method: str, include_incomplete: bool, bark: bool) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    x_column, y_column = method_columns(method, bark)
    required = {"village", "speaker", "vowel", "geo_x", "geo_y", x_column, y_column}
    if not include_incomplete:
        required.add("complete")
    available = set(rows[0]) if rows else set()
    if not rows or not required.issubset(available):
        raise ValueError(f"{path} is empty or missing: {', '.join(sorted(required - available))}")

    tokens = []
    for row in rows:
        if row["vowel"] not in VOWELS:
            continue
        if not include_incomplete and row["complete"] != "1":
            continue
        try:
            # Both axes are reversed to match a traditional vowel chart.
            x, y = -float(row[x_column]), -float(row[y_column])
            geo_x, geo_y = float(row["geo_x"]), float(row["geo_y"])
        except (TypeError, ValueError):
            continue
        if all(map(np.isfinite, (x, y, geo_x, geo_y))):
            tokens.append(row | {"display_x": x, "display_y": y,
                                 "geo_x_value": geo_x, "geo_y_value": geo_y})
    return tokens


def speaker_vowels(tokens: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in tokens:
        grouped[(row["village"], row["speaker"], row["vowel"])].append(row)
    result = []
    for (village, speaker, vowel), values in sorted(grouped.items()):
        result.append({
            "village": village, "speaker": speaker, "vowel": vowel,
            "n_tokens": len(values),
            "geo_x": finite_median(row["geo_x_value"] for row in values),
            "geo_y": finite_median(row["geo_y_value"] for row in values),
            "display_x": finite_median(row["display_x"] for row in values),
            "display_y": finite_median(row["display_y"] for row in values),
        })
    return result


def aggregate(speaker_rows: list[dict], grouping: str) -> list[dict]:
    if grouping == "speaker":
        return speaker_rows
    grouped = defaultdict(list)
    for row in speaker_rows:
        grouped[(row["village"], row["vowel"])].append(row)
    result = []
    for (village, vowel), values in sorted(grouped.items()):
        result.append({
            "village": village, "speaker": "", "vowel": vowel,
            "n_speakers": len(values),
            "n_tokens": sum(int(row["n_tokens"]) for row in values),
            "geo_x": finite_median(row["geo_x"] for row in values),
            "geo_y": finite_median(row["geo_y"] for row in values),
            # Median of speaker medians gives every speaker equal weight.
            "display_x": finite_median(row["display_x"] for row in values),
            "display_y": finite_median(row["display_y"] for row in values),
        })
    return result


def ellipse_metrics(points: np.ndarray) -> dict:
    centroid = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points - centroid, rowvar=False))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    vector = vectors[:, 0]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0
    signed = angle - 180.0 if angle > 90.0 else angle
    major = math.sqrt(max(float(values[0]), 0.0))
    minor = math.sqrt(max(float(values[1]), 0.0))
    ratio = major / minor if minor else math.inf
    return {
        "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]),
        "ellipse_angle_deg": angle,
        "ellipse_angle_from_horizontal_signed_deg": signed,
        "major_axis_sd": major, "minor_axis_sd": minor,
        "axis_ratio": ratio, "orientation_reliable": int(ratio >= 1.2),
    }


def ellipse_outline(metrics: dict, scale: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * math.pi, 180)
    points = np.vstack([
        metrics["major_axis_sd"] * scale * np.cos(theta),
        metrics["minor_axis_sd"] * scale * np.sin(theta),
    ])
    angle = math.radians(metrics["ellipse_angle_deg"])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    result = rotation @ points
    return result[0] + metrics["centroid_x"], result[1] + metrics["centroid_y"]


def group_key(row: dict, grouping: str) -> tuple[str, str]:
    return row["village"], row["speaker"] if grouping == "speaker" else ""


def analyze(rows: list[dict], method: str, grouping: str) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row, grouping)].append(row)
    midpoint_rows, angle_rows = [], []
    for (village, speaker), values in sorted(groups.items()):
        by_vowel = {row["vowel"]: row for row in values}
        points_by_basis = {"vowels": [], "midpoints": []}
        for vowel in VOWELS:
            if vowel in by_vowel:
                points_by_basis["vowels"].append(
                    (by_vowel[vowel]["display_x"], by_vowel[vowel]["display_y"])
                )
        for first, second in PAIRS:
            if first not in by_vowel or second not in by_vowel:
                continue
            start, end = by_vowel[first], by_vowel[second]
            x = (start["display_x"] + end["display_x"]) / 2
            y = (start["display_y"] + end["display_y"]) / 2
            points_by_basis["midpoints"].append((x, y))
            midpoint_rows.append({
                "method": method, "grouping": grouping,
                "village": village, "speaker": speaker,
                "pair": f"{first}->{second}", "first_vowel": first, "second_vowel": second,
                "display_x": x, "display_y": y,
                "geo_x": values[0]["geo_x"], "geo_y": values[0]["geo_y"],
            })
        missing = [vowel for vowel in VOWELS if vowel not in by_vowel]
        for basis, points in points_by_basis.items():
            if len(points) < 3:
                continue
            metrics = ellipse_metrics(np.asarray(points, dtype=float))
            angle_rows.append({
                "method": method, "grouping": grouping,
                "village": village, "speaker": speaker,
                "fit_basis": basis, "n_points": len(points),
                "n_vowels": len(by_vowel), "missing_vowels": " ".join(missing),
                "geo_x": values[0]["geo_x"], "geo_y": values[0]["geo_y"],
                **metrics,
            })
    return midpoint_rows, angle_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_group(ax, values: list[dict], metrics: dict[tuple, dict], method: str, grouping: str) -> None:
    by_vowel = {row["vowel"]: row for row in values}
    for vowel in VOWELS:
        if vowel not in by_vowel:
            continue
        row = by_vowel[vowel]
        ax.scatter(row["display_x"], row["display_y"], color=COLORS[vowel], s=42, zorder=4)
        ax.annotate(f"/{vowel}/", (row["display_x"], row["display_y"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    for first, second in PAIRS:
        if first not in by_vowel or second not in by_vowel:
            continue
        start, end = by_vowel[first], by_vowel[second]
        midpoint = ((start["display_x"] + end["display_x"]) / 2,
                    (start["display_y"] + end["display_y"]) / 2)
        ax.annotate("", xy=(end["display_x"], end["display_y"]),
                    xytext=(start["display_x"], start["display_y"]),
                    arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1,
                                "alpha": .55, "shrinkA": 6, "shrinkB": 6})
        ax.scatter(*midpoint, marker="x", color="#222", s=25, zorder=5)
    village, speaker = group_key(values[0], grouping)
    for basis, color, linestyle in (("vowels", "#4c78a8", "-"),
                                     ("midpoints", "#f58518", "--")):
        metric = metrics.get((village, speaker, basis))
        if not metric:
            continue
        x, y = ellipse_outline(metric)
        angle = metric["ellipse_angle_from_horizontal_signed_deg"]
        description = (f"{angle:+.1f}°" if metric["orientation_reliable"]
                       else f"unstable ({metric['axis_ratio']:.2f})")
        ax.plot(x, y, color=color, ls=linestyle, lw=1.7,
                label=f"{basis}: {description}")
    label = village if not speaker else f"{village}: {speaker}"
    ax.set_title(f"{label}\n{len(by_vowel)}/8 vowels", fontsize=9)
    ax.grid(alpha=.2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6, loc="best")


def make_plots(output: Path, rows: list[dict], angle_rows: list[dict], method: str,
               grouping: str, groups_per_page: int, bark: bool) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row, grouping)].append(row)
    keys = sorted(grouped)
    metrics = {(row["village"], row["speaker"], row["fit_basis"]): row for row in angle_rows}
    columns = 4
    for page, offset in enumerate(range(0, len(keys), groups_per_page), 1):
        page_keys = keys[offset:offset + groups_per_page]
        nrows = math.ceil(len(page_keys) / columns)
        fig, axes = plt.subplots(nrows, columns, figsize=(4 * columns, 3.6 * nrows),
                                 squeeze=False)
        for ax, key in zip(axes.flat, page_keys):
            draw_group(ax, grouped[key], metrics, method, grouping)
        for ax in list(axes.flat)[len(page_keys):]:
            ax.set_visible(False)
        if method == "pca":
            xlabel, ylabel = "−PC1 (score dB)", "−PC2 (score dB)"
        else:
            unit = "Bark" if bark else "Hz"
            xlabel, ylabel = f"−F2 ({unit})", f"−F1 ({unit})"
        fig.supxlabel(xlabel); fig.supylabel(ylabel)
        fig.suptitle(f"Midpoint vowel spaces: {method}, grouped by {grouping}")
        fig.tight_layout(rect=(0, 0, 1, .985))
        suffix = f"_{page:02d}" if len(keys) > groups_per_page else ""
        fig.savefig(output / f"midpoint_vowel_spaces{suffix}.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--method", choices=tuple(METHOD_COLUMNS), default="praat-fixed")
    parser.add_argument(
        "--bark", action="store_true",
        help="Use Bark F1/F2 for Praat or FastTrack; PCA is unchanged.",
    )
    parser.add_argument("--group-by", choices=("village", "speaker"), default="village")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--groups-per-page", type=int, default=24)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.groups_per_page < 1:
        parser.error("--groups-per-page must be positive")
    space_suffix = "_bark" if args.bark and args.method != "pca" else ""
    output = args.output_dir or Path(
        f"Analyses/Midpoint_vowel_spaces_{args.method}{space_suffix}_{args.group_by}"
    )
    output.mkdir(parents=True, exist_ok=True)

    tokens = read_tokens(args.input, args.method, args.include_incomplete, args.bark)
    speaker_rows = speaker_vowels(tokens)
    positions = aggregate(speaker_rows, args.group_by)
    midpoints, angles = analyze(positions, args.method, args.group_by)
    write_csv(output / "vowel_positions.csv", positions)
    write_csv(output / "pair_midpoints.csv", midpoints)
    write_csv(output / "ellipse_angles.csv", angles)
    if args.plot:
        make_plots(
            output, positions, angles, args.method, args.group_by,
            args.groups_per_page, args.bark,
        )
    print(
        f"{args.method}: {len(tokens)} tokens, {len(positions)} group-vowel positions, "
        f"{len(angles)} ellipse rows; wrote {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
