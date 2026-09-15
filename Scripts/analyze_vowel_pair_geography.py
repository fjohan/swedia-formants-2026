#!/usr/bin/env python3
"""Analyze vowel-pair endpoints, midpoints, and separation over geography."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-pair-geography-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f as f_distribution

from plot_midpoint_vowel_spaces import PAIRS, VOWELS, aggregate, read_tokens, speaker_vowels


OUTCOMES = ("first_endpoint", "second_endpoint", "midpoint", "separation_change")


def corpus_anchors(speaker_rows: list[dict]) -> dict[str, np.ndarray]:
    grouped = defaultdict(list)
    for row in speaker_rows:
        grouped[row["vowel"]].append((row["display_x"], row["display_y"]))
    return {
        vowel: np.median(np.asarray(grouped[vowel], dtype=float), axis=0)
        for vowel in VOWELS if grouped[vowel]
    }


def pair_measurements(rows: list[dict], anchors: dict[str, np.ndarray], method: str,
                      grouping: str) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["village"], row["speaker"] if grouping == "speaker" else "")
        grouped[key][row["vowel"]] = row
    output = []
    for (village, speaker), vowels in sorted(grouped.items()):
        for first, second in PAIRS:
            if first not in vowels or second not in vowels or first not in anchors or second not in anchors:
                continue
            first_anchor, second_anchor = anchors[first], anchors[second]
            pair_vector = second_anchor - first_anchor
            anchor_length = float(np.linalg.norm(pair_vector))
            if not anchor_length:
                continue
            unit = pair_vector / anchor_length
            first_observed = np.asarray([
                vowels[first]["display_x"], vowels[first]["display_y"]
            ])
            second_observed = np.asarray([
                vowels[second]["display_x"], vowels[second]["display_y"]
            ])
            observed_midpoint = (first_observed + second_observed) / 2
            anchor_midpoint = (first_anchor + second_anchor) / 2
            output.append({
                "method": method, "grouping": grouping,
                "village": village, "speaker": speaker,
                "pair": f"{first}->{second}",
                "first_vowel": first, "second_vowel": second,
                "geo_x": vowels[first]["geo_x"], "geo_y": vowels[first]["geo_y"],
                "anchor_pair_length": anchor_length,
                "first_endpoint": float((first_observed - first_anchor) @ unit),
                "second_endpoint": float((second_observed - second_anchor) @ unit),
                "midpoint": float((observed_midpoint - anchor_midpoint) @ unit),
                "observed_separation": float((second_observed - first_observed) @ unit),
                "separation_change": float((second_observed - first_observed) @ unit - anchor_length),
            })
    return output


def village_means(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["village"])].append(row)
    output = []
    for (pair, village), values in sorted(grouped.items()):
        output.append({
            "pair": pair, "village": village,
            "geo_x": values[0]["geo_x"], "geo_y": values[0]["geo_y"],
            "n_observations": len(values),
            **{outcome: float(np.median([row[outcome] for row in values]))
               for outcome in OUTCOMES},
        })
    return output


def natural_spline(x: np.ndarray, knots: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if knots is None:
        knots = np.quantile(x, [0, .25, .75, 1])
    knots = np.asarray(knots, dtype=float)

    def truncated(knot: float) -> np.ndarray:
        return ((np.maximum(x - knot, 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3)
                / (knots[-1] - knot))

    last = truncated(knots[-2])
    return np.column_stack([
        np.ones(len(x)), x,
        *(truncated(knots[index]) - last for index in range(len(knots) - 2)),
    ]), knots


def nested_f_test(y: np.ndarray, reduced: np.ndarray, full: np.ndarray) -> tuple[float, float, float]:
    reduced_residual = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    full_residual = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    reduced_rss = float(reduced_residual @ reduced_residual)
    full_rss = float(full_residual @ full_residual)
    df1, df2 = full.shape[1] - reduced.shape[1], len(y) - full.shape[1]
    if df1 <= 0 or df2 <= 0 or full_rss <= 0:
        return math.nan, math.nan, full_rss
    value = max(0.0, ((reduced_rss - full_rss) / df1) / (full_rss / df2))
    return value, float(f_distribution.sf(value, df1, df2)), full_rss


def statistics(villages: list[dict], method: str, grouping: str) -> list[dict]:
    output = []
    for first, second in PAIRS:
        pair = f"{first}->{second}"
        selected = [row for row in villages if row["pair"] == pair]
        x = np.asarray([row["geo_y"] for row in selected], dtype=float)
        linear = np.column_stack([np.ones(len(x)), x])
        spline, knots = natural_spline(x)
        null = np.ones((len(x), 1))
        for outcome in OUTCOMES:
            y = np.asarray([row[outcome] for row in selected], dtype=float)
            nonlinear_f, nonlinear_p, rss = nested_f_test(y, linear, spline)
            overall_f, overall_p, _ = nested_f_test(y, null, spline)
            total = float(np.sum((y - y.mean()) ** 2))
            output.append({
                "method": method, "grouping": grouping, "pair": pair,
                "outcome": outcome, "n_villages": len(selected),
                "spline_df": spline.shape[1],
                "spline_overall_f": overall_f, "spline_overall_p": overall_p,
                "nonlinearity_f": nonlinear_f, "nonlinearity_p": nonlinear_p,
                "spline_r2": 1 - rss / total if total else math.nan,
                "knots_geo_y": ";".join(f"{value:.6g}" for value in knots),
            })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_label(pair: str) -> str:
    first, second = pair.split("->")
    return f"/{first}/ → /{second}/"


def fitted_curve(villages: list[dict], outcome: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["geo_y"] for row in villages], dtype=float)
    y = np.asarray([row[outcome] for row in villages], dtype=float)
    design, knots = natural_spline(x)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    grid = np.linspace(x.min(), x.max(), 250)
    grid_design, _ = natural_spline(grid, knots)
    return grid, grid_design @ coefficients


def plot(path: Path, observations: list[dict], villages: list[dict], stats: list[dict],
         unit: str) -> None:
    fig, axes = plt.subplots(len(PAIRS), 3, figsize=(16, 20), sharex=True,
                             constrained_layout=True)
    for row_index, (first, second) in enumerate(PAIRS):
        pair = f"{first}->{second}"
        observed = [row for row in observations if row["pair"] == pair]
        village = [row for row in villages if row["pair"] == pair]
        x = np.asarray([row["geo_y"] for row in observed], dtype=float)
        panels = (
            ("endpoints", (("first_endpoint", "#2678b8", "first vowel"),
                           ("second_endpoint", "#e87522", "second vowel"))),
            ("midpoint", (("midpoint", "#6842a8", "pair midpoint"),)),
            ("separation", (("separation_change", "#18875b", "pair separation"),)),
        )
        for column, (kind, fields) in enumerate(panels):
            ax = axes[row_index, column]
            descriptions = []
            for outcome, color, label in fields:
                values = np.asarray([row[outcome] for row in observed], dtype=float)
                ax.scatter(x, values, s=14, alpha=.25, color=color, edgecolors="none")
                grid, prediction = fitted_curve(village, outcome)
                ax.plot(grid, prediction, color=color, lw=2.4, label=label)
                stat = next(row for row in stats if row["pair"] == pair
                            and row["outcome"] == outcome)
                prefix = ("first " if outcome == "first_endpoint" else
                          "second " if outcome == "second_endpoint" else "")
                descriptions.append(
                    f"{prefix}overall p={stat['spline_overall_p']:.3g}; "
                    f"nonlinear p={stat['nonlinearity_p']:.3g}"
                )
            ax.axhline(0, color="#777", lw=.75)
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
            heading = {
                "endpoints": "Both endpoints",
                "midpoint": "Corpus-anchor midpoint",
                "separation": "Separation: compression − / expansion +",
            }[kind]
            ax.set_title(heading + "\n" + "; ".join(descriptions), fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{pair_label(pair)}\n{unit} along first → second")
    for ax in axes[-1]:
        ax.set_xlabel("geographic y (north → south)")
    fig.suptitle("Vowel-pair endpoints, midpoints, and separation over north–south position")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--method", choices=("praat-fixed", "fasttrack", "pca"),
                        default="praat-fixed")
    parser.add_argument("--bark", action="store_true",
                        help="Use Bark F1/F2 for Praat or FastTrack; PCA is unchanged.")
    parser.add_argument("--group-by", choices=("speaker", "village"), default="speaker")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    tokens = read_tokens(args.input, args.method, args.include_incomplete, args.bark)
    if not args.include_reference:
        tokens = [row for row in tokens if row["village"] != "ref"]
    speakers = speaker_vowels(tokens)
    anchors = corpus_anchors(speakers)
    grouped = aggregate(speakers, args.group_by)
    measurements = pair_measurements(grouped, anchors, args.method, args.group_by)
    villages = village_means(measurements)
    stats = statistics(villages, args.method, args.group_by)

    space = "pca" if args.method == "pca" else "bark" if args.bark else "hz"
    output = args.output_dir or Path(
        f"Analyses/Vowel_pair_geography_{args.method}_{space}_{args.group_by}"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "corpus_vowel_anchors.csv", [
        {"method": args.method, "space": space, "vowel": vowel,
         "display_x": anchors[vowel][0], "display_y": anchors[vowel][1]}
        for vowel in VOWELS if vowel in anchors
    ])
    write_csv(output / "pair_measurements.csv", measurements)
    write_csv(output / "pair_village_means.csv", villages)
    write_csv(output / "pair_north_south_statistics.csv", stats)
    if args.plot:
        unit = "PC-score dB" if args.method == "pca" else "Bark" if args.bark else "Hz"
        plot(output / "pair_endpoints_midpoints_separation.png",
             measurements, villages, stats, unit)
    print(
        f"{args.method} ({space}): {len(measurements)} pair observations, "
        f"{len(villages)} village-pair means; wrote {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
