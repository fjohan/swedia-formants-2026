#!/usr/bin/env python3
"""Model one curved geographic-axis trajectory per vowel in acoustic space."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-vowel-trajectories-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import f as f_distribution

from plot_midpoint_vowel_spaces import VOWELS, read_tokens, speaker_vowels
from plot_regional_pair_midpoint_summary import ALIASES, resource_provinces


FILE_LABELS = {
    "uː": "u", "oː": "o", "ɑː": "open_back_a", "æː": "ae",
    "eː": "e", "yː": "y", "ʉ̟ː": "central_u", "øː": "oe",
}
DEFAULT_EXCLUDED_REGIONS = {"Gotland", "Åboland", "Nyland", "Österbotten", "Åland"}
GROUP_COLORS = {"South": "#c43c39", "Centre": "#777777", "North": "#315fa8"}
AXIS_CMAP = LinearSegmentedColormap.from_list(
    "south_centre_north", [GROUP_COLORS["South"], GROUP_COLORS["Centre"],
                            GROUP_COLORS["North"]]
)


def geographic_axis(speakers: list[dict], south_name: str, north_name: str,
                    central_names: list[str], margin: float,
                    coordinate_mode: str, partition: str) -> dict:
    grouped = {}
    for row in speakers:
        grouped.setdefault(row["village"], []).append((row["geo_x"], row["geo_y"]))
    coordinates = {name: np.median(np.asarray(values), axis=0)
                   for name, values in grouped.items()}
    required = ([south_name, north_name, *central_names]
                if coordinate_mode == "axis" or partition == "band" else [])
    missing = [name for name in required if name not in coordinates]
    if missing:
        raise ValueError("Geographic reference villages absent: " + ", ".join(missing))

    if coordinate_mode == "axis":
        south, north = coordinates[south_name], coordinates[north_name]
        vector = north - south
        axis_length = float(np.linalg.norm(vector))
        unit = vector / axis_length
        positions = {name: float((point - south) @ unit)
                     for name, point in coordinates.items()}
        increases_north = True
    else:
        south = north = unit = None
        axis_length = math.nan
        positions = {name: float(point[1]) for name, point in coordinates.items()}
        increases_north = False

    if partition == "band":
        central = [positions[name] for name in central_names]
        lower, upper = min(central) - margin, max(central) + margin
        if increases_north:
            groups = {name: "South" if value < lower else "North" if value > upper else "Centre"
                      for name, value in positions.items()}
        else:
            groups = {name: "North" if value < lower else "South" if value > upper else "Centre"
                      for name, value in positions.items()}
    else:
        lower = upper = math.nan
        groups = {}

    return {
        "coordinates": coordinates, "positions": positions, "groups": groups,
        "increases_north": increases_north, "south": south, "north": north,
        "unit": unit, "axis_length": axis_length,
        "central_lower": lower, "central_upper": upper,
    }


def village_vowels(speakers: list[dict]) -> list[dict]:
    grouped = {}
    for row in speakers:
        grouped.setdefault((row["village"], row["vowel"]), []).append(row)
    output = []
    for (village, vowel), values in sorted(grouped.items()):
        output.append({
            "village": village, "vowel": vowel, "n_speakers": len(values),
            "geo_x": float(np.median([row["geo_x"] for row in values])),
            "geo_y": float(np.median([row["geo_y"] for row in values])),
            "display_x": float(np.median([row["display_x"] for row in values])),
            "display_y": float(np.median([row["display_y"] for row in values])),
        })
    return output


def spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)

    def truncated(knot: float) -> np.ndarray:
        return ((np.maximum(x - knot, 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3)
                / (knots[-1] - knot))

    last = truncated(knots[-2])
    return np.column_stack([
        np.ones(len(x)), x,
        *(truncated(knots[index]) - last for index in range(len(knots) - 2)),
    ])


def nested_test(response: np.ndarray, reduced: np.ndarray,
                full: np.ndarray) -> tuple[float, float]:
    reduced_residual = response - reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    full_residual = response - full @ np.linalg.lstsq(full, response, rcond=None)[0]
    reduced_rss = float(reduced_residual @ reduced_residual)
    full_rss = float(full_residual @ full_residual)
    df1, df2 = full.shape[1] - reduced.shape[1], len(response) - full.shape[1]
    if df1 <= 0 or df2 <= 0 or full_rss <= 0:
        return math.nan, math.nan
    value = max(0.0, ((reduced_rss - full_rss) / df1) / (full_rss / df2))
    return value, float(f_distribution.sf(value, df1, df2))


def fit_response(response: np.ndarray, x: np.ndarray, basis: np.ndarray) -> dict:
    null = np.ones((len(x), 1))
    linear = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(basis, response, rcond=None)[0]
    fitted = basis @ coefficients
    residual = response - fitted
    total = float(np.sum((response - response.mean()) ** 2))
    overall_f, overall_p = nested_test(response, null, basis)
    nonlinear_f, nonlinear_p = nested_test(response, linear, basis)
    return {
        "coefficients": coefficients,
        "r2": 1 - float(residual @ residual) / total if total else math.nan,
        "overall_f": overall_f, "overall_p": overall_p,
        "nonlinearity_f": nonlinear_f, "nonlinearity_p": nonlinear_p,
    }


def bootstrap_predictions(x: np.ndarray, responses: np.ndarray, knots: np.ndarray,
                          grid_basis: np.ndarray, iterations: int,
                          rng: np.random.Generator) -> np.ndarray:
    predictions = np.empty((iterations, len(grid_basis), 2), dtype=float)
    for iteration in range(iterations):
        indices = rng.integers(0, len(x), len(x))
        basis = spline_basis(x[indices], knots)
        for coordinate in range(2):
            coefficients = np.linalg.lstsq(
                basis, responses[indices, coordinate], rcond=None
            )[0]
            predictions[iteration, :, coordinate] = grid_basis @ coefficients
    return predictions


def confidence_ellipse(ax, points: np.ndarray, color: str = "#222") -> None:
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    # Approximately 95% for a bivariate normal bootstrap cloud.
    scale = math.sqrt(5.991)
    ellipse = Ellipse(center, 2 * scale * math.sqrt(max(values[0], 0)),
                      2 * scale * math.sqrt(max(values[1], 0)), angle=angle,
                      facecolor=color, edgecolor=color, alpha=.10, lw=1.2, zorder=2)
    ax.add_patch(ellipse)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_vowel(path: Path, vowel: str, speakers: list[dict], villages: list[dict],
               grid: np.ndarray, predicted: np.ndarray, bootstrap: np.ndarray,
               models: dict[str, dict], method: str, space: str,
               marker_positions: list[tuple[float, str, str]],
               coordinate_label: str, increases_north: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    speaker_x = np.asarray([row["display_x"] for row in speakers])
    speaker_y = np.asarray([row["display_y"] for row in speakers])
    speaker_geo = np.asarray([row["geo_position"] for row in speakers])
    norm = Normalize(grid.min(), grid.max())
    cmap = AXIS_CMAP if increases_north else AXIS_CMAP.reversed()
    speaker_colors = [GROUP_COLORS.get(row.get("geographic_group"),
                                       cmap(norm(row["geo_position"])))
                      for row in speakers]
    ax.scatter(speaker_x, speaker_y, c=speaker_colors,
               s=22, alpha=.38, edgecolors="none", zorder=1)
    ax.scatter([row["display_x"] for row in villages],
               [row["display_y"] for row in villages],
               facecolors="none", edgecolors="#333", s=30, linewidth=.7,
               label="village medians", zorder=3)

    segments = np.stack([predicted[:-1], predicted[1:]], axis=1)
    collection = LineCollection(segments, cmap=cmap, norm=norm,
                                linewidth=4, zorder=4)
    collection.set_array((grid[:-1] + grid[1:]) / 2)
    ax.add_collection(collection)
    for position, label, marker in marker_positions:
        index = int(np.argmin(np.abs(grid - position)))
        confidence_ellipse(ax, bootstrap[:, index, :])
        ax.scatter(*predicted[index], marker=marker, color=GROUP_COLORS[label],
                   edgecolor="#111", s=85, zorder=6, label=label)
    arrow_index = round(.70 * (len(grid) - 1)) if increases_north else round(.30 * (len(grid) - 1))
    arrow_step = 2 if increases_north else -2
    ax.annotate("", xy=predicted[arrow_index + arrow_step],
                xytext=predicted[arrow_index - arrow_step],
                arrowprops={"arrowstyle": "->", "color": "#111", "lw": 2}, zorder=7)

    if method == "pca":
        xlabel, ylabel = "−PC1 (score dB)", "−PC2 (score dB)"
    else:
        unit = "Bark" if space == "bark" else "Hz"
        xlabel, ylabel = f"−F2 ({unit})", f"−F1 ({unit})"
    stats_text = (
        f'horizontal: R²={models["display_x"]["r2"]:.2f}, '
        f'overall p={models["display_x"]["overall_p"]:.3g}, '
        f'nonlinear p={models["display_x"]["nonlinearity_p"]:.3g}\n'
        f'vertical: R²={models["display_y"]["r2"]:.2f}, '
        f'overall p={models["display_y"]["overall_p"]:.3g}, '
        f'nonlinear p={models["display_y"]["nonlinearity_p"]:.3g}'
    )
    ax.text(.02, .02, stats_text, transform=ax.transAxes, va="bottom", fontsize=8,
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#aaa"})
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=f"/{vowel}/: fitted south-to-north acoustic trajectory\n"
                 f"{len(speakers)} speakers, {len(villages)} villages")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.18)
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label=coordinate_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--method", choices=("praat-fixed", "fasttrack", "pca"),
                        default="praat-fixed")
    parser.add_argument("--bark", action="store_true",
                        help="Use Bark F1/F2 for Praat or FastTrack; PCA is unchanged.")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--coordinate", choices=("axis", "raw-y"), default="axis",
                        help="Continuous geographic predictor; raw-y reproduces the old model.")
    parser.add_argument("--south-endpoint", default="loderup")
    parser.add_argument("--north-endpoint", default="arjeplog")
    parser.add_argument("--partition", choices=("band", "proportional"), default="band",
                        help="Band uses geographic groups; proportional uses old 10/50/90%% markers.")
    parser.add_argument("--central-villages", nargs="+", default=["tjallmo", "rimforsa"])
    parser.add_argument("--central-margin", type=float, default=35.0)
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--include-finland-gotland", action="store_true",
                        help="Retain Gotland and the four Finland regions; excluded by default.")
    parser.add_argument("--vowels", nargs="+", choices=VOWELS, default=list(VOWELS))
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error("--bootstrap must be at least 20")
    if args.central_margin < 0:
        parser.error("--central-margin must be non-negative")

    tokens = read_tokens(args.input, args.method, args.include_incomplete, args.bark)
    if not args.include_reference:
        tokens = [row for row in tokens if row["village"] != "ref"]
    provinces = resource_provinces(args.resource)
    excluded_villages = sorted({
        row["village"] for row in tokens
        if not args.include_finland_gotland
        and provinces.get(ALIASES.get(row["village"], row["village"]))
        in DEFAULT_EXCLUDED_REGIONS
    })
    tokens = [row for row in tokens if row["village"] not in excluded_villages]
    speakers = speaker_vowels(tokens)
    geography = geographic_axis(
        speakers, args.south_endpoint, args.north_endpoint,
        args.central_villages, args.central_margin,
        args.coordinate, args.partition,
    )
    for row in speakers:
        row["geo_position"] = geography["positions"][row["village"]]
        row["geographic_group"] = geography["groups"].get(row["village"], "")
    villages = village_vowels(speakers)
    for row in villages:
        row["geo_position"] = geography["positions"][row["village"]]
        row["geographic_group"] = geography["groups"].get(row["village"], "")
    space = "pca" if args.method == "pca" else "bark" if args.bark else "hz"
    output_label = args.method if args.method == "pca" else f"{args.method}_{space}"
    geography_label = "axis_band" if args.coordinate == "axis" and args.partition == "band" else f"{args.coordinate}_{args.partition}"
    output = args.output_dir or Path(
        f"Analyses/Vowel_NS_trajectories_{output_label}_{geography_label}"
    )
    output.mkdir(parents=True, exist_ok=True)
    statistics_rows, prediction_rows = [], []

    for vowel_index, vowel in enumerate(args.vowels):
        speaker_subset = [row for row in speakers if row["vowel"] == vowel]
        village_subset = [row for row in villages if row["vowel"] == vowel]
        x = np.asarray([row["geo_position"] for row in village_subset], dtype=float)
        responses = np.asarray([[row["display_x"], row["display_y"]]
                                for row in village_subset], dtype=float)
        if len(x) < 8 or len(np.unique(x)) < 4:
            print(f"Skipping /{vowel}/: insufficient geographic coverage")
            continue
        knots = np.quantile(x, [0, .25, .75, 1])
        basis = spline_basis(x, knots)
        grid = np.linspace(x.min(), x.max(), 180)
        grid_basis = spline_basis(grid, knots)
        models = {}
        predicted = np.empty((len(grid), 2))
        for coordinate, name in enumerate(("display_x", "display_y")):
            model = fit_response(responses[:, coordinate], x, basis)
            models[name] = model
            predicted[:, coordinate] = grid_basis @ model["coefficients"]
            statistics_rows.append({
                "method": args.method, "space": space, "vowel": vowel,
                "response": name, "n_villages": len(x),
                "spline_df": basis.shape[1], "r2": model["r2"],
                "overall_f": model["overall_f"], "overall_p": model["overall_p"],
                "nonlinearity_f": model["nonlinearity_f"],
                "nonlinearity_p": model["nonlinearity_p"],
                "geographic_coordinate": args.coordinate,
                "knots_geographic_position": ";".join(f"{value:.6g}" for value in knots),
                **{f"coefficient_{index}": float(value)
                   for index, value in enumerate(model["coefficients"])},
            })
        bootstrap = bootstrap_predictions(
            x, responses, knots, grid_basis, args.bootstrap,
            np.random.default_rng(args.seed + vowel_index),
        )
        lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        for index, geo_position in enumerate(grid):
            prediction_rows.append({
                "method": args.method, "space": space, "vowel": vowel,
                "geographic_coordinate": args.coordinate,
                "geographic_position": geo_position,
                "predicted_display_x": predicted[index, 0],
                "predicted_display_y": predicted[index, 1],
                "display_x_lower_95": lower[index, 0],
                "display_x_upper_95": upper[index, 0],
                "display_y_lower_95": lower[index, 1],
                "display_y_upper_95": upper[index, 1],
            })
        filename = f"{VOWELS.index(vowel) + 1:02d}_{FILE_LABELS[vowel]}_trajectory.png"
        if args.partition == "band":
            marker_positions = []
            for label, marker in (("North", "^"), ("Centre", "o"), ("South", "s")):
                values = [row["geo_position"] for row in village_subset
                          if row["geographic_group"] == label]
                if values:
                    marker_positions.append((float(np.median(values)), label, marker))
        else:
            proportions = ((.1, "South", "s"), (.5, "Centre", "o"), (.9, "North", "^"))
            if not geography["increases_north"]:
                proportions = ((.1, "North", "^"), (.5, "Centre", "o"), (.9, "South", "s"))
            marker_positions = [(float(np.quantile(x, proportion)), label, marker)
                                for proportion, label, marker in proportions]
        coordinate_label = (f"position on {args.south_endpoint} → {args.north_endpoint} axis "
                            "(south red → north blue)" if args.coordinate == "axis"
                            else "geographic y (north blue → south red)")
        plot_vowel(output / filename, vowel, speaker_subset, village_subset,
                   grid, predicted, bootstrap, models, args.method, space,
                   marker_positions, coordinate_label, geography["increases_north"])

    write_csv(output / "village_vowel_positions.csv",
              [row for row in villages if row["vowel"] in args.vowels])
    write_csv(output / "trajectory_model_statistics.csv", statistics_rows)
    write_csv(output / "trajectory_predictions.csv", prediction_rows)
    write_csv(output / "excluded_villages.csv", [
        {"village": village,
         "region": provinces[ALIASES.get(village, village)]}
        for village in excluded_villages
    ])
    group_counts = {
        group: len({(row["village"], row["speaker"]) for row in speakers
                    if row["geographic_group"] == group})
        for group in ("North", "Centre", "South")
    }
    settings = {
        "input": str(args.input), "method": args.method, "space": space,
        "coordinate": args.coordinate, "partition": args.partition,
        "south_endpoint": args.south_endpoint, "north_endpoint": args.north_endpoint,
        "central_villages": args.central_villages,
        "central_margin": args.central_margin,
        "central_band": [geography["central_lower"], geography["central_upper"]],
        "axis_length": geography["axis_length"],
        "axis_south_coordinates": (None if geography["south"] is None
                                   else geography["south"].tolist()),
        "axis_north_coordinates": (None if geography["north"] is None
                                   else geography["north"].tolist()),
        "group_speaker_counts": group_counts,
        "include_incomplete": args.include_incomplete,
        "excluded_regions": ([] if args.include_finland_gotland
                             else sorted(DEFAULT_EXCLUDED_REGIONS)),
        "excluded_villages": excluded_villages,
        "colors": "red south, blue north",
    }
    (output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(statistics_rows) // 2} vowel models to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
