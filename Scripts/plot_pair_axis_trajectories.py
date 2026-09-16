#!/usr/bin/env python3
"""Plot paired vowel trajectories and directed speaker pair geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-pca-pair-trajectories-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

from model_vowel_axis_trajectories import (
    DEFAULT_EXCLUDED_REGIONS, fit_response, geographic_axis, spline_basis,
)
from plot_midpoint_vowel_spaces import VOWELS, read_tokens, speaker_vowels
from plot_regional_pair_midpoint_summary import ALIASES, resource_provinces


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def paired_speakers(rows: list[dict], first: str, second: str) -> list[dict]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["village"], row["speaker"]][row["vowel"]] = row
    output = []
    for (village, speaker), vowels in sorted(grouped.items()):
        if first not in vowels or second not in vowels:
            continue
        a, b = vowels[first], vowels[second]
        dx, dy = b["display_x"] - a["display_x"], b["display_y"] - a["display_y"]
        output.append({
            "village": village, "speaker": speaker,
            "first_vowel": first, "second_vowel": second,
            "geo_x": a["geo_x"], "geo_y": a["geo_y"],
            "first_x": a["display_x"], "first_y": a["display_y"],
            "second_x": b["display_x"], "second_y": b["display_y"],
            "pair_dx": dx, "pair_dy": dy,
            "pair_length": math.hypot(dx, dy),
            "directed_angle_from_horizontal_deg": math.degrees(math.atan2(dy, dx)),
        })
    return output


def village_pairs(speakers: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in speakers:
        grouped[row["village"]].append(row)
    output = []
    for village, values in sorted(grouped.items()):
        output.append({
            "village": village,
            "geo_x": float(np.median([row["geo_x"] for row in values])),
            "geo_y": float(np.median([row["geo_y"] for row in values])),
            "geo_position": float(np.median([row["geo_position"] for row in values])),
            "geographic_group": values[0].get("geographic_group", ""),
            **{field: float(np.median([row[field] for row in values]))
               for field in ("first_x", "first_y", "second_x", "second_y")},
        })
    return output


def fit_trajectory(villages: list[dict], fields: tuple[str, str], grid: np.ndarray,
                   knots: np.ndarray) -> np.ndarray:
    geo_position = np.asarray([row["geo_position"] for row in villages], dtype=float)
    basis = spline_basis(geo_position, knots)
    grid_basis = spline_basis(grid, knots)
    predicted = np.empty((len(grid), 2))
    for coordinate, field in enumerate(fields):
        response = np.asarray([row[field] for row in villages], dtype=float)
        model = fit_response(response, geo_position, basis)
        predicted[:, coordinate] = grid_basis @ model["coefficients"]
    return predicted


def colored_curve(ax, points: np.ndarray, grid: np.ndarray, norm: Normalize,
                  cmap_name: str, linewidth: float = 4) -> None:
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap=cmap_name, norm=norm,
                                linewidth=linewidth, zorder=5)
    collection.set_array((grid[:-1] + grid[1:]) / 2)
    ax.add_collection(collection)


def predictive_geometry_models(geo_y: np.ndarray, angles: np.ndarray,
                               distances: np.ndarray, seed: int = 20260916) -> list[dict]:
    specifications = {
        "angle_only": np.column_stack([np.ones(len(geo_y)), angles]),
        "distance_only": np.column_stack([np.ones(len(geo_y)), distances]),
        "angle_and_distance": np.column_stack([np.ones(len(geo_y)), angles, distances]),
    }
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(geo_y)), 10)
    output = []
    total = float(np.sum((geo_y - geo_y.mean()) ** 2))
    for name, design in specifications.items():
        coefficients = np.linalg.lstsq(design, geo_y, rcond=None)[0]
        residual = geo_y - design @ coefficients
        rss = float(residual @ residual)
        r2 = 1 - rss / total
        predictors = design.shape[1] - 1
        adjusted = 1 - (1 - r2) * (len(geo_y) - 1) / (len(geo_y) - predictors - 1)
        predictions = np.empty(len(geo_y))
        for test in folds:
            train = np.setdiff1d(np.arange(len(geo_y)), test)
            beta = np.linalg.lstsq(design[train], geo_y[train], rcond=None)[0]
            predictions[test] = design[test] @ beta
        cv_r2 = 1 - float(np.sum((geo_y - predictions) ** 2)) / total
        output.append({
            "model": name, "n_villages": len(geo_y), "n_predictors": predictors,
            "r2": r2, "adjusted_r2": adjusted, "cv_r2_10fold": cv_r2,
            **{f"coefficient_{index}": float(value)
               for index, value in enumerate(coefficients)},
        })
    return output


def plot(path: Path, speakers: list[dict], villages: list[dict], first_curve: np.ndarray,
         second_curve: np.ndarray, grid: np.ndarray, first: str,
         second: str, method: str, space: str,
         marker_positions: list[tuple[float, str, str]],
         coordinate_label: str, increases_north: bool
         ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    fig, (space_ax, angle_ax, distance_ax) = plt.subplots(
        1, 3, figsize=(21, 7), constrained_layout=True
    )
    norm = Normalize(grid.min(), grid.max())
    cmap_name = "coolwarm_r" if increases_north else "coolwarm"
    cmap = plt.get_cmap(cmap_name)

    segments = np.asarray([[[row["first_x"], row["first_y"]],
                            [row["second_x"], row["second_y"]]] for row in speakers])
    segment_collection = LineCollection(
        segments, cmap=cmap_name, norm=norm, linewidth=.8, alpha=.22, zorder=1,
    )
    segment_collection.set_array(np.asarray([row["geo_position"] for row in speakers]))
    space_ax.add_collection(segment_collection)
    space_ax.scatter([row["first_x"] for row in speakers], [row["first_y"] for row in speakers],
                     c=[row["geo_position"] for row in speakers], cmap=cmap_name, norm=norm,
                     s=13, alpha=.35, edgecolors="none")
    space_ax.scatter([row["second_x"] for row in speakers], [row["second_y"] for row in speakers],
                     c=[row["geo_position"] for row in speakers], cmap=cmap_name, norm=norm,
                     s=13, alpha=.35, marker="s", edgecolors="none")
    colored_curve(space_ax, first_curve, grid, norm, cmap_name)
    colored_curve(space_ax, second_curve, grid, norm, cmap_name)
    middle = len(grid) // 2
    space_ax.annotate(f"/{first}/ path", first_curve[middle], xytext=(7, 10),
                      textcoords="offset points", fontsize=9, fontweight="bold")
    space_ax.annotate(f"/{second}/ path", second_curve[middle], xytext=(7, -15),
                      textcoords="offset points", fontsize=9, fontweight="bold")

    fitted_rows = []
    angle_labels = []
    for position, label, marker in marker_positions:
        index = int(np.argmin(np.abs(grid - position)))
        a, b = first_curve[index], second_curve[index]
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        space_ax.annotate("", xy=b, xytext=a,
                          arrowprops={"arrowstyle": "->", "color": cmap(norm(grid[index])),
                                      "lw": 3, "shrinkA": 5, "shrinkB": 5}, zorder=7)
        angle_labels.append(f"{label}: {angle:+.1f}°")
        space_ax.scatter(*a, marker=marker, s=75, color=cmap(norm(grid[index])),
                         edgecolor="#111", zorder=8, label=label)
        space_ax.scatter(*b, marker=marker, s=75, color=cmap(norm(grid[index])),
                         edgecolor="#111", zorder=8)
        fitted_rows.append({
            "region_marker": label, "geographic_position": grid[index],
            "first_x": a[0], "first_y": a[1],
            "second_x": b[0], "second_y": b[1],
            "pair_length": float(np.linalg.norm(b - a)),
            "directed_angle_from_horizontal_deg": angle,
        })
    space_ax.plot([], [], color="#555", lw=1, label="speaker pair vectors")
    space_ax.plot([], [], color="#222", lw=4, label=f"fitted /{first}/ and /{second}/ paths")
    space_ax.text(.02, .02, "Fitted pair angles\n" + "\n".join(angle_labels),
                  transform=space_ax.transAxes, va="bottom", fontsize=8,
                  bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#aaa"})
    if method == "pca":
        xlabel, ylabel = "−PC1 (score dB)", "−PC2 (score dB)"
        distance_unit = "PC-score dB"
    else:
        unit = "Bark" if space == "bark" else "Hz"
        xlabel, ylabel = f"−F2 ({unit})", f"−F1 ({unit})"
        distance_unit = unit
    space_ax.set(xlabel=xlabel, ylabel=ylabel,
                 title=f"/{first}/ → /{second}/ vectors and fitted {method} trajectories")
    space_ax.set_aspect("equal", adjustable="datalim"); space_ax.grid(alpha=.18)
    space_ax.legend(fontsize=8)

    geo_position = np.asarray([row["geo_position"] for row in speakers], dtype=float)
    angles = np.asarray([row["directed_angle_from_horizontal_deg"] for row in speakers])
    # Unwrap only for modelling/visual continuity; retain ordinary directed angles in CSV.
    reference = float(np.angle(np.mean(np.exp(1j * np.radians(angles))), deg=True))
    unwrapped = reference + ((angles - reference + 180) % 360 - 180)
    village_angles = []
    village_distances = []
    for village in villages:
        indices = [index for index, row in enumerate(speakers)
                   if row["village"] == village["village"]]
        village_angles.append(float(np.median(unwrapped[indices])))
        village_distances.append(float(np.median(
            [speakers[index]["pair_length"] for index in indices]
        )))
    village_geo = np.asarray([row["geo_position"] for row in villages])
    village_angles_array = np.asarray(village_angles)
    village_distances_array = np.asarray(village_distances)
    knots = np.quantile(village_geo, [0, .25, .75, 1])
    basis = spline_basis(village_geo, knots)
    angle_model = fit_response(village_angles_array, village_geo, basis)
    distance_model = fit_response(village_distances_array, village_geo, basis)
    grid_basis = spline_basis(grid, knots)
    angle_grid = grid_basis @ angle_model["coefficients"]
    distance_grid = grid_basis @ distance_model["coefficients"]
    angle_ax.scatter(geo_position, unwrapped, c=geo_position, cmap=cmap_name, norm=norm,
                     s=18, alpha=.35, edgecolors="none")
    angle_ax.plot(grid, angle_grid, color="#222", lw=3)
    angle_ax.axhline(0, color="#888", lw=.7)
    angle_ax.text(.03, .97,
                  f'village-balanced spline\nR²={angle_model["r2"]:.2f}\n'
                  f'overall p={angle_model["overall_p"]:.3g}\n'
                  f'nonlinear p={angle_model["nonlinearity_p"]:.3g}',
                  transform=angle_ax.transAxes, va="top",
                  bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#aaa"})
    angle_ax.set(xlabel=coordinate_label,
                 ylabel="directed pair angle from horizontal (degrees)",
                 title="Speaker pair angle over north–south position")
    angle_ax.grid(alpha=.18)
    speaker_distances = np.asarray([row["pair_length"] for row in speakers])
    distance_ax.scatter(geo_position, speaker_distances, c=geo_position,
                        cmap=cmap_name, norm=norm,
                        s=18, alpha=.35, edgecolors="none")
    distance_ax.plot(grid, distance_grid, color="#222", lw=3)
    distance_ax.text(
        .03, .97,
        f'village-balanced spline\nR²={distance_model["r2"]:.2f}\n'
        f'overall p={distance_model["overall_p"]:.3g}\n'
        f'nonlinear p={distance_model["nonlinearity_p"]:.3g}',
        transform=distance_ax.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#aaa"},
    )
    distance_ax.set(xlabel=coordinate_label,
                    ylabel=f"speaker pair distance ({distance_unit})",
                    title="Speaker pair distance over north–south position")
    distance_ax.grid(alpha=.18)

    geometry_models = predictive_geometry_models(
        village_geo, village_angles_array, village_distances_array
    )
    spline_models = [
        {"response": response, "n_villages": len(villages),
         "r2": model["r2"], "overall_f": model["overall_f"],
         "overall_p": model["overall_p"],
         "nonlinearity_f": model["nonlinearity_f"],
         "nonlinearity_p": model["nonlinearity_p"]}
        for response, model in (("angle", angle_model), ("distance", distance_model))
    ]
    combined = next(row for row in geometry_models if row["model"] == "angle_and_distance")
    distance_ax.text(
        .03, .03,
        "Reverse geographic prediction (village medians)\n"
        f'angle + distance: adjusted R²={combined["adjusted_r2"]:.2f}, '
        f'10-fold CV R²={combined["cv_r2_10fold"]:.2f}',
        transform=distance_ax.transAxes, va="bottom", fontsize=8,
        bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#aaa"},
    )
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_name), ax=space_ax,
                 label=coordinate_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    village_geometry = [
        {"village": village["village"], "geo_x": village["geo_x"],
         "geo_y": village["geo_y"], "geographic_position": village["geo_position"],
         "geographic_group": village["geographic_group"],
         "median_unwrapped_angle_deg": angle,
         "median_pair_distance": distance}
        for village, angle, distance in zip(villages, village_angles, village_distances)
    ]
    return fitted_rows, village_geometry, spline_models, geometry_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--method", choices=("praat-fixed", "fasttrack", "pca"),
                        default="pca")
    parser.add_argument("--bark", action="store_true",
                        help="Use Bark F1/F2 with Praat or FastTrack; PCA is unchanged.")
    parser.add_argument("--first", choices=VOWELS, default="uː")
    parser.add_argument("--second", choices=VOWELS, default="oː")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--coordinate", choices=("axis", "raw-y"), default="axis")
    parser.add_argument("--south-endpoint", default="loderup")
    parser.add_argument("--north-endpoint", default="arjeplog")
    parser.add_argument("--partition", choices=("band", "proportional"), default="band")
    parser.add_argument("--central-villages", nargs="+", default=["tjallmo", "rimforsa"])
    parser.add_argument("--central-margin", type=float, default=35.0)
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--include-finland-gotland", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.first == args.second:
        parser.error("--first and --second must differ")
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
    positions = speaker_vowels(tokens)
    geography = geographic_axis(
        positions, args.south_endpoint, args.north_endpoint,
        args.central_villages, args.central_margin,
        args.coordinate, args.partition,
    )
    for row in positions:
        row["geo_position"] = geography["positions"][row["village"]]
        row["geographic_group"] = geography["groups"].get(row["village"], "")
    paired = paired_speakers(positions, args.first, args.second)
    for row in paired:
        row["geo_position"] = geography["positions"][row["village"]]
        row["geographic_group"] = geography["groups"].get(row["village"], "")
    villages = village_pairs(paired)
    geo_position = np.asarray([row["geo_position"] for row in villages])
    knots = np.quantile(geo_position, [0, .25, .75, 1])
    grid = np.linspace(geo_position.min(), geo_position.max(), 180)
    first_curve = fit_trajectory(villages, ("first_x", "first_y"), grid, knots)
    second_curve = fit_trajectory(villages, ("second_x", "second_y"), grid, knots)

    if args.partition == "band":
        marker_positions = []
        for label, marker in (("North", "^"), ("Centre", "o"), ("South", "s")):
            values = [row["geo_position"] for row in villages
                      if row["geographic_group"] == label]
            if values:
                marker_positions.append((float(np.median(values)), label, marker))
    else:
        proportions = ((.1, "South", "s"), (.5, "Centre", "o"), (.9, "North", "^"))
        if not geography["increases_north"]:
            proportions = ((.1, "North", "^"), (.5, "Centre", "o"), (.9, "South", "s"))
        marker_positions = [(float(np.quantile(geo_position, proportion)), label, marker)
                            for proportion, label, marker in proportions]
    coordinate_label = (f"position on {args.south_endpoint} → {args.north_endpoint} axis"
                        if args.coordinate == "axis" else "geographic y")
    space = "pca" if args.method == "pca" else "bark" if args.bark else "hz"
    pair_label = f"{VOWELS.index(args.first)+1}_{VOWELS.index(args.second)+1}"
    output_label = args.method if args.method == "pca" else f"{args.method}_{space}"
    geography_label = ("axis_band" if args.coordinate == "axis" and args.partition == "band"
                       else f"{args.coordinate}_{args.partition}")
    output = args.output_dir or Path(
        f"Analyses/Pair_trajectory_{output_label}_{pair_label}_{geography_label}"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "speaker_pair_angles.csv", paired)
    write_csv(output / "village_pair_positions.csv", villages)
    fitted, village_geometry, spline_models, geometry_models = plot(
        output / "pair_trajectories_angles_distances.png", paired, villages,
        first_curve, second_curve, grid, args.first, args.second, args.method, space,
        marker_positions, coordinate_label, geography["increases_north"],
    )
    write_csv(output / "fitted_pair_summary.csv", fitted)
    write_csv(output / "village_pair_geometry.csv", village_geometry)
    write_csv(output / "pair_geometry_spline_models.csv", spline_models)
    write_csv(output / "pair_geometry_predictive_models.csv", geometry_models)
    write_csv(output / "excluded_villages.csv", [
        {"village": village, "region": provinces[ALIASES.get(village, village)]}
        for village in excluded_villages
    ])
    settings = {
        "input": str(args.input), "method": args.method, "space": space,
        "first": args.first, "second": args.second,
        "coordinate": args.coordinate, "partition": args.partition,
        "south_endpoint": args.south_endpoint, "north_endpoint": args.north_endpoint,
        "central_villages": args.central_villages, "central_margin": args.central_margin,
        "central_band": [geography["central_lower"], geography["central_upper"]],
        "excluded_regions": ([] if args.include_finland_gotland
                             else sorted(DEFAULT_EXCLUDED_REGIONS)),
        "excluded_villages": excluded_villages,
        "colors": "red south, blue north",
    }
    (output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(paired)} speaker pairs across {len(villages)} villages to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
