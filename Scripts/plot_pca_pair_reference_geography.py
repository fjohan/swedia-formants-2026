#!/usr/bin/env python3
"""Model a PCA vowel pair relative to geographic and acoustic reference villages.

The default reference is the equal-village mean of Tjallmo and Rimforsa.  Unlike
the older north--south analysis, this script retains both geographic axes and
compares distance-only, bearing-aware, and smooth 2-D spatial models by
leave-one-village-out cross-validation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-pca-reference-geography-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

from model_vowel_north_south_trajectories import spline_basis
from plot_midpoint_vowel_spaces import VOWELS, read_tokens, speaker_vowels


DEFAULT_EXCLUDED_REGIONS = ("Gotland", "Åboland", "Nyland", "Österbotten", "Åland")
RESOURCE_ALIASES = {
    "n_rorum": "nrorum", "pitea": "pite", "s_finnskoga": "sodrafinnskoga",
    "s_mellosa": "stmellosa", "st_anna": "stanna", "v_vingaker": "vingaker",
    "ref": "reference",
}


def resource_regions(path: Path) -> dict[str, str]:
    """Read village-to-region mappings from the Latin-1 Tcl resource file."""
    result = {}
    for raw in path.read_bytes().decode("latin-1").splitlines():
        line = raw.strip()
        if line.startswith("{") and line.endswith("}"):
            fields = shlex.split(line[1:-1])
            if len(fields) >= 6:
                result[fields[0]] = fields[3]
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair_speakers(rows: list[dict], first: str, second: str) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["village"], row["speaker"]][row["vowel"]] = row
    output = []
    for (village, speaker), vowels in sorted(grouped.items()):
        if first not in vowels or second not in vowels:
            continue
        a, b = vowels[first], vowels[second]
        output.append({
            "village": village, "speaker": speaker,
            "geo_x": a["geo_x"], "geo_y": a["geo_y"],
            "first_x": a["display_x"], "first_y": a["display_y"],
            "second_x": b["display_x"], "second_y": b["display_y"],
        })
    return output


def village_means(speakers: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in speakers:
        grouped[row["village"]].append(row)
    output = []
    for village, values in sorted(grouped.items()):
        output.append({
            "village": village, "n_speakers": len(values),
            **{field: float(np.mean([row[field] for row in values]))
               for field in ("geo_x", "geo_y", "first_x", "first_y",
                             "second_x", "second_y")},
        })
    return output


def reference_values(villages: list[dict], names: list[str]) -> dict:
    found = [row for row in villages if row["village"] in names]
    missing = sorted(set(names) - {row["village"] for row in found})
    if missing:
        raise ValueError(f"Reference villages not found with paired vowels: {', '.join(missing)}")
    # Each reference village has exactly the same weight, irrespective of speaker count.
    return {
        "reference_villages": "+".join(names),
        **{field: float(np.mean([row[field] for row in found]))
           for field in ("geo_x", "geo_y", "first_x", "first_y",
                         "second_x", "second_y")},
    }


def wrap_degrees(values: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(values) + 180.0) % 360.0 - 180.0


def add_reference_geometry(villages: list[dict], reference: dict) -> None:
    ref_dx = reference["second_x"] - reference["first_x"]
    ref_dy = reference["second_y"] - reference["first_y"]
    reference["pair_angle_deg"] = math.degrees(math.atan2(ref_dy, ref_dx))
    reference["pair_distance"] = math.hypot(ref_dx, ref_dy)
    for row in villages:
        gx, gy = row["geo_x"] - reference["geo_x"], row["geo_y"] - reference["geo_y"]
        dx, dy = row["second_x"] - row["first_x"], row["second_y"] - row["first_y"]
        angle = math.degrees(math.atan2(dy, dx))
        row.update({
            "geo_dx_from_reference": gx, "geo_dy_from_reference": gy,
            "geo_distance_from_reference": math.hypot(gx, gy),
            "geo_bearing_deg": math.degrees(math.atan2(gx, -gy)) % 360.0,
            "pair_angle_deg": angle,
            "pair_angle_difference_deg": float(wrap_degrees(angle - reference["pair_angle_deg"])),
            "pair_distance": math.hypot(dx, dy),
            "pair_distance_difference": math.hypot(dx, dy) - reference["pair_distance"],
        })


def farthest_centres(points: np.ndarray, count: int) -> np.ndarray:
    """Select reproducible, geographically spread thin-plate basis centres."""
    chosen = [int(np.argmin(np.sum(points ** 2, axis=1)))]
    while len(chosen) < min(count, len(points)):
        distance = np.min(
            np.sum((points[:, None, :] - points[np.asarray(chosen)][None, :, :]) ** 2,
                   axis=2), axis=1
        )
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    return points[np.asarray(chosen)]


def thin_plate(r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, r * r * np.log(r), 0.0)


def model_basis(kind: str, points: np.ndarray, distances: np.ndarray,
                distance_knots: np.ndarray, centres: np.ndarray) -> np.ndarray:
    if kind == "null":
        return np.ones((len(points), 1))
    if kind == "distance":
        return spline_basis(distances, distance_knots)
    x, y = points[:, 0], points[:, 1]
    if kind == "distance_bearing":
        # A quadratic geographic surface: radial change plus direction and curvature.
        return np.column_stack([np.ones(len(points)), x, y, x*x, x*y, y*y])
    if kind == "spatial_2d":
        radius = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        return np.column_stack([np.ones(len(points)), x, y, thin_plate(radius)])
    raise ValueError(kind)


def fit_ridge(design: np.ndarray, response: np.ndarray, penalty: float) -> np.ndarray:
    penalizer = np.eye(design.shape[1])
    penalizer[:min(3, design.shape[1]), :min(3, design.shape[1])] = 0
    return np.linalg.solve(design.T @ design + penalty * penalizer,
                           design.T @ response)


def cross_validate_models(points: np.ndarray, distances: np.ndarray,
                          response: np.ndarray, centres: np.ndarray,
                          distance_knots: np.ndarray) -> tuple[list[dict], dict]:
    kinds = ("null", "distance", "distance_bearing", "spatial_2d")
    penalties = {
        "null": (0.0,), "distance": (0.0,), "distance_bearing": (0.0, .01, .1, 1.0),
        "spatial_2d": (.001, .01, .1, 1.0, 10.0, 100.0, 1000.0),
    }
    scale = np.std(response, axis=0, ddof=1)
    scale[scale == 0] = 1
    standardized = (response - response.mean(axis=0)) / scale
    total = float(np.sum(standardized ** 2))
    rows, fits = [], {}
    for kind in kinds:
        design = model_basis(kind, points, distances, distance_knots, centres)
        best = None
        for penalty in penalties[kind]:
            predictions = np.empty_like(standardized)
            for held_out in range(len(points)):
                train = np.arange(len(points)) != held_out
                beta = fit_ridge(design[train], standardized[train], penalty)
                predictions[held_out] = design[held_out] @ beta
            sse = float(np.sum((standardized - predictions) ** 2))
            if best is None or sse < best[0]:
                best = (sse, penalty, predictions)
        assert best is not None
        sse, penalty, predictions = best
        beta = fit_ridge(design, response, penalty)
        fitted = design @ beta
        training_total = float(np.sum(((response - response.mean(axis=0)) / scale) ** 2))
        training_sse = float(np.sum(((response - fitted) / scale) ** 2))
        rows.append({
            "model": kind, "n_villages": len(points), "n_parameters": design.shape[1],
            "selected_ridge_penalty": penalty,
            "loov_cv_r2_multivariate": 1.0 - sse / total,
            "training_r2_multivariate": 1.0 - training_sse / training_total,
            "loov_rmse_standardized": math.sqrt(sse / standardized.size),
        })
        fits[kind] = {"beta": beta, "penalty": penalty}
    return rows, fits


def predict(kind: str, points: np.ndarray, reference: dict, geo_scale: np.ndarray,
            distance_knots: np.ndarray, centres: np.ndarray, beta: np.ndarray) -> np.ndarray:
    standardized = (points - np.array([reference["geo_x"], reference["geo_y"]])) / geo_scale
    distances = np.linalg.norm(standardized, axis=1)
    return model_basis(kind, standardized, distances, distance_knots, centres) @ beta


def supported_rays(villages: list[dict], reference: dict, count: int = 8) -> list[dict]:
    """Make rays ending at the farthest observed village in each bearing sector."""
    sectors = [[] for _ in range(count)]
    width = 360 / count
    for row in villages:
        index = int(((row["geo_bearing_deg"] + width / 2) % 360) // width)
        sectors[index].append(row)
    rays = []
    for index, members in enumerate(sectors):
        if not members:
            continue
        endpoint = max(members, key=lambda row: row["geo_distance_from_reference"])
        steps = np.linspace(0, 1, 80)
        points = np.column_stack([
            reference["geo_x"] + steps * (endpoint["geo_x"] - reference["geo_x"]),
            reference["geo_y"] + steps * (endpoint["geo_y"] - reference["geo_y"]),
        ])
        rays.append({"bearing": index * width, "endpoint": endpoint["village"],
                     "points": points})
    return rays


def plot_main(path: Path, villages: list[dict], reference: dict, first: str, second: str,
              cv_rows: list[dict], fits: dict, geo_scale: np.ndarray,
              distance_knots: np.ndarray, centres: np.ndarray) -> list[dict]:
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    map_ax, vowel_ax, angle_ax, distance_ax = axes.flat
    cmap = plt.get_cmap("twilight_shifted")
    bearing_norm = Normalize(0, 360)

    gx = np.asarray([row["geo_x"] for row in villages])
    gy = np.asarray([row["geo_y"] for row in villages])
    bearings = np.asarray([row["geo_bearing_deg"] for row in villages])
    map_ax.scatter(gx, gy, c=bearings, cmap=cmap, norm=bearing_norm, s=38,
                   edgecolor="#333", linewidth=.35)
    map_ax.scatter(reference["geo_x"], reference["geo_y"], marker="*", s=220,
                   color="#111", label="Tjällmo–Rimforsa reference", zorder=5)
    for name in reference["reference_villages"].split("+"):
        row = next(item for item in villages if item["village"] == name)
        map_ax.plot([reference["geo_x"], row["geo_x"]],
                    [reference["geo_y"], row["geo_y"]], color="#111", lw=.8)
        map_ax.annotate(name, (row["geo_x"], row["geo_y"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
    rays = supported_rays(villages, reference)
    trajectory_rows = []
    for ray in rays:
        color = cmap(bearing_norm(ray["bearing"]))
        map_ax.plot(ray["points"][:, 0], ray["points"][:, 1], color=color,
                    lw=2, alpha=.8)
        # The bearing-aware quadratic is used for drawn paths.  It is less flexible
        # than the thin-plate surface and therefore avoids visually misleading
        # loops between geographically sparse observations.
        prediction = predict("distance_bearing", ray["points"], reference, geo_scale,
                             distance_knots, centres, fits["distance_bearing"]["beta"])
        for vowel_index, linestyle in ((0, "-"), (1, "--")):
            xy = prediction[:, vowel_index*2:vowel_index*2+2]
            vowel_ax.plot(xy[:, 0], xy[:, 1], color=color, ls=linestyle, lw=2.2)
            vowel_ax.scatter(*xy[-1], color=color, marker="o" if vowel_index == 0 else "s",
                             s=28, zorder=4)
        for step, (geographic, acoustic) in enumerate(zip(ray["points"], prediction)):
            trajectory_rows.append({
                "bearing_deg": ray["bearing"], "endpoint_village": ray["endpoint"],
                "step": step, "geo_x": geographic[0], "geo_y": geographic[1],
                "first_x": acoustic[0], "first_y": acoustic[1],
                "second_x": acoustic[2], "second_y": acoustic[3],
            })
    map_ax.set(title="Geographic reference and supported radial paths",
               xlabel="geographic x", ylabel="geographic y")
    map_ax.invert_yaxis()
    map_ax.set_aspect("equal", adjustable="datalim"); map_ax.grid(alpha=.18); map_ax.legend()

    for row in villages:
        color = cmap(bearing_norm(row["geo_bearing_deg"]))
        vowel_ax.plot([row["first_x"], row["second_x"]],
                      [row["first_y"], row["second_y"]], color=color, alpha=.13, lw=.7)
    vowel_ax.annotate("", xy=(reference["second_x"], reference["second_y"]),
                      xytext=(reference["first_x"], reference["first_y"]),
                      arrowprops={"arrowstyle": "->", "lw": 3, "color": "#111"})
    vowel_ax.scatter(reference["first_x"], reference["first_y"], marker="o", s=85,
                     color="#fff", edgecolor="#111", zorder=6)
    vowel_ax.scatter(reference["second_x"], reference["second_y"], marker="s", s=85,
                     color="#fff", edgecolor="#111", zorder=6)
    vowel_ax.annotate(f"/{first}/ ref", (reference["first_x"], reference["first_y"]),
                      xytext=(5, 5), textcoords="offset points")
    vowel_ax.annotate(f"/{second}/ ref", (reference["second_x"], reference["second_y"]),
                      xytext=(5, 5), textcoords="offset points")
    vowel_ax.plot([], [], color="#444", lw=2.2, label=f"/{first}/ fitted paths")
    vowel_ax.plot([], [], color="#444", lw=2.2, ls="--", label=f"/{second}/ fitted paths")
    drawn = next(row for row in cv_rows if row["model"] == "distance_bearing")
    vowel_ax.set(title="Bearing-aware geographic paths projected into PCA vowel space\n"
                       f"distance + bearing LOOV CV R²={drawn['loov_cv_r2_multivariate']:.2f}",
                 xlabel="−PC1 (score dB)", ylabel="−PC2 (score dB)")
    vowel_ax.set_aspect("equal", adjustable="datalim"); vowel_ax.grid(alpha=.18)
    vowel_ax.legend(fontsize=8)

    angle_values = np.asarray([row["pair_angle_difference_deg"] for row in villages])
    distance_values = np.asarray([row["pair_distance_difference"] for row in villages])
    angle_limit = max(10.0, float(np.nanmax(np.abs(angle_values))))
    distance_limit = max(1.0, float(np.nanmax(np.abs(distance_values))))
    a = angle_ax.scatter(gx, gy, c=angle_values, cmap="coolwarm_r",
                         vmin=-angle_limit, vmax=angle_limit, s=55, edgecolor="#333", lw=.35)
    d = distance_ax.scatter(gx, gy, c=distance_values, cmap="PuOr",
                            vmin=-distance_limit, vmax=distance_limit, s=55,
                            edgecolor="#333", lw=.35)
    for ax in (angle_ax, distance_ax):
        ax.scatter(reference["geo_x"], reference["geo_y"], marker="*", s=190,
                   color="#111", zorder=5)
        ax.set(xlabel="geographic x", ylabel="geographic y")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.18)
    angle_ax.set_title(f"Village /{first}–{second}/ angle minus reference angle")
    distance_ax.set_title(f"Village /{first}–{second}/ distance minus reference distance")
    fig.colorbar(a, ax=angle_ax, label="directed angle difference (degrees)")
    fig.colorbar(d, ax=distance_ax, label="pair-distance difference (PC-score dB)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return trajectory_rows


def plot_cv(path: Path, rows: list[dict]) -> None:
    labels = {"null": "constant", "distance": "distance only",
              "distance_bearing": "distance + bearing", "spatial_2d": "smooth 2-D"}
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    outcomes = list(dict.fromkeys(row["outcome"] for row in rows))
    models = ("null", "distance", "distance_bearing", "spatial_2d")
    x = np.arange(len(models), dtype=float)
    width = .8 / len(outcomes)
    colors = ("#4c78a8", "#e45756", "#72b7b2")
    all_values = []
    for index, outcome in enumerate(outcomes):
        lookup = {row["model"]: row for row in rows if row["outcome"] == outcome}
        values = [lookup[model]["loov_cv_r2_multivariate"] for model in models]
        positions = x + (index - (len(outcomes)-1)/2) * width
        bars = ax.bar(positions, values, width=width, color=colors[index], label=outcome)
        all_values.extend(values)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value,
                    f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top",
                    fontsize=7, rotation=90)
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xticks(x, [labels[model] for model in models])
    ax.set(ylabel="leave-one-village-out CV R²",
           title="Reference-centred geographic model comparisons")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--first", choices=VOWELS, default="uː")
    parser.add_argument("--second", choices=VOWELS, default="oː")
    parser.add_argument("--reference-villages", nargs="+", default=["tjallmo", "rimforsa"])
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--include-finland-gotland", action="store_true",
                        help="Retain Gotland and the four Finland regions; excluded by default.")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--include-reference", action="store_true",
                        help="Include the synthetic 'ref' speaker (normally excluded).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("Analyses/PCA_pair_reference_geography_u_o"))
    args = parser.parse_args()
    if args.first == args.second:
        parser.error("--first and --second must differ")

    tokens = read_tokens(args.input, "pca", args.include_incomplete, False)
    if not args.include_reference:
        tokens = [row for row in tokens if row["village"] != "ref"]
    speakers = pair_speakers(speaker_vowels(tokens), args.first, args.second)
    excluded_villages = []
    if not args.include_finland_gotland:
        regions = resource_regions(args.resource)
        unknown = sorted({row["village"] for row in speakers
                          if RESOURCE_ALIASES.get(row["village"], row["village"]) not in regions})
        if unknown:
            raise ValueError(f"Villages missing from {args.resource}: {', '.join(unknown)}")
        excluded_villages = sorted({
            row["village"] for row in speakers
            if regions[RESOURCE_ALIASES.get(row["village"], row["village"])]
            in DEFAULT_EXCLUDED_REGIONS
        })
        speakers = [row for row in speakers if row["village"] not in excluded_villages]
    villages = village_means(speakers)
    reference = reference_values(villages, args.reference_villages)
    add_reference_geometry(villages, reference)

    raw_geo = np.asarray([[row["geo_x"], row["geo_y"]] for row in villages])
    origin = np.asarray([reference["geo_x"], reference["geo_y"]])
    geo_scale = np.std(raw_geo - origin, axis=0, ddof=1)
    points = (raw_geo - origin) / geo_scale
    distances = np.linalg.norm(points, axis=1)
    distance_knots = np.quantile(distances, [0, .25, .75, 1])
    centres = farthest_centres(points, 14)
    responses = np.asarray([[row[field] for field in
                             ("first_x", "first_y", "second_x", "second_y")]
                            for row in villages])
    endpoint_cv, fits = cross_validate_models(points, distances, responses, centres,
                                               distance_knots)
    angle_response = np.asarray([[row["pair_angle_difference_deg"]] for row in villages])
    distance_response = np.asarray([[row["pair_distance_difference"]] for row in villages])
    angle_cv, _ = cross_validate_models(points, distances, angle_response, centres,
                                         distance_knots)
    pair_distance_cv, _ = cross_validate_models(points, distances, distance_response,
                                                 centres, distance_knots)
    for label, rows in (("vowel endpoints", endpoint_cv), ("pair angle", angle_cv),
                        ("pair distance", pair_distance_cv)):
        for row in rows:
            row["outcome"] = label
    cv_rows = endpoint_cv + angle_cv + pair_distance_cv

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "speaker_pair_positions.csv", speakers)
    write_csv(args.output_dir / "village_pair_reference_geometry.csv", villages)
    write_csv(args.output_dir / "reference_definition.csv", [reference])
    write_csv(args.output_dir / "excluded_villages.csv", [
        {"village": village,
         "region": resource_regions(args.resource)[RESOURCE_ALIASES.get(village, village)]}
        for village in excluded_villages
    ])
    write_csv(args.output_dir / "geographic_model_comparison.csv", cv_rows)
    trajectories = plot_main(
        args.output_dir / "reference_geography_and_acoustic_paths.png", villages,
        reference, args.first, args.second, endpoint_cv, fits, geo_scale,
        distance_knots, centres,
    )
    write_csv(args.output_dir / "fitted_spatial_ray_trajectories.csv", trajectories)
    plot_cv(args.output_dir / "geographic_model_cross_validation.png", cv_rows)
    print(f"Reference: {reference['reference_villages']} at "
          f"({reference['geo_x']:.1f}, {reference['geo_y']:.1f})")
    print("Excluded Gotland/Finland villages: " +
          (", ".join(excluded_villages) if excluded_villages else "none"))
    print(f"Wrote {len(speakers)} speaker pairs and {len(villages)} villages to "
          f"{args.output_dir}")
    for row in cv_rows:
        print(f"  {row['outcome']} / {row['model']}: "
              f"LOOV CV R2={row['loov_cv_r2_multivariate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
