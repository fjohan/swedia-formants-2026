#!/usr/bin/env python3
"""Run comparable 2-D spatial GAMs for vowels and vowel-pair midpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-spatial-vowels-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
import numpy as np
from scipy.spatial import ConvexHull

from model_u_o_spatial_midpoint_gam import (
    fit_surface, grouped_cv, permutation_tests, write_csv,
)
from model_vowel_axis_trajectories import DEFAULT_EXCLUDED_REGIONS, FILE_LABELS
from plot_midpoint_vowel_spaces import PAIRS, VOWELS, read_tokens, speaker_vowels
from plot_regional_pair_midpoint_summary import ALIASES, resource_provinces


METHOD_LABELS = {
    "pca": "spectral PCA",
    "praat-fixed": "fixed-Praat formants",
    "fasttrack": "FastTrack formants",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--output", type=Path,
                        default=Path("Analyses/Spatial_vowel_midpoint_GAMs"))
    parser.add_argument("--methods", nargs="+",
                        choices=("pca", "praat-fixed", "fasttrack"),
                        default=["pca", "praat-fixed", "fasttrack"])
    parser.add_argument("--targets", choices=("individual", "midpoints", "all"),
                        default="all")
    parser.add_argument("--vowels", nargs="+", choices=VOWELS, default=list(VOWELS))
    parser.add_argument(
        "--pairs", nargs="+", default=[f"{a},{b}" for a, b in PAIRS],
        help="Comma-separated vowel pairs, for example 'uː,oː'.",
    )
    parser.add_argument("--include-finland-gotland", action="store_true")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--grid-size", type=int, default=85)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    parsed_pairs = []
    for value in args.pairs:
        fields = value.split(",")
        if len(fields) != 2 or any(vowel not in VOWELS for vowel in fields):
            parser.error(f"Invalid pair {value!r}; use two comma-separated vowel symbols")
        if fields[0] == fields[1]:
            parser.error(f"Pair members must differ: {value!r}")
        parsed_pairs.append(tuple(fields))
    args.pairs = parsed_pairs
    if args.permutations < 99:
        parser.error("--permutations must be at least 99")
    return args


def individual_rows(speakers: list[dict], vowel: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in speakers:
        if row["vowel"] == vowel:
            grouped[row["village"]].append(row)
    output = []
    for village, values in sorted(grouped.items()):
        output.append({
            "village": village, "geo_x": values[0]["geo_x"], "geo_y": values[0]["geo_y"],
            "acoustic_x": float(np.median([row["display_x"] for row in values])),
            "acoustic_y": float(np.median([row["display_y"] for row in values])),
            "n_speakers": len(values),
        })
    return output


def midpoint_rows(speakers: list[dict], first: str, second: str) -> list[dict]:
    by_speaker: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in speakers:
        if row["vowel"] in {first, second}:
            by_speaker[row["village"], row["speaker"]][row["vowel"]] = row
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    coordinates = {}
    for (village, _), values in by_speaker.items():
        if first not in values or second not in values:
            continue
        grouped[village].append((
            (values[first]["display_x"] + values[second]["display_x"]) / 2,
            (values[first]["display_y"] + values[second]["display_y"]) / 2,
        ))
        coordinates[village] = (values[first]["geo_x"], values[first]["geo_y"])
    return [{
        "village": village, "geo_x": coordinates[village][0], "geo_y": coordinates[village][1],
        "acoustic_x": float(np.median(np.asarray(values)[:, 0])),
        "acoustic_y": float(np.median(np.asarray(values)[:, 1])),
        "n_speakers": len(values),
    } for village, values in sorted(grouped.items())]


def labels_for(method: str) -> tuple[str, str]:
    if method == "pca":
        return "−PC1 (score dB)", "−PC2 (score dB)"
    return "−F2 (Hz)", "−F1 (Hz)"


def analyze_target(
    rows: list[dict], method: str, target_type: str, target: str,
    output: Path, permutations: int, grid_size: int, seed: int,
) -> dict:
    X = np.asarray([[row["geo_x"], row["geo_y"]] for row in rows])
    responses = np.asarray([[row["acoustic_x"], row["acoustic_y"]] for row in rows])
    models = [fit_surface(X, responses[:, coordinate]) for coordinate in range(2)]
    p_values, joint_p, improvements, joint_statistic = permutation_tests(
        models, X, responses, permutations, np.random.default_rng(seed)
    )
    cv_r2 = grouped_cv(X, responses, seed + 1)
    explained = [float(model.statistics_["pseudo_r2"]["explained_deviance"])
                 for model in models]

    gx = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_size)
    gy = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    hull = ConvexHull(X)
    supported = PlotPath(X[hull.vertices]).contains_points(grid)
    grid_predictions = np.column_stack([model.predict(grid) for model in models])
    grid_predictions[~supported] = np.nan
    fitted = np.column_stack([model.predict(X) for model in models])

    output.mkdir(parents=True, exist_ok=True)
    for row, prediction in zip(rows, fitted):
        row["fitted_acoustic_x"] = float(prediction[0])
        row["fitted_acoustic_y"] = float(prediction[1])
    write_csv(output / "village_predictions.csv", rows)
    write_csv(output / "surface.csv", [{
        "geo_x": point[0], "geo_y": point[1],
        "fitted_acoustic_x": prediction[0], "fitted_acoustic_y": prediction[1],
    } for point, prediction, keep in zip(grid, grid_predictions, supported) if keep])

    x_label, y_label = labels_for(method)
    coordinate_names = (x_label.replace(" (score dB)", "").replace(" (Hz)", ""),
                        y_label.replace(" (score dB)", "").replace(" (Hz)", ""))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    for coordinate, ax in enumerate(axes):
        zz = grid_predictions[:, coordinate].reshape(xx.shape)
        contour = ax.contourf(xx, yy, zz, levels=18, cmap="viridis")
        ax.contour(xx, yy, zz, levels=9, colors="black", linewidths=.5, alpha=.6)
        ax.scatter(X[:, 0], X[:, 1], s=18, color="white", edgecolor="black", linewidth=.4)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="geo_x", title=f"Fitted {coordinate_names[coordinate]}")
        fig.colorbar(contour, ax=ax, shrink=.68,
                     label="PCA score dB" if method == "pca" else "Hz")
    axes[0].set_ylabel("geo_y (larger values downward)")
    axes[0].invert_yaxis()
    fig.suptitle(f"{target}: continuous spatial GAM — {METHOD_LABELS[method]}")
    fig.savefig(output / "geographic_surfaces.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.scatter(responses[:, 0], responses[:, 1], s=24, color="0.72", alpha=.55,
               label="observed village value")
    fitted_scatter = ax.scatter(
        fitted[:, 0], fitted[:, 1], s=30, c=X[:, 1], cmap="coolwarm",
        edgecolor="white", linewidth=.35, label="spatial-GAM fitted value"
    )
    for observed, prediction in zip(responses, fitted):
        ax.plot([observed[0], prediction[0]], [observed[1], prediction[1]],
                color="0.45", lw=.4, alpha=.22)
    ax.set(xlabel=x_label, ylabel=y_label,
           title=f"{target}: observed and spatially fitted values")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.15); ax.legend()
    colorbar = fig.colorbar(fitted_scatter, ax=ax, shrink=.78)
    colorbar.set_label("geo_y (north/blue → south/red)")
    fig.savefig(output / "fitted_acoustic_space.png", dpi=180)
    plt.close(fig)

    summary = {
        "method": method, "target_type": target_type, "target": target,
        "n_villages": len(rows),
        "x_explained_deviance": explained[0], "x_cv_r2_10fold": cv_r2[0],
        "x_permutation_p": p_values[0],
        "y_explained_deviance": explained[1], "y_cv_r2_10fold": cv_r2[1],
        "y_permutation_p": p_values[1], "joint_permutation_p": joint_p,
        "joint_statistic": joint_statistic,
        "x_rss_improvement": improvements[0], "y_rss_improvement": improvements[1],
        "directory": str(output),
    }
    write_csv(output / "model_statistics.csv", [summary])
    return summary


def main() -> int:
    args = arguments()
    provinces = resource_provinces(args.resource)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    target_index = 0
    for method in args.methods:
        tokens = [row for row in read_tokens(args.input, method, False, False)
                  if row["village"] != "ref"]
        excluded = sorted({
            row["village"] for row in tokens
            if not args.include_finland_gotland
            and provinces.get(ALIASES.get(row["village"], row["village"]))
            in DEFAULT_EXCLUDED_REGIONS
        })
        tokens = [row for row in tokens if row["village"] not in excluded]
        speakers = speaker_vowels(tokens)
        method_dir = args.output / method.replace("-", "_")
        if args.targets in {"individual", "all"}:
            for vowel in args.vowels:
                rows = individual_rows(speakers, vowel)
                summaries.append(analyze_target(
                    rows, method, "vowel", vowel,
                    method_dir / "vowels" / FILE_LABELS[vowel],
                    args.permutations, args.grid_size, args.seed + target_index * 20,
                ))
                target_index += 1
        if args.targets in {"midpoints", "all"}:
            for first, second in args.pairs:
                rows = midpoint_rows(speakers, first, second)
                pair_slug = f"{FILE_LABELS[first]}_{FILE_LABELS[second]}"
                summaries.append(analyze_target(
                    rows, method, "midpoint", f"{first}→{second}",
                    method_dir / "midpoints" / pair_slug,
                    args.permutations, args.grid_size, args.seed + target_index * 20,
                ))
                target_index += 1
    write_csv(args.output / "all_model_statistics.csv", summaries)
    settings = {
        "methods": args.methods, "targets": args.targets, "vowels": args.vowels,
        "pairs": args.pairs, "permutations": args.permutations,
        "model": "acoustic coordinate ~ te(geo_x, geo_y)",
        "aggregation": "village median of speaker values; paired speaker midpoints",
        "excluded_regions": [] if args.include_finland_gotland else sorted(DEFAULT_EXCLUDED_REGIONS),
    }
    (args.output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows_html = "\n".join(
        f'<tr><td>{row["method"]}</td><td>{row["target_type"]}</td>'
        f'<td>{row["target"]}</td><td>{row["x_explained_deviance"]:.2f}</td>'
        f'<td>{row["x_cv_r2_10fold"]:.2f}</td><td>{row["y_explained_deviance"]:.2f}</td>'
        f'<td>{row["y_cv_r2_10fold"]:.2f}</td><td>{row["joint_permutation_p"]:.4g}</td>'
        f'<td><a href="{Path(row["directory"]).relative_to(args.output)}/geographic_surfaces.png">maps</a></td></tr>'
        for row in summaries
    )
    (args.output / "index.html").write_text(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Spatial vowel and midpoint GAMs</title><style>
body{{font:15px/1.5 system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.35rem .5rem;text-align:right}}
th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}</style></head><body>
<h1>Comparable spatial GAMs for vowels and vowel-pair midpoints</h1>
<p>Every row applies the same analysis to one target: separate joint-geographic GAMs for its
two acoustic coordinates, village-level permutation tests, and 10-fold village cross-validation.
PCA and formant coordinates have different scales and meanings, so compare geographic structure,
explained deviance, and predictive performance rather than raw effect magnitudes.</p>
<table><thead><tr><th>Method</th><th>Type</th><th>Target</th><th>X R²</th><th>X CV R²</th>
<th>Y R²</th><th>Y CV R²</th><th>Joint p</th><th>Figures</th></tr></thead><tbody>
{rows_html}</tbody></table><p><a href="all_model_statistics.csv">All statistics</a> ·
<a href="settings.json">Settings</a></p></body></html>""", encoding="utf-8")
    print(f"Wrote {len(summaries)} analyses to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
