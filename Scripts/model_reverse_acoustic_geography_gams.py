#!/usr/bin/env python3
"""Project geographic coordinates over vowel and vowel-pair acoustic spaces."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-reverse-geography-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
import numpy as np
from scipy.spatial import ConvexHull

from model_u_o_spatial_midpoint_gam import fit_surface, fixed_smoother
from model_vowel_axis_trajectories import (
    DEFAULT_EXCLUDED_REGIONS, FILE_LABELS, geographic_axis,
)
from plot_midpoint_vowel_spaces import PAIRS, VOWELS, read_tokens, speaker_vowels
from plot_regional_pair_midpoint_summary import ALIASES, resource_provinces


METHOD_LABELS = {
    "pca": "spectral PCA", "praat-fixed": "fixed-Praat formants",
    "fasttrack": "FastTrack formants",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--output", type=Path,
                        default=Path("Analyses/Reverse_acoustic_geography_GAMs"))
    parser.add_argument("--methods", nargs="+",
                        choices=("pca", "praat-fixed", "fasttrack"),
                        default=["pca", "praat-fixed", "fasttrack"])
    parser.add_argument("--targets", choices=("individual", "pairs", "whole-space", "all"),
                        default="all")
    parser.add_argument("--geographic-responses", nargs="+",
                        choices=("geo-x", "geo-y", "axis"), default=["geo-x", "geo-y"])
    parser.add_argument("--vowels", nargs="+", choices=VOWELS, default=list(VOWELS))
    parser.add_argument("--pairs", nargs="+", default=[f"{a},{b}" for a, b in PAIRS])
    parser.add_argument("--south-endpoint", default="loderup")
    parser.add_argument("--north-endpoint", default="arjeplog")
    parser.add_argument("--include-finland-gotland", action="store_true")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    parsed_pairs = []
    for value in args.pairs:
        fields = value.split(",")
        if len(fields) != 2 or any(vowel not in VOWELS for vowel in fields):
            parser.error(f"Invalid pair {value!r}; use two comma-separated vowel symbols")
        parsed_pairs.append(tuple(fields))
    args.pairs = parsed_pairs
    if args.permutations < 99:
        parser.error("--permutations must be at least 99")
    return args


def village_vowels(speakers: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in speakers:
        grouped[row["village"], row["vowel"]].append(row)
    return {(village, vowel): {
        "village": village, "vowel": vowel,
        "geo_x": values[0]["geo_x"], "geo_y": values[0]["geo_y"],
        "acoustic_x": float(np.median([row["display_x"] for row in values])),
        "acoustic_y": float(np.median([row["display_y"] for row in values])),
        "n_speakers": len(values),
    } for (village, vowel), values in grouped.items()}


def target_data(
    values: dict[tuple[str, str], dict], vowels: tuple[str, ...],
    axis_positions: dict[str, float],
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    villages = sorted({village for village, vowel in values if all(
        (village, requested) in values for requested in vowels
    )})
    rows, points, group_ids = [], [], []
    supports = []
    for vowel in vowels:
        vowel_points = []
        for group_id, village in enumerate(villages):
            row = values[village, vowel].copy()
            row["axis"] = axis_positions[village]
            row["group_id"] = group_id
            rows.append(row)
            point = [row["acoustic_x"], row["acoustic_y"]]
            points.append(point); vowel_points.append(point); group_ids.append(group_id)
        supports.append(np.asarray(vowel_points))
    village_geography = np.asarray([
        [values[village, vowels[0]]["geo_x"], values[village, vowels[0]]["geo_y"],
         axis_positions[village]] for village in villages
    ])
    return rows, np.asarray(points), np.asarray(group_ids), village_geography, supports


def response_spec(name: str, values: np.ndarray) -> dict:
    if name == "geo-x":
        return {"column": 0, "label": "geo_x", "cmap": "viridis",
                "low": "Low geo_x", "high": "High geo_x"}
    if name == "geo-y":
        return {"column": 1, "label": "geo_y", "cmap": "coolwarm",
                "low": "Low geo_y · north", "high": "High geo_y · south"}
    return {"column": 2, "label": "Löderup–Arjeplog axis", "cmap": "coolwarm_r",
            "low": "Löderup · south", "high": "Arjeplog · north"}


def permutation_p(
    model, X: np.ndarray, response: np.ndarray, group_ids: np.ndarray,
    village_response: np.ndarray, iterations: int, rng: np.random.Generator,
) -> tuple[float, float]:
    smoother = fixed_smoother(model, X)
    null_rss = float(np.sum((response - response.mean()) ** 2))
    observed = null_rss - float(np.sum((response - smoother @ response) ** 2))
    exceed = 0
    for _ in range(iterations):
        permuted = village_response[rng.permutation(len(village_response))][group_ids]
        statistic = null_rss - float(np.sum((permuted - smoother @ permuted) ** 2))
        exceed += int(statistic >= observed - 1e-12)
    return (exceed + 1) / (iterations + 1), observed


def grouped_cv(
    X: np.ndarray, response: np.ndarray, group_ids: np.ndarray,
    n_groups: int, seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n_groups), 10)
    predicted = np.empty(len(response))
    for test_groups in folds:
        test = np.isin(group_ids, test_groups)
        model = fit_surface(X[~test], response[~test])
        predicted[test] = model.predict(X[test])
    total = float(np.sum((response - response.mean()) ** 2))
    return 1 - float(np.sum((response - predicted) ** 2)) / total


def acoustic_labels(method: str) -> tuple[str, str]:
    return ("−PC1 (score dB)", "−PC2 (score dB)") if method == "pca" else (
        "−F2 (Hz)", "−F1 (Hz)"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze_target(
    rows: list[dict], X: np.ndarray, group_ids: np.ndarray,
    village_geography: np.ndarray, supports: list[np.ndarray], method: str,
    target_type: str, target: str, response_names: list[str], output: Path,
    permutations: int, grid_size: int, seed: int,
) -> list[dict]:
    gx = np.linspace(X[:, 0].min(), X[:, 0].max(), grid_size)
    gy = np.linspace(X[:, 1].min(), X[:, 1].max(), grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    supported = np.zeros(len(grid), dtype=bool)
    for points in supports:
        hull = ConvexHull(points)
        supported |= PlotPath(points[hull.vertices]).contains_points(grid)

    output.mkdir(parents=True, exist_ok=True)
    x_label, y_label = acoustic_labels(method)
    fig, axes = plt.subplots(1, len(response_names), figsize=(7 * len(response_names), 6.5),
                             squeeze=False, constrained_layout=True)
    summaries, surface_columns = [], {"acoustic_x": grid[:, 0], "acoustic_y": grid[:, 1]}
    for index, response_name in enumerate(response_names):
        spec = response_spec(response_name, village_geography)
        village_response = village_geography[:, spec["column"]]
        response = village_response[group_ids]
        model = fit_surface(X, response)
        p_value, statistic = permutation_p(
            model, X, response, group_ids, village_response, permutations,
            np.random.default_rng(seed + index),
        )
        cv_r2 = grouped_cv(X, response, group_ids, len(village_response), seed + index + 10)
        prediction = model.predict(grid)
        prediction[~supported] = np.nan
        surface_columns[f"fitted_{response_name.replace('-', '_')}"] = prediction
        levels = np.linspace(village_response.min(), village_response.max(), 17)
        ax = axes[0, index]
        zz = prediction.reshape(xx.shape)
        contour = ax.contourf(xx, yy, zz, levels=levels, cmap=spec["cmap"], extend="both")
        ax.contour(xx, yy, zz, levels=levels[::2], colors="black", linewidths=.5, alpha=.6)
        if len(supports) == 2:
            for first, second in zip(supports[0], supports[1]):
                ax.plot([first[0], second[0]], [first[1], second[1]],
                        color="0.25", lw=.4, alpha=.14)
        markers = ("o", "s", "^", "D", "P", "X", "v", "h")
        for vowel_index, points in enumerate(supports):
            ax.scatter(points[:, 0], points[:, 1], c=village_response,
                       cmap=spec["cmap"], vmin=village_response.min(),
                       vmax=village_response.max(), marker=markers[vowel_index], s=29,
                       edgecolors="white", linewidths=.35,
                       label=(rows[vowel_index * len(village_response)]["vowel"]
                              if len(supports) > 1 else None))
        ax.set(xlabel=x_label, ylabel=y_label,
               title=f'Fitted {spec["label"]}\nCV R²={cv_r2:.2f}, permutation p={p_value:.4g}')
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.1)
        if len(supports) > 1:
            ax.legend(title="vowel", fontsize=8, ncol=2 if len(supports) > 2 else 1)
        colorbar = fig.colorbar(contour, ax=ax, shrink=.78)
        colorbar.set_ticks([village_response.min(), village_response.max()])
        colorbar.set_ticklabels([spec["low"], spec["high"]])
        summaries.append({
            "method": method, "target_type": target_type, "target": target,
            "geographic_response": response_name,
            "n_villages": len(village_response), "n_acoustic_points": len(X),
            "explained_deviance": float(
                model.statistics_["pseudo_r2"]["explained_deviance"]
            ),
            "edof": float(model.statistics_["edof"]), "cv_r2_10fold": cv_r2,
            "permutation_p": p_value, "rss_improvement": statistic,
            "directory": str(output),
        })
    fig.suptitle(f"{target}: geography projected over {METHOD_LABELS[method]} space")
    fig.savefig(output / "reverse_geography_surfaces.png", dpi=180)
    plt.close(fig)

    surface_rows = []
    for point_index in np.flatnonzero(supported):
        surface_rows.append({key: values[point_index] for key, values in surface_columns.items()})
    write_csv(output / "surface.csv", surface_rows)
    write_csv(output / "model_statistics.csv", summaries)
    write_csv(output / "village_acoustic_points.csv", rows)
    return summaries


def main() -> int:
    args = arguments()
    provinces = resource_provinces(args.resource)
    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
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
        geography = geographic_axis(
            speakers, args.south_endpoint, args.north_endpoint,
            ["tjallmo", "rimforsa"], 35.0, "axis", "band",
        )
        values = village_vowels(speakers)
        method_dir = args.output / method.replace("-", "_")
        if args.targets in {"individual", "all"}:
            for vowel in args.vowels:
                data = target_data(values, (vowel,), geography["positions"])
                all_summaries.extend(analyze_target(
                    *data, method, "vowel", vowel, args.geographic_responses,
                    method_dir / "vowels" / FILE_LABELS[vowel],
                    args.permutations, args.grid_size, args.seed + target_index * 30,
                ))
                target_index += 1
        if args.targets in {"pairs", "all"}:
            for first, second in args.pairs:
                data = target_data(values, (first, second), geography["positions"])
                slug = f"{FILE_LABELS[first]}_{FILE_LABELS[second]}"
                all_summaries.extend(analyze_target(
                    *data, method, "shared_pair_space", f"{first}+{second}",
                    args.geographic_responses, method_dir / "pairs" / slug,
                    args.permutations, args.grid_size, args.seed + target_index * 30,
                ))
                target_index += 1
        if args.targets in {"whole-space", "all"}:
            data = target_data(values, tuple(args.vowels), geography["positions"])
            all_summaries.extend(analyze_target(
                *data, method, "whole_vowel_space", "all vowels",
                args.geographic_responses, method_dir / "whole_vowel_space",
                args.permutations, args.grid_size, args.seed + target_index * 30,
            ))
            target_index += 1
    write_csv(args.output / "all_model_statistics.csv", all_summaries)
    settings = {
        "methods": args.methods, "targets": args.targets,
        "geographic_responses": args.geographic_responses,
        "model": "one geographic response ~ te(acoustic_x, acoustic_y)",
        "pair_definition": "both vowels occupy one shared acoustic predictor plane",
        "whole_space_definition": "all selected vowels occupy one shared acoustic predictor plane",
        "cross_validation": "10-fold, grouped by village for shared pair spaces",
        "permutations": args.permutations,
        "excluded_regions": [] if args.include_finland_gotland else sorted(DEFAULT_EXCLUDED_REGIONS),
    }
    (args.output / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows_html = "\n".join(
        f'<tr><td>{row["method"]}</td><td>{row["target_type"]}</td><td>{row["target"]}</td>'
        f'<td>{row["geographic_response"]}</td><td>{row["explained_deviance"]:.2f}</td>'
        f'<td>{row["cv_r2_10fold"]:.2f}</td><td>{row["permutation_p"]:.4g}</td>'
        f'<td><a href="{Path(row["directory"]).relative_to(args.output)}/reverse_geography_surfaces.png">map</a></td></tr>'
        for row in all_summaries
    )
    (args.output / "index.html").write_text(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Reverse acoustic-geography GAMs</title><style>
body{{font:15px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.35rem .5rem;text-align:right}}
th:nth-child(-n+4),td:nth-child(-n+4){{text-align:left}}</style></head><body>
<h1>Geography projected over vowel and vowel-pair acoustic spaces</h1>
<p>Each model uses one geographic response and a joint two-dimensional acoustic smooth. Individual
vowels receive their own acoustic map. Pair maps place both vowels in one common predictor plane,
and whole-space maps place all eight vowels in that plane; vowel identity affects marker shape
only. Cross-validation and permutations keep every observation from the same village together.</p>
<table><thead><tr><th>Method</th><th>Type</th>
<th>Target</th><th>Response</th><th>R²</th><th>CV R²</th><th>p</th><th>Figure</th></tr></thead>
<tbody>{rows_html}</tbody></table><p><a href="all_model_statistics.csv">All statistics</a> ·
<a href="settings.json">Settings</a></p></body></html>""", encoding="utf-8")
    print(f"Wrote {len(all_summaries)} reverse-projection models to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
