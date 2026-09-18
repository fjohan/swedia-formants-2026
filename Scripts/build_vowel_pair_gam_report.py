#!/usr/bin/env python3
"""Build a focused spatial-GAM report for any directed vowel pair."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-u-o-gam-report-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from model_spatial_vowel_midpoint_gams import (
    METHOD_LABELS, analyze_target as analyze_forward_target, individual_rows,
    labels_for, midpoint_rows,
)
from model_reverse_acoustic_geography_gams import (
    analyze_target as analyze_reverse_target, target_data as reverse_target_data,
    village_vowels,
)
from model_vowel_axis_trajectories import DEFAULT_EXCLUDED_REGIONS, FILE_LABELS
from plot_midpoint_vowel_spaces import PAIRS, VOWELS, read_tokens, speaker_vowels
from plot_regional_pair_midpoint_summary import ALIASES, resource_provinces


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--method", choices=("pca", "praat-fixed", "fasttrack"),
                        default="pca")
    parser.add_argument("--bark", action="store_true",
                        help="Use Bark-transformed formants with Praat or FastTrack.")
    parser.add_argument("--timepoint", type=int, choices=(20, 50, 80), default=50,
                        help="Percent of vowel duration to analyze (default: 50).")
    pair_aliases = "; ".join(
        f"{index}={first}→{second}"
        for index, (first, second) in enumerate(PAIRS, start=1)
    )
    parser.add_argument(
        "--pair", default="1",
        help=("Canonical pair number 1–6 or a directed comma-separated pair "
              f"such as 'uː,oː'. Aliases: {pair_aliases}."),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--grid-size", type=int, default=85)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--include-finland-gotland", action="store_true")
    args = parser.parse_args()
    if args.permutations < 99:
        parser.error("--permutations must be at least 99")
    if args.bark and args.method == "pca":
        parser.error("--bark is available only with --method praat-fixed or fasttrack")
    if args.pair in {str(index) for index in range(1, len(PAIRS) + 1)}:
        pair = PAIRS[int(args.pair) - 1]
    else:
        pair = tuple(value.strip() for value in args.pair.split(","))
    if len(pair) != 2 or any(value not in VOWELS for value in pair):
        parser.error(
            f"--pair must be 1–{len(PAIRS)} or contain two of: {', '.join(VOWELS)}"
        )
    if pair[0] == pair[1]:
        parser.error("--pair members must differ")
    args.pair = pair
    pair_slug = f"{FILE_LABELS[pair[0]]}_{FILE_LABELS[pair[1]]}"
    method_slug = "PCA" if args.method == "pca" else args.method.replace("-", "_")
    if args.bark:
        method_slug += "_bark"
    if args.timepoint != 50:
        method_slug += f"_t{args.timepoint}"
    if args.output is None:
        args.output = Path("Analyses") / f"{pair_slug}_{method_slug}_GAM_report"
    return args


def paired_vector_rows(speakers: list[dict], first: str, second: str) -> list[dict]:
    """Village medians of within-speaker directed vectors and endpoints."""
    by_speaker: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in speakers:
        if row["vowel"] in {first, second}:
            by_speaker[row["village"], row["speaker"]][row["vowel"]] = row
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    coordinates = {}
    for (village, _), values in by_speaker.items():
        if not {first, second}.issubset(values):
            continue
        start = np.asarray([values[first]["display_x"], values[first]["display_y"]])
        end = np.asarray([values[second]["display_x"], values[second]["display_y"]])
        grouped[village].append(np.r_[end - start, start, end])
        coordinates[village] = (values[first]["geo_x"], values[first]["geo_y"])
    rows = []
    for village, values in sorted(grouped.items()):
        med = np.median(values, axis=0)
        rows.append({
            "village": village, "geo_x": coordinates[village][0],
            "geo_y": coordinates[village][1], "acoustic_x": float(med[0]),
            "acoustic_y": float(med[1]), "start_x": float(med[2]),
            "start_y": float(med[3]), "end_x": float(med[4]), "end_y": float(med[5]),
            "n_speakers": len(values),
        })
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pair_figures(output: Path, first: str, second: str, method: str) -> None:
    vector_rows = read_csv(output / "pair_vector" / "village_predictions.csv")
    midpoint_rows_data = read_csv(output / "midpoint" / "village_predictions.csv")
    midpoint_by_village = {row["village"]: row for row in midpoint_rows_data}
    geo_y = np.asarray([float(row["geo_y"]) for row in vector_rows])
    norm = Normalize(float(geo_y.min()), float(geo_y.max()))
    cmap = plt.get_cmap("coolwarm")

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    for row in vector_rows:
        ax.plot([float(row["start_x"]), float(row["end_x"])],
                [float(row["start_y"]), float(row["end_y"])],
                color="0.7", lw=.55, alpha=.32, zorder=1)
        midpoint = midpoint_by_village[row["village"]]
        mx = float(midpoint["fitted_acoustic_x"])
        my = float(midpoint["fitted_acoustic_y"])
        dx = float(row["fitted_acoustic_x"])
        dy = float(row["fitted_acoustic_y"])
        color = cmap(norm(float(row["geo_y"])))
        ax.annotate("", xy=(mx + dx / 2, my + dy / 2),
                    xytext=(mx - dx / 2, my - dy / 2),
                    arrowprops={"arrowstyle": "->", "color": color,
                                "lw": 1.15, "alpha": .78}, zorder=3)
    ax.scatter([], [], color="0.7", label="observed paired village vector")
    ax.scatter([], [], color=cmap(.75), label="GAM-fitted vector at fitted midpoint")
    x_label, y_label = labels_for(method)
    ax.set(xlabel=x_label, ylabel=y_label,
           title=f"Observed and spatial-GAM fitted /{first}→{second}/ vectors")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.14); ax.legend()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=.76, label="geo_y (north/blue → south/red)")
    fig.savefig(output / "pair_vectors_in_acoustic_space.png", dpi=190)
    plt.close(fig)

    surface = read_csv(output / "pair_vector" / "surface.csv")
    gx = np.asarray([float(row["geo_x"]) for row in surface])
    gy = np.asarray([float(row["geo_y"]) for row in surface])
    dx = np.asarray([float(row["fitted_acoustic_x"]) for row in surface])
    dy = np.asarray([float(row["fitted_acoustic_y"]) for row in surface])
    derived = (np.degrees(np.arctan2(dy, dx)), np.hypot(dx, dy))
    titles = (f"Fitted /{first}→{second}/ direction",
              f"Fitted /{first}→{second}/ distance")
    distance_unit = ("PCA distance (score dB)" if method == "pca" else
                     "formant distance (Bark)" if method in {"praat-bark", "fasttrack-bark"}
                     else "formant distance (Hz)")
    labels = ("angle (degrees)", distance_unit)
    cmaps = ("twilight", "viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, values, title, label, color_map in zip(axes, derived, titles, labels, cmaps):
        contour = ax.tricontourf(gx, gy, values, levels=18, cmap=color_map)
        ax.tricontour(gx, gy, values, levels=9, colors="black", linewidths=.45,
                      alpha=.48)
        ax.scatter([float(row["geo_x"]) for row in vector_rows],
                   [float(row["geo_y"]) for row in vector_rows],
                   s=13, color="white", edgecolor="black", linewidth=.3)
        ax.set(xlabel="geo_x", title=title); ax.set_aspect("equal", adjustable="box")
        fig.colorbar(contour, ax=ax, shrink=.68, label=label)
    axes[0].set_ylabel("geo_y (larger values downward)")
    axes[0].invert_yaxis()
    fig.suptitle("Geometry derived from the two fitted vector-component GAMs")
    fig.savefig(output / "pair_angle_distance_geographic_surfaces.png", dpi=190)
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> int:
    args = arguments()
    first, second = args.pair
    analysis_method = ({"praat-fixed": "praat-bark", "fasttrack": "fasttrack-bark"}[
        args.method] if args.bark else args.method)
    method_label = METHOD_LABELS[analysis_method]
    space_name = "PCA space" if args.method == "pca" else "formant space"
    tokens = [row for row in read_tokens(args.input, args.method, False, args.bark,
                                         args.timepoint)
              if row["village"] != "ref"]
    provinces = resource_provinces(args.resource)
    if not args.include_finland_gotland:
        tokens = [row for row in tokens if
                  provinces.get(ALIASES.get(row["village"], row["village"]))
                  not in DEFAULT_EXCLUDED_REGIONS]
    speakers = speaker_vowels(tokens)
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = [
        (individual_rows(speakers, first), "vowel", f"/{first}/", "first"),
        (individual_rows(speakers, second), "vowel", f"/{second}/", "second"),
        (paired_vector_rows(speakers, first, second), "pair vector",
         f"/{first}→{second}/ vector", "pair_vector"),
        (midpoint_rows(speakers, first, second), "midpoint",
         f"/{first}→{second}/", "midpoint"),
    ]
    results = {}
    for index, (rows, kind, label, directory) in enumerate(jobs):
        results[directory] = analyze_forward_target(
            rows, analysis_method, kind, label, args.output / directory,
            args.permutations, args.grid_size, args.seed + 20 * index,
        )
    reverse_values = village_vowels(speakers)
    reverse_axis_placeholder = {
        village: 0.0 for village, _ in reverse_values
    }  # target_data also carries an unused axis column
    reverse_results = {}
    reverse_jobs = (
        ("first", (first,), "vowel", first),
        ("second", (second,), "vowel", second),
        ("pair", (first, second), "shared_pair_space", f"{first}+{second}"),
    )
    for offset, (key, vowels, kind, label) in enumerate(reverse_jobs):
        reverse_data = reverse_target_data(
            reverse_values, vowels, reverse_axis_placeholder,
        )
        reverse_results[key] = analyze_reverse_target(
            *reverse_data, analysis_method, kind, label, ["geo-y"],
            args.output / f"reverse_{key}", args.permutations,
            max(args.grid_size, 100), args.seed + 100 + 20 * offset,
        )[0]
    pair_figures(args.output, first, second, analysis_method)

    first_result, second_result, vector, midpoint = (results[key] for key in
                              ("first", "second", "pair_vector", "midpoint"))
    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>/{first}–{second}/ in {space_name}: full spatial GAM account</title><style>
:root{{--ink:#26343b;--muted:#68767d;--line:#d8dee2;--panel:#f6f8f9}}
body{{max-width:1250px;margin:auto;padding:2rem;color:var(--ink);font:16px/1.58 system-ui,sans-serif}}
h1,h2{{line-height:1.2}}.lead{{font-size:1.08rem;color:var(--muted);max-width:980px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start}}
figure{{margin:0;padding:.6rem;border:1px solid var(--line);border-radius:8px;background:#fff}}
img{{display:block;width:100%;height:auto}}figcaption{{padding:.5rem .25rem 0;color:var(--muted);font-size:.9rem}}
.single{{max-width:920px;margin:1rem auto}}.result,.caution{{background:var(--panel);border-left:4px solid #657b86;padding:.8rem 1rem}}
.caution{{border-color:#a8813c}}code{{background:#eef1f2;padding:.1rem .25rem}}a{{color:#075e8c}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}body{{padding:1rem}}}}</style></head><body>
<h1>A full spatial-GAM account of /{first}/ and /{second}/ in {space_name}</h1>
<p class="lead"><strong>Acoustic representation:</strong> {method_label}. <strong>Measurement
point:</strong> {args.timepoint}% of vowel duration.</p>
<p class="lead">This report models geographic organization in the two-dimensional acoustic
positions of /{first}/ and /{second}/, their directed relationship, and their midpoint. The forward
models fit each acoustic coordinate with <code>te(geo_x, geo_y)</code>. Villages remain equally
weighted, and Gotland and the Finnish regions are excluded. Blue indicates northern/lower
<code>geo_y</code>, red southern/higher <code>geo_y</code>.</p>

<h2>1. Individual vowel surfaces and acoustic trajectories</h2>
<p>Separate coordinate GAMs map the geographic plane into the acoustic position of each vowel. The
deformed grids display the fitted two-dimensional spatial surface directly in acoustic space:
solid curves move north-to-south at fixed <code>geo_x</code>, and dotted curves vary
<code>geo_x</code> at fixed <code>geo_y</code>.</p>
<div class="grid"><figure><a href="first/gam_geographic_mesh_in_acoustic_space.png"><img
src="first/gam_geographic_mesh_in_acoustic_space.png" alt="first-vowel spatial GAM mesh"></a><figcaption>/{first}/ fitted geographic grid transformed into {space_name}.</figcaption></figure>
<figure><a href="second/gam_geographic_mesh_in_acoustic_space.png"><img
src="second/gam_geographic_mesh_in_acoustic_space.png" alt="second-vowel spatial GAM mesh"></a><figcaption>/{second}/ fitted geographic grid transformed into {space_name}.</figcaption></figure></div>
<p class="result"><strong>Result.</strong> /{first}/ has joint permutation p={first_result['joint_permutation_p']:.4g};
its coordinate R² values are {fmt(first_result['x_explained_deviance'])} and {fmt(first_result['y_explained_deviance'])}
(10-fold CV R² {fmt(first_result['x_cv_r2_10fold'])}, {fmt(first_result['y_cv_r2_10fold'])}). /{second}/ has joint
p={second_result['joint_permutation_p']:.4g}, coordinate R² {fmt(second_result['x_explained_deviance'])} and
{fmt(second_result['y_explained_deviance'])} (CV R² {fmt(second_result['x_cv_r2_10fold'])},
{fmt(second_result['y_cv_r2_10fold'])}). The curved and fanning paths describe geographic organization that
cannot be reduced to one straight geographic axis.</p>

<h2>2. North–south geography projected over acoustic space</h2>
<p>A tensor-product GAM models the north–south coordinate <code>geo_y</code> as a smooth function of
the two acoustic coordinates. Separate models are fitted to /{first}/ and /{second}/, followed by a
model containing both vowels in a shared acoustic plane. The colored contours show the fitted
geographic value at each acoustic position; they are model estimates rather than interpolation of
the observed point colors.</p>
<div class="grid"><figure><a href="reverse_first/reverse_geography_surfaces.png"><img
src="reverse_first/reverse_geography_surfaces.png" alt="first vowel reverse geo_y GAM"></a>
<figcaption>Fitted geo_y over the /{first}/ {space_name}.</figcaption></figure>
<figure><a href="reverse_second/reverse_geography_surfaces.png"><img
src="reverse_second/reverse_geography_surfaces.png" alt="second vowel reverse geo_y GAM"></a>
<figcaption>Fitted geo_y over the /{second}/ {space_name}.</figcaption></figure></div>
<figure class="single"><a href="reverse_pair/reverse_geography_surfaces.png"><img
src="reverse_pair/reverse_geography_surfaces.png" alt="shared pair reverse geo_y GAM"></a>
<figcaption>Fitted geo_y over the combined /{first}/–/{second}/ {space_name}.</figcaption></figure>
<p class="result"><strong>Result.</strong> The /{first}/ reverse geo_y model has R²
{fmt(reverse_results['first']['explained_deviance'])}, CV R²
{fmt(reverse_results['first']['cv_r2_10fold'])}, p={reverse_results['first']['permutation_p']:.4g};
/{second}/ has R² {fmt(reverse_results['second']['explained_deviance'])}, CV R²
{fmt(reverse_results['second']['cv_r2_10fold'])}, p={reverse_results['second']['permutation_p']:.4g};
the shared pair space has R² {fmt(reverse_results['pair']['explained_deviance'])}, CV R²
{fmt(reverse_results['pair']['cv_r2_10fold'])}, p={reverse_results['pair']['permutation_p']:.4g}.</p>

<h2>3. The directed /{first}→{second}/ relationship</h2>
<p>Within-speaker vectors are summarized by village. Two GAMs model their horizontal and vertical
components over geography. Direction and length in the right panel are derived from those fitted
components; they are visual summaries rather than separately fitted responses.</p>
<div class="grid"><figure><a href="pair_vectors_in_acoustic_space.png"><img src="pair_vectors_in_acoustic_space.png"
alt="observed and fitted pair vectors"></a><figcaption>Observed village vectors in grey and fitted
vectors, centered on fitted pair midpoints, in geographic color.</figcaption></figure>
<figure><a href="pair_angle_distance_geographic_surfaces.png"><img
src="pair_angle_distance_geographic_surfaces.png" alt="pair angle and distance surfaces"></a>
<figcaption>Direction and distance calculated from the fitted component surfaces.</figcaption></figure></div>
<p class="result"><strong>Result.</strong> The vector components vary jointly across geography
(permutation p={vector['joint_permutation_p']:.4g}). Component R² values are
{fmt(vector['x_explained_deviance'])} and {fmt(vector['y_explained_deviance'])}, with CV R²
{fmt(vector['x_cv_r2_10fold'])} and {fmt(vector['y_cv_r2_10fold'])}. This tests whether the
directed relationship changes spatially without treating wrapped angular values as an ordinary
linear response.</p>
<p class="caution"><strong>Caution.</strong> Derived direction is unstable where the fitted vector
is short, and angles on opposite sides of ±180° are adjacent rather than far apart. Interpret the
angle map together with the distance map and fitted arrows.</p>

<h2>4. Continuously fitted pair midpoints</h2>
<p>The paired within-speaker midpoint is aggregated within villages and both acoustic coordinates
are fitted over the full geographic plane. No North/Centre/South boundaries are supplied to the
model.</p>
<div class="grid"><figure><a href="midpoint/geographic_surfaces.png"><img
src="midpoint/geographic_surfaces.png" alt="midpoint geographic GAM surfaces"></a><figcaption>
The two fitted midpoint coordinates over geography.</figcaption></figure>
<figure><a href="midpoint/gam_geographic_mesh_in_acoustic_space.png"><img
src="midpoint/gam_geographic_mesh_in_acoustic_space.png" alt="midpoint acoustic GAM mesh"></a>
<figcaption>The fitted geographic grid transformed into midpoint {space_name}.</figcaption></figure></div>
<p class="result"><strong>Result.</strong> Midpoint position has joint permutation
p={midpoint['joint_permutation_p']:.4g}. Coordinate R² values are
{fmt(midpoint['x_explained_deviance'])} and {fmt(midpoint['y_explained_deviance'])}; CV R² values
are {fmt(midpoint['x_cv_r2_10fold'])} and {fmt(midpoint['y_cv_r2_10fold'])}. This supplies a
continuous estimate of how the production midpoint varies across geographic space.</p>

<p><strong>Detailed reports and data:</strong> <a href="first/index.html">/{first}/</a> ·
<a href="second/index.html">/{second}/</a> · <a href="pair_vector/index.html">pair vector</a> ·
<a href="midpoint/index.html">midpoint</a> ·
<a href="reverse_first/model_statistics.csv">first-vowel reverse GAM</a> ·
<a href="reverse_second/model_statistics.csv">second-vowel reverse GAM</a> ·
<a href="reverse_pair/model_statistics.csv">pair-space reverse GAM</a></p></body></html>"""
    (args.output / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
