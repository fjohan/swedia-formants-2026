#!/usr/bin/env python3
"""Fit first-pass vowel-space ellipses by village from formant token measurements."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np


VOWEL_ORDER = ["u:", "o:", "A:", "ä:", "e:", "y:", "U:", "ö:"]
BOUNDARY_PAIRS = [("u:", "o:"), ("o:", "A:"), ("A:", "ä:"), ("ä:", "e:"), ("y:", "U:"), ("U:", "ö:")]
VOWEL_IPA = {
    "u:": "u:",
    "o:": "o:",
    "A:": "ɑ:",
    "ä:": "æ:",
    "e:": "e:",
    "y:": "y:",
    "U:": "ʉ̟:",
    "ö:": "ø:",
}
WORD_VOWELS = set("aeiouyåäöAEIOUYÅÄÖ")


def vowel_label(vowel: str) -> str:
    return f"/{VOWEL_IPA[vowel]}/"


def boundary_label(left: str, right: str) -> str:
    return f"{vowel_label(left)} - {vowel_label(right)}"


def missing_vowel_labels(rows: list[dict]) -> list[str]:
    present = {row["target_label"] for row in rows}
    return [vowel_label(vowel) for vowel in VOWEL_ORDER if vowel not in present]


def finite_median(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else math.nan


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def has_r_context(row: dict) -> bool:
    """Return true when r/R occurs after the first vowel letter in the word."""
    folded = row.get("word", "").casefold()
    first_vowel = next((idx for idx, char in enumerate(folded) if char in WORD_VOWELS), None)
    return first_vowel is not None and "r" in folded[first_vowel + 1:]


def aggregate_surface_notes(rows: list[dict], max_items: int = 8) -> str:
    notes = defaultdict(int)
    for row in rows:
        note = row.get("surface_target_note", "")
        if row.get("surface_target_status") == "allowed_surface" and note:
            notes[note] += 1
            continue
        note = row.get("surface_target_notes", "")
        if note:
            notes[note] += 1
    items = sorted(notes.items(), key=lambda item: (-item[1], item[0]))
    if len(items) > max_items:
        items = items[:max_items] + [("...", 0)]
    return "; ".join(f"{note} ({count})" if count else note for note, count in items)


def wrapped_note(text: str, width: int = 115) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(lines)


def read_tokens(path: Path, method: str, space: str, context_mode: str = "all") -> list[dict]:
    f1_col = f"{method}_f1_{space}"
    f2_col = f"{method}_f2_{space}"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"village", "speaker", "target_label", f1_col, f2_col}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

    tokens = []
    for row in rows:
        if row["target_label"] not in VOWEL_ORDER:
            continue
        r_context = has_r_context(row)
        if context_mode == "non_r" and r_context:
            continue
        if context_mode == "r_only" and not r_context:
            continue
        f1 = parse_float(row[f1_col])
        f2 = parse_float(row[f2_col])
        if math.isfinite(f1) and math.isfinite(f2):
            tokens.append(row | {"f1": f1, "f2": f2, "r_context": int(r_context)})
    return tokens


def speaker_category_medians(tokens: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in tokens:
        grouped[(row["village"], row["speaker"], row["target_label"])].append(row)

    rows = []
    for (village, speaker, vowel), values in sorted(grouped.items()):
        rows.append({
            "village": village,
            "speaker": speaker,
            "target_label": vowel,
            "ipa": VOWEL_IPA[vowel],
            "n_tokens": len(values),
            "f1": finite_median([row["f1"] for row in values]),
            "f2": finite_median([row["f2"] for row in values]),
            "surface_target_notes": aggregate_surface_notes(values),
        })
    return rows


def village_category_medians(speaker_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in speaker_rows:
        grouped[(row["village"], row["target_label"])].append(row)

    rows = []
    for (village, vowel), values in sorted(grouped.items()):
        rows.append({
            "village": village,
            "target_label": vowel,
            "ipa": VOWEL_IPA[vowel],
            "n_speakers": len({row["speaker"] for row in values}),
            "n_tokens": sum(int(row["n_tokens"]) for row in values),
            "f1": finite_median([row["f1"] for row in values]),
            "f2": finite_median([row["f2"] for row in values]),
            "surface_target_notes": aggregate_surface_notes(values),
        })
    return rows


def recording_category_medians(tokens: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in tokens:
        grouped[(row["recording"], row["village"], row["speaker"], row["target_label"])].append(row)

    rows = []
    for (recording, village, speaker, vowel), values in sorted(grouped.items()):
        rows.append({
            "recording": recording,
            "village": village,
            "speaker": speaker,
            "target_label": vowel,
            "ipa": VOWEL_IPA[vowel],
            "n_tokens": len(values),
            "f1": finite_median([row["f1"] for row in values]),
            "f2": finite_median([row["f2"] for row in values]),
            "surface_target_notes": aggregate_surface_notes(values),
        })
    return rows


def boundary_points(rows: list[dict]) -> list[dict]:
    by_vowel = {row["target_label"]: row for row in rows}
    points = []
    for left, right in BOUNDARY_PAIRS:
        if left not in by_vowel or right not in by_vowel:
            continue
        left_row = by_vowel[left]
        right_row = by_vowel[right]
        points.append({
            "target_label": f"{left}-{right}",
            "ipa": boundary_label(left, right),
            "n_tokens": int(left_row["n_tokens"]) + int(right_row["n_tokens"]),
            "f1": (float(left_row["f1"]) + float(right_row["f1"])) / 2,
            "f2": (float(left_row["f2"]) + float(right_row["f2"])) / 2,
        })
    return points


def fit_rows_for_basis(rows: list[dict], basis: str) -> list[dict]:
    if basis == "vowels":
        return rows
    if basis == "boundaries":
        return boundary_points(rows)
    raise ValueError(f"unknown basis: {basis}")


def ellipse_metrics(points: np.ndarray) -> dict[str, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    major = math.sqrt(max(float(values[0]), 0.0))
    minor = math.sqrt(max(float(values[1]), 0.0))
    vector = vectors[:, 0]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0
    signed_angle = angle - 180.0 if angle > 90.0 else angle
    return {
        "centroid_f2": float(centroid[0]),
        "centroid_f1": float(centroid[1]),
        "ellipse_angle_deg": angle,
        "ellipse_angle_signed_deg": signed_angle,
        "major_axis_sd": major,
        "minor_axis_sd": minor,
        "axis_ratio": major / minor if minor else math.inf,
        "ellipse_area_sd": math.pi * major * minor,
    }


def ellipse_outline(metrics: dict[str, float], scale: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(metrics["ellipse_angle_deg"])
    major = metrics["major_axis_sd"] * scale
    minor = metrics["minor_axis_sd"] * scale
    theta = np.linspace(0, 2 * math.pi, 160)
    x = major * np.cos(theta)
    y = minor * np.sin(theta)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated = rotation @ np.vstack([x, y])
    return rotated[0] + metrics["centroid_f2"], rotated[1] + metrics["centroid_f1"]


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_boundary_arrows(ax, rows: list[dict], color: str = "#333333", alpha: float = 0.55) -> None:
    by_vowel = {row["target_label"]: row for row in rows}
    for left, right in BOUNDARY_PAIRS:
        if left not in by_vowel or right not in by_vowel:
            continue
        start = by_vowel[left]
        end = by_vowel[right]
        ax.annotate(
            "",
            xy=(end["f2"], end["f1"]),
            xytext=(start["f2"], start["f1"]),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.15,
                "alpha": alpha,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=2,
        )


def plot_overview(
    output: Path,
    prototype_rows: list[dict],
    ellipse_rows: list[dict],
    unit: str,
    arrows: bool = False,
) -> None:
    by_village = defaultdict(list)
    for row in prototype_rows:
        by_village[row["village"]].append(row)
    ellipse_by_group = {(row["village"], row["fit_basis"]): row for row in ellipse_rows}

    colors = dict(zip(VOWEL_ORDER, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    for village, rows in by_village.items():
        for basis, style in (("vowels", "-"), ("boundaries", "--")):
            if (village, basis) not in ellipse_by_group:
                continue
            metrics = {
                key: float(value) if key not in {"id", "level", "village", "context_mode", "fit_basis", "missing_vowels"} else value
                for key, value in ellipse_by_group[(village, basis)].items()
            }
            x, y = ellipse_outline(metrics)
            ax.plot(x, y, color="#999999", ls=style, lw=0.8, alpha=0.35)
            ax.scatter(metrics["centroid_f2"], metrics["centroid_f1"], color="#333333", s=10, alpha=0.45)
            ax.text(metrics["centroid_f2"], metrics["centroid_f1"], village, fontsize=6, alpha=0.7)
        for row in rows:
            ax.scatter(row["f2"], row["f1"], color=colors[row["target_label"]], s=14, alpha=0.65)
        if arrows:
            draw_boundary_arrows(ax, rows, alpha=0.18)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=vowel_label(vowel),
                   markerfacecolor=colors[vowel], markersize=6)
        for vowel in VOWEL_ORDER
    ]
    ax.legend(handles=handles, loc="lower left", ncol=4, fontsize=8)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(f"F2 ({unit})")
    ax.set_ylabel(f"F1 ({unit})")
    ax.set_title("Village vowel-space ellipses")
    fig.savefig(output / "village_ellipses_overview.png", dpi=160)
    plt.close(fig)


def plot_angles(output: Path, ellipse_rows: list[dict]) -> None:
    rows = sorted(ellipse_rows, key=lambda row: float(row["ellipse_angle_signed_deg"]))
    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 0.24), 4.8), constrained_layout=True)
    basis_colors = {"vowels": "#4c78a8", "boundaries": "#f58518"}
    ax.scatter(
        range(len(rows)),
        [float(row["ellipse_angle_signed_deg"]) for row in rows],
        color=[basis_colors.get(row["fit_basis"], "#4c78a8") for row in rows],
        s=28,
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f'{row["id"]}:{row["fit_basis"]}' for row in rows], rotation=90, fontsize=6)
    ax.set_ylabel("Ellipse long-axis signed angle (degrees)")
    ax.set_title("Village vowel-space orientation")
    ax.set_ylim(-90, 90)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.savefig(output / "village_ellipse_angles.png", dpi=160)
    plt.close(fig)


def read_resource_coordinates(path: Path) -> dict[str, tuple[int, int]]:
    text = path.read_bytes().decode("latin-1")
    rows = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        fields = shlex.split(line[1:-1])
        if len(fields) >= 6:
            rows[fields[0]] = (int(fields[4]), int(fields[5]))
    return rows


def plot_angle_map(output: Path, ellipse_rows: list[dict], resource: Path) -> None:
    coordinates = read_resource_coordinates(resource)
    rows = [row for row in ellipse_rows if row["village"] in coordinates and row["fit_basis"] == "vowels"]
    if not rows:
        return
    xs = [coordinates[row["village"]][0] for row in rows]
    ys = [coordinates[row["village"]][1] for row in rows]
    angles = [float(row["ellipse_angle_signed_deg"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 10.0), constrained_layout=True)
    max_abs = max(abs(angle) for angle in angles) or 1.0
    scatter = ax.scatter(xs, ys, c=angles, cmap="coolwarm", vmin=-max_abs, vmax=max_abs,
                         s=70, edgecolor="black", linewidth=0.5, zorder=3)
    for row, x, y in zip(rows, xs, ys):
        ax.text(x + 3, y - 3, row["village"], fontsize=6, zorder=4)
    fig.colorbar(scatter, ax=ax, label="Ellipse long-axis signed angle (degrees)")
    padding = 18
    ax.set_xlim(min(xs) - padding, max(xs) + padding)
    ax.set_ylim(max(ys) + padding, min(ys) - padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("resource x")
    ax.set_ylabel("resource y")
    ax.set_title("Village vowel-space orientation by coordinate")
    ax.grid(True, color="#dddddd", linewidth=0.5, zorder=0)
    fig.savefig(output / "village_ellipse_angle_map.png", dpi=160)
    plt.close(fig)


def plot_single_fit(
    output: Path,
    group_id: str,
    rows: list[dict],
    ellipse_rows: list[dict],
    unit: str,
    arrows: bool = False,
) -> None:
    if not rows or not ellipse_rows:
        return
    colors = dict(zip(VOWEL_ORDER, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for row in rows:
        ax.scatter(row["f2"], row["f1"], color=colors[row["target_label"]], s=70, zorder=4)
        ax.text(row["f2"] + 0.03, row["f1"] - 0.03, vowel_label(row["target_label"]), fontsize=10, zorder=5)
    if arrows:
        draw_boundary_arrows(ax, rows)

    boundaries = boundary_points(rows)
    for row in boundaries:
        ax.scatter(row["f2"], row["f1"], color="black", marker="x", s=48, zorder=4)
        ax.text(row["f2"] + 0.03, row["f1"] + 0.05, row["ipa"], fontsize=7, zorder=5)

    basis_styles = {"vowels": ("#4c78a8", "-"), "boundaries": ("#f58518", "--")}
    for metrics_row in ellipse_rows:
        metrics = {
            key: float(value) if key not in {"id", "level", "village", "context_mode", "fit_basis", "missing_vowels"} else value
            for key, value in metrics_row.items()
        }
        x, y = ellipse_outline(metrics)
        color, linestyle = basis_styles.get(metrics_row["fit_basis"], ("#999999", "-"))
        ax.plot(
            x, y, color=color, ls=linestyle, lw=2.0,
            label=f'{metrics_row["fit_basis"]}: {float(metrics_row["ellipse_angle_signed_deg"]):.1f}°',
            zorder=3,
        )
        ax.scatter(metrics["centroid_f2"], metrics["centroid_f1"], color=color, s=25, zorder=4)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(f"F2 ({unit})")
    ax.set_ylabel(f"F1 ({unit})")
    missing = missing_vowel_labels(rows)
    title = f"Vowel-space ellipse fit: {group_id}"
    notes: list[str] = []
    if missing:
        title += f"   incomplete: missing {', '.join(missing)}"
        notes.append("Incomplete target set: missing " + ", ".join(missing))
    surface_notes = aggregate_surface_notes(rows)
    if surface_notes:
        notes.append("Allowed surface targets: " + surface_notes)
    ax.set_title(title)
    ax.legend(loc="best")
    if notes:
        fig.text(
            0.02,
            0.02,
            wrapped_note("\n".join(notes)),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7fbff", "edgecolor": "#9ecae1", "alpha": 0.92},
        )
        fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(output / f"{group_id}_ellipse_fit.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True, help="token_comparison.csv from compare_formant_trackers.py")
    parser.add_argument("--output", type=Path, default=Path("VowelEllipseAnalysis"))
    parser.add_argument("--method", choices=["praat", "fasttrack"], default="fasttrack")
    parser.add_argument("--space", choices=["bark", "hz"], default="bark")
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--id", help="Recording id such as bara_ym_1, or village id such as bara.")
    parser.add_argument("--level", choices=["auto", "recording", "village"], default="auto")
    parser.add_argument("--fit-basis", choices=["vowels", "boundaries", "both"], default="vowels")
    parser.add_argument(
        "--context-mode",
        choices=["all", "non_r", "r_only"],
        default="all",
        help="Lexical context filter. non_r excludes words where r/R occurs after the vowel; r_only keeps only them.",
    )
    parser.add_argument(
        "--arrows",
        choices=["none", "single", "overview", "both"],
        default="none",
        help="Draw arrows between vowel pairs used to calculate boundary midpoints.",
    )
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--min-vowels", type=int, default=6)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tokens = read_tokens(args.tokens, args.method, args.space, args.context_mode)
    if args.id:
        if args.level == "auto":
            is_recording = any(row["recording"] == args.id for row in tokens)
            level = "recording" if is_recording else "village"
        else:
            level = args.level
        key = "recording" if level == "recording" else "village"
        tokens = [row for row in tokens if row[key] == args.id]
        if not tokens:
            raise ValueError(f"no tokens matched {key} id: {args.id}")
    else:
        level = "village"

    speaker_rows = speaker_category_medians(tokens)
    prototype_rows = village_category_medians(speaker_rows)
    recording_rows = recording_category_medians(tokens)

    if level == "recording":
        grouped_rows = defaultdict(list)
        for row in recording_rows:
            grouped_rows[row["recording"]].append(row)
    else:
        grouped_rows = defaultdict(list)
        for row in prototype_rows:
            grouped_rows[row["village"]].append(row)

    ellipse_rows = []
    bases = ["vowels", "boundaries"] if args.fit_basis == "both" else [args.fit_basis]
    for group_id, rows in sorted(grouped_rows.items()):
        village = rows[0]["village"]
        speakers = {row["speaker"] for row in speaker_rows if row["village"] == village}
        if level == "recording":
            speakers = {rows[0]["speaker"]}
        complete_rows = [row for row in rows if math.isfinite(row["f1"]) and math.isfinite(row["f2"])]
        if level == "village" and len(speakers) < args.min_speakers:
            continue
        for basis in bases:
            fit_rows = fit_rows_for_basis(complete_rows, basis)
            if len(fit_rows) < args.min_vowels:
                continue
            points = np.array([[row["f2"], row["f1"]] for row in fit_rows], dtype=float)
            metrics = ellipse_metrics(points)
            ellipse_rows.append({
                "id": group_id,
                "level": level,
                "village": village,
                "context_mode": args.context_mode,
                "fit_basis": basis,
                "n_speakers": len(speakers),
                "n_points": len(fit_rows),
                "n_vowels": len(complete_rows),
                "missing_vowels": ";".join(missing_vowel_labels(complete_rows)),
                "n_tokens": sum(int(row["n_tokens"]) for row in complete_rows),
                **{key: f"{value:.6g}" for key, value in metrics.items()},
            })

    write_rows(
        args.output / "speaker_vowel_medians.csv",
        speaker_rows,
        ["village", "speaker", "target_label", "ipa", "n_tokens", "f1", "f2", "surface_target_notes"],
    )
    write_rows(
        args.output / "recording_vowel_medians.csv",
        recording_rows,
        ["recording", "village", "speaker", "target_label", "ipa", "n_tokens", "f1", "f2", "surface_target_notes"],
    )
    write_rows(
        args.output / "village_vowel_prototypes.csv",
        prototype_rows,
        ["village", "target_label", "ipa", "n_speakers", "n_tokens", "f1", "f2", "surface_target_notes"],
    )
    write_rows(
        args.output / "village_ellipse_metrics.csv",
        ellipse_rows,
        [
            "id", "level", "village", "context_mode", "fit_basis", "n_speakers", "n_points",
            "n_vowels", "missing_vowels", "n_tokens",
            "centroid_f2", "centroid_f1",
            "ellipse_angle_deg", "ellipse_angle_signed_deg",
            "major_axis_sd", "minor_axis_sd", "axis_ratio", "ellipse_area_sd",
        ],
    )
    if ellipse_rows:
        if args.id:
            plot_rows = recording_rows if level == "recording" else prototype_rows
            plot_rows = [
                row for row in plot_rows
                if (row["recording"] if level == "recording" else row["village"]) == args.id
            ]
            plot_single_fit(
                args.output,
                args.id,
                plot_rows,
                ellipse_rows,
                args.space,
                arrows=args.arrows in {"single", "both"},
            )
        plot_overview(
            args.output,
            prototype_rows,
            ellipse_rows,
            args.space,
            arrows=args.arrows in {"overview", "both"},
        )
        plot_angles(args.output, ellipse_rows)
        if args.resource.exists():
            plot_angle_map(args.output, ellipse_rows, args.resource)

    print(f"Read {len(tokens)} usable tokens from {args.tokens}")
    print(f"Wrote {len(speaker_rows)} speaker-vowel medians")
    print(f"Wrote {len(prototype_rows)} village-vowel prototypes")
    print(f"Wrote {len(ellipse_rows)} village ellipses to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
