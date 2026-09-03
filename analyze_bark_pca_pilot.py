#!/usr/bin/env python3
"""Bark-filter spectral PCA pilot for four SweDia villages and eight vowels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import wave
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swedia-pca-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window, resample_poly

from inventory_base_word_targets import (
    ALTERNATIVE_TARGET_MAP,
    BASE_TARGETS,
    EXTRA_COUNTS,
    looks_vowel_like,
    normalized_word,
    parse_textgrid_tier,
    second_lat_window,
)


DEFAULT_RECORDINGS = [
    f"{village}_ym_{speaker}"
    for village in ("bara", "karsta", "ostad", "pitea")
    for speaker in (1, 2, 3)
]
TIME_POINTS = (0.10, 0.25, 0.50, 0.75, 0.90)
VOWEL_ORDER = [target["target"] for target in BASE_TARGETS]
VOWEL_IPA = {target["target"]: target["ipa"] for target in BASE_TARGETS}
BOUNDARY_PAIRS = [
    ("u:", "o:"), ("o:", "A:"), ("A:", "ä:"), ("ä:", "e:"),
    ("y:", "U:"), ("U:", "ö:"),
]


def lexical_targets() -> dict[str, tuple[str, str]]:
    """Map inventory base and approved alternative words to base categories."""
    result = {target["word"]: (target["target"], "base") for target in BASE_TARGETS}
    alternative_to_base = {
        alternative: base
        for base, alternatives in ALTERNATIVE_TARGET_MAP.items()
        for alternative in alternatives
    }
    for target in EXTRA_COUNTS:
        base = alternative_to_base.get(target["target"])
        if base:
            result[target["word"]] = (base, "alternative")
    return result


def read_wav(path: Path, target_rate: int = 16000) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit PCM audio in {path}, found {width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, rate // divisor)
        rate = target_rate
    return rate, samples


def hz_to_bark(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=float)
    return 26.81 / (1.0 + 1960.0 / np.maximum(hz, 1e-12)) - 0.53


def bark_filterbank(freqs: np.ndarray, n_bands: int, max_bark: float) -> tuple[np.ndarray, np.ndarray]:
    """Return equally spaced triangular filters on the Traunmüller Bark scale."""
    edges = np.linspace(0.0, max_bark, n_bands + 2)
    centers = edges[1:-1]
    bark_freqs = hz_to_bark(freqs)
    filters = np.zeros((n_bands, len(freqs)))
    for index, (left, center, right) in enumerate(zip(edges[:-2], centers, edges[2:])):
        filters[index] = np.maximum(
            0.0,
            np.minimum((bark_freqs - left) / (center - left), (right - bark_freqs) / (right - center)),
        )
    return centers, filters


def spectral_vector(
    samples: np.ndarray,
    rate: int,
    center_s: float,
    window_ms: float,
    n_fft: int,
    filters: np.ndarray,
) -> tuple[np.ndarray, float]:
    length = round(rate * window_ms / 1000.0)
    center = round(center_s * rate)
    start = center - length // 2
    frame = np.zeros(length)
    source_start, source_end = max(0, start), min(len(samples), start + length)
    if source_end > source_start:
        frame[source_start - start:source_end - start] = samples[source_start:source_end]
    rms_dbfs = 20.0 * np.log10(max(np.sqrt(np.mean(frame ** 2)), 1e-12))
    frame *= get_window("hann", length, fftbins=True)
    power = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
    energies = (filters @ power) / np.maximum(filters.sum(axis=1), 1e-12)
    db = 10.0 * np.log10(np.maximum(energies, 1e-20))
    # Remove overall level while retaining spectral shape; 80 is a readable reference level.
    total_db = 10.0 * np.log10(np.maximum(energies.sum(), 1e-20))
    return db + (80.0 - total_db), float(rms_dbfs)


def find_vowel_segment(word_start: float, word_end: float, segments: list[tuple[float, float, str]]):
    candidates = [
        segment for segment in segments
        if word_start <= (segment[0] + segment[1]) / 2 <= word_end and looks_vowel_like(segment[2])
    ]
    return candidates[0] if len(candidates) == 1 else None


def locate_recording(stem: str, media_dirs: list[Path]) -> Path:
    matches = [directory / f"{stem}.wav" for directory in media_dirs if (directory / f"{stem}.wav").exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one WAV for {stem}, found: {matches}")
    return matches[0]


def collect_measurements(args, centers: np.ndarray, filters: np.ndarray) -> tuple[list[dict], list[dict]]:
    mapping = lexical_targets()
    expected_segments = {target["target"]: target["seg"] for target in BASE_TARGETS}
    rows, exclusions = [], []
    for stem in args.recordings:
        wav_path = locate_recording(stem, args.media_dirs)
        textgrid = args.textgrids / f"{stem}.TextGrid"
        if not textgrid.exists():
            raise FileNotFoundError(textgrid)
        rate, samples = read_wav(wav_path, args.sample_rate)
        words = parse_textgrid_tier(textgrid, "ord")
        segments = parse_textgrid_tier(textgrid, "seg")
        second_window = second_lat_window(words)
        token_number = 0
        for word_start, word_end, word in words:
            word_key = normalized_word(word)
            target = mapping.get(word_key)
            if target is None:
                continue
            target_label, lexical_status = target
            if word_key == "låt" and second_window is not None:
                midpoint = (word_start + word_end) / 2
                if second_window[0] <= midpoint <= second_window[1]:
                    continue
            segment = find_vowel_segment(word_start, word_end, segments)
            if segment is None:
                exclusions.append({"recording": stem, "word": word, "reason": "not_exactly_one_vowel_segment"})
                continue
            start, end, surface = segment
            if args.strict_seg_label and surface != expected_segments[target_label]:
                exclusions.append({
                    "recording": stem, "word": word,
                    "reason": f"surface_label_{surface}_expected_{expected_segments[target_label]}",
                })
                continue
            token_number += 1
            token_id = f"{stem}_{token_number:03d}_{target_label.replace(':', 'L')}"
            village = stem.rsplit("_ym_", 1)[0]
            speaker = f"ym_{stem.rsplit('_ym_', 1)[1].split('_')[0]}"
            for proportion in args.time_points:
                vector, rms_dbfs = spectral_vector(
                    samples, rate, start + proportion * (end - start), args.window_ms, args.n_fft, filters
                )
                row = {
                    "token_id": token_id, "recording": stem, "village": village, "speaker": speaker,
                    "word": word, "lexical_status": lexical_status, "target_label": target_label,
                    "ipa": VOWEL_IPA[target_label], "surface_seg_label": surface,
                    "vowel_start_s": start, "vowel_end_s": end, "duration_s": end - start,
                    "time_proportion": proportion, "measurement_s": start + proportion * (end - start),
                    "rms_dbfs": rms_dbfs,
                }
                row.update({f"bark_{i + 1:02d}_db": value for i, value in enumerate(vector)})
                rows.append(row)
    return rows, exclusions


def fit_pca(rows: list[dict], band_fields: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    grouped: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (row["recording"], row["village"], row["speaker"], row["target_label"], row["time_proportion"])
        grouped[key].append(np.array([row[field] for field in band_fields]))
    balanced = []
    for key, vectors in sorted(grouped.items()):
        recording, village, speaker, target, time = key
        balanced.append({
            "recording": recording, "village": village, "speaker": speaker, "target_label": target,
            "ipa": VOWEL_IPA[target], "time_proportion": time, "n_tokens": len(vectors),
            **{field: value for field, value in zip(band_fields, np.mean(vectors, axis=0))},
        })
    matrix = np.array([[row[field] for field in band_fields] for row in balanced])
    mean = matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt
    variance = singular ** 2 / (len(matrix) - 1)
    ratio = variance / variance.sum()
    return mean, components, ratio, balanced


def project(rows: list[dict], mean: np.ndarray, components: np.ndarray, band_fields: list[str]) -> None:
    matrix = np.array([[row[field] for field in band_fields] for row in rows])
    scores = (matrix - mean) @ components[:3].T
    for row, score in zip(rows, scores):
        row.update({"pc1": score[0], "pc2": score[1], "pc3": score[2]})


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scree(path: Path, ratio: np.ndarray) -> None:
    shown = min(10, len(ratio))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(1, shown + 1), ratio[:shown] * 100, color="#4477aa")
    ax.plot(np.arange(1, shown + 1), np.cumsum(ratio[:shown]) * 100, "o-", color="#cc6677")
    ax.set(xlabel="Principal component", ylabel="Variance (%)", title="Bark-spectrum PCA variance")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_loadings(path: Path, centers: np.ndarray, components: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for index, color in zip(range(3), ("#4477aa", "#cc6677", "#228833")):
        ax.plot(centers, components[index], marker="o", label=f"PC{index + 1}", color=color)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set(xlabel="Filter center (Bark)", ylabel="Loading", title="PCA loading curves")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def panel_grid(villages: list[str], columns: int = 4):
    rows = math.ceil(len(villages) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flat
    return fig, list(axes)


def ellipse_metrics(points: np.ndarray) -> dict[str, float]:
    centroid = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points - centroid, rowvar=False))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    vector = vectors[:, 0]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0
    signed_angle = angle - 180.0 if angle > 90.0 else angle
    vertical_angle = math.copysign(90.0 - abs(signed_angle), signed_angle if signed_angle else 1.0)
    major = math.sqrt(max(float(values[0]), 0.0))
    minor = math.sqrt(max(float(values[1]), 0.0))
    axis_ratio = major / minor if minor else math.inf
    return {
        "centroid_pc2": float(centroid[0]), "centroid_pc1": float(centroid[1]),
        "ellipse_angle_deg": angle, "ellipse_angle_signed_deg": signed_angle,
        "ellipse_angle_from_vertical_signed_deg": vertical_angle,
        "major_axis_sd": major, "minor_axis_sd": minor, "axis_ratio": axis_ratio,
        "orientation_reliable": int(axis_ratio >= 1.2),
    }


def ellipse_outline(metrics: dict[str, float], scale: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(metrics["ellipse_angle_deg"])
    theta = np.linspace(0, 2 * math.pi, 160)
    points = np.vstack([
        metrics["major_axis_sd"] * scale * np.cos(theta),
        metrics["minor_axis_sd"] * scale * np.sin(theta),
    ])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated = rotation @ points
    return rotated[0] + metrics["centroid_pc2"], rotated[1] + metrics["centroid_pc1"]


def plot_midpoints(path: Path, rows: list[dict]) -> list[dict]:
    subset = [row for row in rows if math.isclose(row["time_proportion"], 0.5)]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in subset:
        grouped[(row["village"], row["target_label"])].append(row)
    colors = dict(zip(VOWEL_ORDER, plt.get_cmap("tab10").colors))
    villages = sorted({row["village"] for row in rows})
    ellipse_rows = []
    fig, axes = panel_grid(villages)
    for ax, village in zip(axes, villages):
        positions = {}
        for target in VOWEL_ORDER:
            values = grouped.get((village, target), [])
            if not values:
                continue
            x, y = np.mean([r["pc2"] for r in values]), np.mean([r["pc1"] for r in values])
            positions[target] = (x, y)
            ax.scatter(x, y, color=colors[target], s=38)
            ax.annotate(f"/{VOWEL_IPA[target]}/", (x, y), xytext=(4, 3), textcoords="offset points")
        midpoints = []
        for first, second in BOUNDARY_PAIRS:
            if first not in positions or second not in positions:
                continue
            start, end = positions[first], positions[second]
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            midpoints.append(midpoint)
            ax.annotate(
                "", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0,
                            "alpha": 0.55, "shrinkA": 6, "shrinkB": 6},
                zorder=2,
            )
            ax.scatter(*midpoint, color="#222222", marker="x", s=24, lw=1.0, zorder=4)
        bases = [("vowels", list(positions.values()), "#4c78a8", "-"),
                 ("midpoints", midpoints, "#f58518", "--")]
        for basis, points, color, linestyle in bases:
            if len(points) < 3:
                continue
            metrics = ellipse_metrics(np.asarray(points, dtype=float))
            outline_x, outline_y = ellipse_outline(metrics)
            if metrics["orientation_reliable"]:
                angle_label = f'{metrics["ellipse_angle_from_vertical_signed_deg"]:+.1f}° from vertical'
            else:
                angle_label = f'angle unstable (ratio {metrics["axis_ratio"]:.2f})'
            ax.plot(
                outline_x, outline_y, color=color, ls=linestyle, lw=1.7,
                label=f"{basis}: {angle_label}", zorder=3,
            )
            ellipse_rows.append({"village": village, "fit_basis": basis, "n_points": len(points), **metrics})
        ax.set_title(village.capitalize()); ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        if positions:
            ax.legend(loc="best", fontsize=6, framealpha=0.75)
    for ax in axes[len(villages):]:
        ax.set_visible(False)
    fig.supxlabel("PC2"); fig.supylabel("PC1"); fig.suptitle("Midpoint vowel spaces: village means")
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(path, dpi=180); plt.close(fig)
    return ellipse_rows


def plot_trajectories(path: Path, balanced: list[dict], tokens: list[dict], show_individual: bool) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in balanced:
        grouped[(row["village"], row["target_label"])].append(row)
    colors = dict(zip(VOWEL_ORDER, plt.get_cmap("tab10").colors))
    villages = sorted({row["village"] for row in balanced})
    fig, axes = panel_grid(villages)
    for ax, village in zip(axes, villages):
        if show_individual:
            token_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for row in tokens:
                if row["village"] == village:
                    token_groups[(row["target_label"], row["token_id"])].append(row)
            for (target, _), token_rows in token_groups.items():
                token_rows.sort(key=lambda row: row["time_proportion"])
                ax.plot(
                    [row["pc2"] for row in token_rows], [row["pc1"] for row in token_rows],
                    color=colors[target], alpha=0.13, lw=0.7, zorder=1,
                )
        for target in VOWEL_ORDER:
            by_time: dict[float, list[dict]] = defaultdict(list)
            for row in grouped.get((village, target), []):
                by_time[row["time_proportion"]].append(row)
            points = [(t, np.mean([r["pc2"] for r in rs]), np.mean([r["pc1"] for r in rs])) for t, rs in sorted(by_time.items())]
            if not points:
                continue
            x, y = np.array([p[1] for p in points]), np.array([p[2] for p in points])
            ax.plot(x, y, "o-", color=colors[target], lw=2.8, ms=4.5, zorder=3)
            ax.annotate(f"/{VOWEL_IPA[target]}/", (x[-1], y[-1]), xytext=(3, 2), textcoords="offset points", fontsize=8)
            if len(x) > 1:
                ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops={"arrowstyle": "->", "color": colors[target]})
        ax.set_title(village.capitalize()); ax.grid(alpha=0.2)
    for ax in axes[len(villages):]:
        ax.set_visible(False)
    fig.supxlabel("PC2"); fig.supylabel("PC1"); fig.suptitle("Mean spectral trajectories (10% → 90%)")
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(path, dpi=180); plt.close(fig)


def plot_speakers(output: Path, balanced: list[dict], tokens: list[dict]) -> None:
    """Plot globally projected token and mean trajectories for each speaker."""
    output.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(VOWEL_ORDER, plt.get_cmap("tab10").colors))
    all_x = np.array([row["pc2"] for row in tokens])
    all_y = np.array([row["pc1"] for row in tokens])
    x_pad = max(2.0, 0.04 * np.ptp(all_x))
    y_pad = max(2.0, 0.04 * np.ptp(all_y))
    xlim = (all_x.min() - x_pad, all_x.max() + x_pad)
    ylim = (all_y.min() - y_pad, all_y.max() + y_pad)
    recordings = sorted({row["recording"] for row in tokens})
    for recording in recordings:
        speaker_tokens = [row for row in tokens if row["recording"] == recording]
        speaker_means = [row for row in balanced if row["recording"] == recording]
        fig, ax = plt.subplots(figsize=(8, 7))
        token_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in speaker_tokens:
            token_groups[(row["target_label"], row["token_id"])].append(row)
        for (target, _), token_rows in token_groups.items():
            token_rows.sort(key=lambda row: row["time_proportion"])
            ax.plot(
                [row["pc2"] for row in token_rows], [row["pc1"] for row in token_rows],
                color=colors[target], alpha=0.18, lw=0.8, zorder=1,
            )
        for target in VOWEL_ORDER:
            mean_rows = sorted(
                (row for row in speaker_means if row["target_label"] == target),
                key=lambda row: row["time_proportion"],
            )
            if not mean_rows:
                continue
            x = np.array([row["pc2"] for row in mean_rows])
            y = np.array([row["pc1"] for row in mean_rows])
            ax.plot(x, y, "o-", color=colors[target], lw=3.0, ms=5, zorder=3)
            ax.annotate(
                f"/{VOWEL_IPA[target]}/", (x[-1], y[-1]), xytext=(4, 3),
                textcoords="offset points", fontsize=10,
            )
            if len(x) > 1:
                ax.annotate(
                    "", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                    arrowprops={"arrowstyle": "->", "color": colors[target], "lw": 1.5},
                )
        missing = [f"/{VOWEL_IPA[target]}/" for target in VOWEL_ORDER if not any(
            row["target_label"] == target for row in speaker_means
        )]
        subtitle = f"Missing: {', '.join(missing)}" if missing else "All eight vowel categories present"
        ax.set(
            xlim=xlim, ylim=ylim, xlabel="PC2", ylabel="PC1",
            title=f"{recording}: token and mean trajectories\n{subtitle}",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(output / f"{recording}_trajectories.png", dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", nargs="+", default=DEFAULT_RECORDINGS)
    parser.add_argument(
        "--recordings-from-plots", type=Path,
        help="Derive recording IDs from *_tracker_ellipse_side_by_side_bark.png files in this directory.",
    )
    parser.add_argument("--media-dirs", nargs="+", type=Path, default=[Path("Media"), Path("sounds")])
    parser.add_argument("--textgrids", type=Path, default=Path("TextGrids_all_wos"))
    parser.add_argument("--output", type=Path, default=Path("Analyses/BarkPCA_pilot"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=float, default=25.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--max-bark", type=float, default=21.0)
    parser.add_argument("--time-points", nargs="+", type=float, default=TIME_POINTS)
    parser.add_argument(
        "--strict-seg-label", action="store_true",
        help="Keep a token only when its surface seg label exactly matches the base vowel's canonical label.",
    )
    parser.add_argument(
        "--individual-trajectories", action="store_true",
        help="Draw faint token trajectories underneath the thick speaker-balanced village mean.",
    )
    parser.add_argument(
        "--speaker-plots", action="store_true",
        help="Write one globally projected token-and-mean trajectory plot per speaker.",
    )
    args = parser.parse_args()
    if args.recordings_from_plots:
        suffix = "_tracker_ellipse_side_by_side_bark.png"
        prefixes = ("media_annotations_", "sounds_sannotations_")
        recordings = []
        for path in args.recordings_from_plots.glob(f"*{suffix}"):
            stem = path.name[:-len(suffix)]
            for prefix in prefixes:
                if stem.startswith(prefix):
                    stem = stem[len(prefix):]
                    break
            recordings.append(stem)
        args.recordings = sorted(set(recordings))
        if not args.recordings:
            parser.error(f"No combined speaker plots found in {args.recordings_from_plots}")
    args.output.mkdir(parents=True, exist_ok=True)
    freqs = np.fft.rfftfreq(args.n_fft, 1.0 / args.sample_rate)
    centers, filters = bark_filterbank(freqs, args.bands, args.max_bark)
    band_fields = [f"bark_{i + 1:02d}_db" for i in range(args.bands)]
    rows, exclusions = collect_measurements(args, centers, filters)
    mean, components, ratio, balanced = fit_pca(rows, band_fields)
    project(rows, mean, components, band_fields)
    project(balanced, mean, components, band_fields)
    write_csv(args.output / "token_spectral_pca.csv", rows)
    write_csv(args.output / "balanced_speaker_vowel_time_pca.csv", balanced)
    write_csv(args.output / "exclusions.csv", exclusions)
    write_csv(args.output / "pca_loadings.csv", [
        {"band": i + 1, "center_bark": centers[i], "mean_db": mean[i],
         **{f"pc{j + 1}_loading": components[j, i] for j in range(3)}}
        for i in range(args.bands)
    ])
    write_csv(args.output / "explained_variance.csv", [
        {"component": i + 1, "explained_variance_ratio": value, "cumulative_ratio": ratio[:i + 1].sum()}
        for i, value in enumerate(ratio)
    ])
    plot_scree(args.output / "scree.png", ratio)
    plot_loadings(args.output / "loadings.png", centers, components)
    ellipse_rows = plot_midpoints(args.output / "midpoint_vowel_spaces.png", balanced)
    write_csv(args.output / "village_pca_ellipses.csv", ellipse_rows)
    plot_trajectories(
        args.output / "vowel_trajectories.png", balanced, rows, args.individual_trajectories
    )
    if args.speaker_plots:
        plot_speakers(args.output / "speakers", balanced, rows)
    settings = vars(args) | {"filter_centers_bark": centers.tolist(), "pca_fit_rows": len(balanced),
                             "token_measurement_rows": len(rows), "excluded_tokens": len(exclusions)}
    settings = {key: [str(x) for x in value] if isinstance(value, list) and value and isinstance(value[0], Path)
                else str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    (args.output / "run_settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} token-time measurements and {len(balanced)} balanced PCA-fit rows to {args.output}")
    print(f"PC1-PC3 variance: {', '.join(f'{100 * value:.1f}%' for value in ratio[:3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
