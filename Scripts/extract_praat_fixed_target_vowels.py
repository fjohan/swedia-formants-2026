#!/usr/bin/env python3
"""Extract token-level fixed-ceiling Praat F1/F2 for all target vowels."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
import parselmouth

from analyze_bark_pca_pilot import (
    VOWEL_IPA,
    bark_filterbank,
    find_vowel_segment,
    lexical_targets,
    locate_recording,
    read_wav,
    spectral_vector,
)
from inventory_base_word_targets import (
    BASE_TARGETS,
    normalized_word,
    parse_textgrid_tier,
    second_lat_window,
)
from plot_pca_angle_maps import RESOURCE_ALIASES, read_coordinates


FIELDS = (
    "village", "speaker", "vowel", "word", "geo_x", "geo_y", "complete",
    "f1_20", "f2_20", "f1_50", "f2_50", "f1_80", "f2_80", "VL",
)
FASTTRACK_FIELDS = (
    "ft_f1_20", "ft_f2_20",
    "ft_f1_50", "ft_f2_50",
    "ft_f1_80", "ft_f2_80", "ft_VL",
)
PCA_FIELDS = (
    "pca_pc1_20", "pca_pc2_20",
    "pca_pc1_50", "pca_pc2_50",
    "pca_pc1_80", "pca_pc2_80", "pca_VL",
)


def canonical_ipa(target: str) -> str:
    """Return the inventory IPA label with the IPA length mark."""
    value = VOWEL_IPA[target]
    return value[:-1] + "ː" if value.endswith(":") else value


def recording_parts(stem: str) -> tuple[str, str]:
    """Return the project village and speaker labels from a recording stem."""
    if "_ym_" not in stem:
        raise ValueError(f"Recording name does not contain '_ym_': {stem}")
    village, number = stem.rsplit("_ym_", 1)
    # Some recordings (notably Orsa) are split across multiple WAV files,
    # e.g. orsa_ym_1_1 and orsa_ym_1_2, but still belong to speaker ym_1.
    return village, f"ym_{number.split('_', 1)[0]}"


def finite_median(values: list[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def discover_recordings(textgrid_dir: Path, media_dirs: list[Path]) -> list[str]:
    annotated = {path.stem for path in textgrid_dir.glob("*.TextGrid")}
    audio = {
        path.stem
        for directory in media_dirs
        if directory.exists()
        for path in directory.glob("*.wav")
    }
    return sorted(stem for stem in annotated & audio if "_ym_" in stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("Analyses/praat_fixed_target_vowels.csv"),
    )
    parser.add_argument("--textgrids", type=Path, default=Path("TextGrids_all_wos"))
    parser.add_argument("--media-dirs", nargs="+", type=Path, default=[Path("Media"), Path("sounds")])
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--ceiling", type=float, default=5500.0)
    parser.add_argument("--window-length", type=float, default=0.025)
    parser.add_argument("--time-step", type=float, default=0.002)
    parser.add_argument("--pre-emphasis", type=float, default=50.0)
    parser.add_argument(
        "--fasttrack", action="store_true",
        help="Add robust FastTrackPy measurements at the same time points.",
    )
    parser.add_argument("--fasttrack-min-ceiling", type=float, default=4000.0)
    parser.add_argument("--fasttrack-max-ceiling", type=float, default=7000.0)
    parser.add_argument("--fasttrack-steps", type=int, default=20)
    parser.add_argument(
        "--pca", action="store_true",
        help="Fit a global speaker-vowel-balanced spectral PCA and add PC1/PC2 trajectories.",
    )
    parser.add_argument("--pca-sample-rate", type=int, default=16000)
    parser.add_argument("--pca-window-ms", type=float, default=25.0)
    parser.add_argument("--pca-n-fft", type=int, default=1024)
    parser.add_argument("--pca-bands", type=int, default=20)
    parser.add_argument("--pca-max-bark", type=float, default=21.0)
    parser.add_argument(
        "--strict-seg-label", action="store_true",
        help="Require the surface segment label to equal the base target label.",
    )
    parser.add_argument("--limit-recordings", type=int, default=0, help="Testing only; zero means all.")
    parser.add_argument(
        "--recordings", nargs="+",
        help="Analyze only these recording stems (useful for diagnosis or small runs).",
    )
    args = parser.parse_args()

    CandidateTracks = None
    if args.fasttrack:
        try:
            from fasttrackpy import CandidateTracks
        except ImportError as error:
            raise SystemExit(
                "FastTrackPy is unavailable in this interpreter. Run with "
                "fasttrackpy/bin/python (the repository directory named "
                "'fasttrackpy' shadows the package in swedia-pca)."
            ) from error
    pca_filters = None
    if args.pca:
        frequencies = np.fft.rfftfreq(args.pca_n_fft, 1.0 / args.pca_sample_rate)
        _, pca_filters = bark_filterbank(frequencies, args.pca_bands, args.pca_max_bark)

    mapping = lexical_targets()
    expected = {target["target"]: target["seg"] for target in BASE_TARGETS}
    coordinates = read_coordinates(args.resource)
    recordings = discover_recordings(args.textgrids, args.media_dirs)
    if args.recordings:
        requested = set(args.recordings)
        missing = requested - set(recordings)
        if missing:
            raise SystemExit(f"Unknown or unavailable recordings: {', '.join(sorted(missing))}")
        recordings = [stem for stem in recordings if stem in requested]
    if args.limit_recordings:
        recordings = recordings[: args.limit_recordings]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_annotations = failed_formants = failed_fasttrack = 0
    missing_coordinates: set[str] = set()

    rows: list[dict] = []
    for index, stem in enumerate(recordings, 1):
            village, speaker = recording_parts(stem)
            resource_key = RESOURCE_ALIASES.get(village, village)
            if resource_key not in coordinates:
                missing_coordinates.add(village)
                continue
            x, y, _ = coordinates[resource_key]
            wav_path = locate_recording(stem, args.media_dirs)
            sound = parselmouth.Sound(str(wav_path))
            if args.pca:
                pca_rate, pca_samples = read_wav(wav_path, args.pca_sample_rate)
            textgrid = args.textgrids / f"{stem}.TextGrid"
            words = parse_textgrid_tier(textgrid, "ord")
            segments = parse_textgrid_tier(textgrid, "seg")
            second_window = second_lat_window(words)

            for word_start, word_end, word in words:
                word_key = normalized_word(word)
                target = mapping.get(word_key)
                if target is None:
                    continue
                vowel, _ = target
                if word_key == "låt" and second_window is not None:
                    word_midpoint = (word_start + word_end) / 2
                    if second_window[0] <= word_midpoint <= second_window[1]:
                        continue
                segment = find_vowel_segment(word_start, word_end, segments)
                if segment is None:
                    skipped_annotations += 1
                    continue
                start, end, surface = segment
                if args.strict_seg_label and surface != expected[vowel]:
                    skipped_annotations += 1
                    continue

                clip = sound.extract_part(
                    max(sound.xmin, start - 0.04),
                    min(sound.xmax, end + 0.04),
                    preserve_times=True,
                )
                formant = clip.to_formant_burg(
                    time_step=args.time_step,
                    max_number_of_formants=5.5,
                    maximum_formant=args.ceiling,
                    window_length=args.window_length,
                    pre_emphasis_from=args.pre_emphasis,
                )
                measurement_times = {
                    percentage: np.linspace(
                        start + (proportion - 0.05) * (end - start),
                        start + (proportion + 0.05) * (end - start),
                        11,
                    )
                    for percentage, proportion in ((20, 0.20), (50, 0.50), (80, 0.80))
                }
                values = {
                    (formant_number, percentage): finite_median([
                        formant.get_value_at_time(formant_number, time) for time in times
                    ])
                    for percentage, times in measurement_times.items()
                    for formant_number in (1, 2)
                }
                if any(value is None or not np.isfinite(value) for value in values.values()):
                    failed_formants += 1
                    continue
                f1_20, f2_20 = values[1, 20], values[2, 20]
                f1_50, f2_50 = values[1, 50], values[2, 50]
                f1_80, f2_80 = values[1, 80], values[2, 80]
                row = {
                    "village": village,
                    "speaker": speaker,
                    "vowel": canonical_ipa(vowel),
                    "word": word,
                    "f1_20": round(f1_20),
                    "f2_20": round(f2_20),
                    "f1_50": round(f1_50),
                    "f2_50": round(f2_50),
                    "f1_80": round(f1_80),
                    "f2_80": round(f2_80),
                    "VL": round(math.hypot(f1_80 - f1_20, f2_80 - f2_20)),
                    "geo_x": x,
                    "geo_y": y,
                }
                if args.pca:
                    for percentage, times in measurement_times.items():
                        spectra = [
                            spectral_vector(
                                pca_samples, pca_rate, float(time), args.pca_window_ms,
                                args.pca_n_fft, pca_filters,
                            )[0]
                            for time in times
                        ]
                        row[f"_pca_{percentage}"] = np.median(np.asarray(spectra), axis=0)
                if args.fasttrack:
                    try:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter("always")
                            candidates = CandidateTracks(
                                sound=clip,
                                min_max_formant=args.fasttrack_min_ceiling,
                                max_max_formant=args.fasttrack_max_ceiling,
                                nstep=args.fasttrack_steps,
                                n_formants=4,
                                window_length=args.window_length,
                                time_step=args.time_step,
                                pre_emphasis_from=args.pre_emphasis,
                            )
                        for warning in caught_warnings:
                            print(
                                f"  FastTrack WARNING {stem} {word} "
                                f"[{start:.3f}-{end:.3f}s]: {warning.message}",
                                flush=True,
                            )
                        winner = candidates.winner
                        track_times = np.asarray(winner.time_domain, dtype=float)
                        track_f1 = np.asarray(winner.smoothed_formants[0], dtype=float)
                        track_f2 = np.asarray(winner.smoothed_formants[1], dtype=float)
                        fasttrack_values = {}
                        for percentage, times in measurement_times.items():
                            fasttrack_values[1, percentage] = finite_median(
                                np.interp(times, track_times, track_f1).tolist()
                            )
                            fasttrack_values[2, percentage] = finite_median(
                                np.interp(times, track_times, track_f2).tolist()
                            )
                        if any(value is None for value in fasttrack_values.values()):
                            raise ValueError("FastTrack produced no finite measurement")
                        ft_f1_20, ft_f2_20 = fasttrack_values[1, 20], fasttrack_values[2, 20]
                        ft_f1_50, ft_f2_50 = fasttrack_values[1, 50], fasttrack_values[2, 50]
                        ft_f1_80, ft_f2_80 = fasttrack_values[1, 80], fasttrack_values[2, 80]
                        row.update({
                            "ft_f1_20": round(ft_f1_20),
                            "ft_f2_20": round(ft_f2_20),
                            "ft_f1_50": round(ft_f1_50),
                            "ft_f2_50": round(ft_f2_50),
                            "ft_f1_80": round(ft_f1_80),
                            "ft_f2_80": round(ft_f2_80),
                            "ft_VL": round(math.hypot(
                                ft_f1_80 - ft_f1_20, ft_f2_80 - ft_f2_20
                            )),
                        })
                    except Exception as error:
                        failed_fasttrack += 1
                        print(f"  FastTrack ERROR {stem} {word}: {type(error).__name__}: {error}", flush=True)
                rows.append(row)
                written += 1
            print(f"[{index}/{len(recordings)}] {stem}", flush=True)

    if missing_coordinates:
        raise RuntimeError(f"Missing coordinates for: {', '.join(sorted(missing_coordinates))}")
    speaker_vowels: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        speaker_vowels.setdefault((row["village"], row["speaker"]), set()).add(row["vowel"])
    all_vowels = {canonical_ipa(target) for target in VOWEL_IPA}
    for row in rows:
        row["complete"] = int(speaker_vowels[(row["village"], row["speaker"])] == all_vowels)
    if args.pca:
        # Equalize lexical-token imbalance before fitting: every
        # speaker-vowel-time cell contributes one mean spectrum.
        balanced: dict[tuple[str, str, str, int], list[np.ndarray]] = {}
        for row in rows:
            for percentage in (20, 50, 80):
                key = (row["village"], row["speaker"], row["vowel"], percentage)
                balanced.setdefault(key, []).append(row[f"_pca_{percentage}"])
        matrix = np.asarray([
            np.mean(vectors, axis=0) for _, vectors in sorted(balanced.items())
        ])
        pca_mean = matrix.mean(axis=0)
        _, _, components = np.linalg.svd(matrix - pca_mean, full_matrices=False)
        for component in components:
            if component[np.argmax(np.abs(component))] < 0:
                component *= -1
        for row in rows:
            scores = {}
            for percentage in (20, 50, 80):
                score = (row.pop(f"_pca_{percentage}") - pca_mean) @ components[:2].T
                scores[percentage] = score
                row[f"pca_pc1_{percentage}"] = float(score[0])
                row[f"pca_pc2_{percentage}"] = float(score[1])
            row["pca_VL"] = float(np.linalg.norm(scores[80] - scores[20]))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                FIELDS
                + (FASTTRACK_FIELDS if args.fasttrack else ())
                + (PCA_FIELDS if args.pca else ())
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {written} tokens to {args.output}; "
        f"annotation exclusions={skipped_annotations}, Praat failures={failed_formants}, "
        f"FastTrack failures={failed_fasttrack}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
