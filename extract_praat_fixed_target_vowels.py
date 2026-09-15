#!/usr/bin/env python3
"""Extract token-level fixed-ceiling Praat F1/F2 for all target vowels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import parselmouth

from analyze_bark_pca_pilot import VOWEL_IPA, find_vowel_segment, lexical_targets, locate_recording
from inventory_base_word_targets import (
    BASE_TARGETS,
    normalized_word,
    parse_textgrid_tier,
    second_lat_window,
)
from plot_pca_angle_maps import RESOURCE_ALIASES, read_coordinates


FIELDS = ("village", "speaker", "vowel", "word", "f1", "f2", "x", "y", "complete")


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


def discover_recordings(textgrid_dir: Path, media_dirs: list[Path]) -> list[str]:
    annotated = {path.stem for path in textgrid_dir.glob("*.TextGrid")}
    audio = {
        path.stem
        for directory in media_dirs
        if directory.exists()
        for path in directory.glob("*.wav")
    }
    return sorted(stem for stem in annotated & audio if "_ym_" in stem)


def finite_median(values: list[float | None]) -> float | None:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


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
        "--strict-seg-label", action="store_true",
        help="Require the surface segment label to equal the base target label.",
    )
    parser.add_argument("--limit-recordings", type=int, default=0, help="Testing only; zero means all.")
    args = parser.parse_args()

    mapping = lexical_targets()
    expected = {target["target"]: target["seg"] for target in BASE_TARGETS}
    coordinates = read_coordinates(args.resource)
    recordings = discover_recordings(args.textgrids, args.media_dirs)
    if args.limit_recordings:
        recordings = recordings[: args.limit_recordings]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_annotations = failed_formants = 0
    missing_coordinates: set[str] = set()

    rows: list[dict] = []
    for index, stem in enumerate(recordings, 1):
            village, speaker = recording_parts(stem)
            resource_key = RESOURCE_ALIASES.get(village, village)
            if resource_key not in coordinates:
                missing_coordinates.add(village)
                continue
            x, y, _ = coordinates[resource_key]
            sound = parselmouth.Sound(str(locate_recording(stem, args.media_dirs)))
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
                times = np.linspace(start + 0.45 * (end - start), start + 0.55 * (end - start), 11)
                f1 = finite_median([formant.get_value_at_time(1, time) for time in times])
                f2 = finite_median([formant.get_value_at_time(2, time) for time in times])
                if f1 is None or f2 is None:
                    failed_formants += 1
                    continue
                rows.append({
                    "village": village,
                    "speaker": speaker,
                    "vowel": canonical_ipa(vowel),
                    "word": word,
                    "f1": round(f1),
                    "f2": round(f2),
                    "x": x,
                    "y": y,
                })
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
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {written} tokens to {args.output}; "
        f"annotation exclusions={skipped_annotations}, formant failures={failed_formants}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
