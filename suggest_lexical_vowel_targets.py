#!/usr/bin/env python3
"""Suggest lexical vowel-target additions from complete SweDia annotations."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from convert_xwaves_to_textgrids import (
    build_matches,
    choose_unique_pass_matches,
    collect_annotations,
    collect_wavs,
    parse_xwaves,
    read_resource_places,
)


TARGET_TO_IPA = {
    "u:": "uː",
    "o:": "oː",
    "A:": "ɑː",
    "ä:": "æː",
    "e:": "eː",
    "y:": "yː",
    "U:": "ʉ̟ː",
    "ö:": "øː",
}
VOWEL_INITIALS = set("aeiouyAEIOUYäöåÄÖÅ29<")
SURFACE_TO_TARGET = {
    "u:": "u:",
    "o:": "o:",
    "A:": "A:",
    "ä:": "ä:",
    "Ä:": "ä:",
    "e:": "e:",
    "E:": "e:",
    "y:": "y:",
    "U:": "U:",
    "ö:": "ö:",
    "Ö:": "ö:",
}


def read_lexical_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"lexical_item", "target_label", "ipa"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    return {row["lexical_item"].strip().lower(): row for row in rows}


def looks_vowel_like(label: str) -> bool:
    return bool(label) and (label[0] in VOWEL_INITIALS or label.startswith("\\}"))


def vowel_in_word(word_start: float, word_end: float, segments: list[tuple[float, float, str]]):
    candidates = [
        interval for interval in segments
        if word_start <= (interval[0] + interval[1]) / 2 <= word_end and looks_vowel_like(interval[2])
    ]
    return max(candidates, key=lambda interval: interval[1] - interval[0]) if candidates else None


def normalize_word(word: str) -> str:
    return word.strip().lower()


def find_annotation(stem: str, extension: str, preferred_dir: Path | None = None) -> Path | None:
    candidates = []
    if preferred_dir:
        candidates.append(preferred_dir / f"{stem}.{extension}")
    candidates.extend([Path("sannotations") / f"{stem}.{extension}", Path("Annotations") / f"{stem}.{extension}"])
    return next((path for path in candidates if path.exists()), None)


def selected_complete_recordings(resource: Path) -> list[tuple[str, str, str, str | None]]:
    resource_places = [
        (code, place, [resource_stem]) for code, place, resource_stem in read_resource_places(resource)
    ]
    parsed_wavs = collect_wavs([Path("sounds"), Path("Media")])
    all_matches, _ = build_matches(
        resource_places,
        parsed_wavs,
        collect_annotations([Path("sannotations"), Path("Annotations")], "ord"),
        collect_annotations([Path("sannotations"), Path("Annotations")], "seg"),
    )
    pair_matches = {}
    for label, media_dir, annotation_dir in (
        ("sounds/sannotations", Path("sounds"), Path("sannotations")),
        ("Media/Annotations", Path("Media"), Path("Annotations")),
    ):
        pair_matches[label], _ = build_matches(
            resource_places,
            collect_wavs([media_dir]),
            collect_annotations([annotation_dir], "ord"),
            collect_annotations([annotation_dir], "seg"),
        )
    selected_matches, provenance, _ = choose_unique_pass_matches(all_matches, pair_matches)

    rows = []
    for match in selected_matches:
        source = provenance.get(match["code"])
        for speaker in match["speakers"]:
            if speaker["ord"] == "MISSING" or speaker["seg"] == "MISSING":
                continue
            rows.append((match["code"], match["place"], speaker["wav"][:-4], source))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--lexical-map", type=Path, default=Path("lexical_vowel_targets.tsv"))
    parser.add_argument("--output", type=Path, default=Path("lexical_vowel_target_suggestions.tsv"))
    parser.add_argument("--coverage-output", type=Path, default=Path("lexical_vowel_target_coverage.tsv"))
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["target-labels", "all"],
        default="target-labels",
        help="target-labels suggests only words with recurring labels from the eight long-vowel targets.",
    )
    args = parser.parse_args()

    lexical_map = read_lexical_map(args.lexical_map)
    mapped_words = set(lexical_map)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    coverage: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[tuple[str, str], set[str]] = defaultdict(set)
    skipped = Counter()

    for code, _place, stem, source in selected_complete_recordings(args.resource):
        annotation_dir = None
        if source == "sounds/sannotations":
            annotation_dir = Path("sannotations")
        elif source == "Media/Annotations":
            annotation_dir = Path("Annotations")
        ord_path = find_annotation(stem, "ord", annotation_dir)
        seg_path = find_annotation(stem, "seg", annotation_dir)
        if not ord_path or not seg_path:
            skipped["missing_annotation_after_inventory"] += 1
            continue

        words = parse_xwaves(ord_path).intervals
        segments = parse_xwaves(seg_path).intervals
        for word_start, word_end, raw_word in words:
            word = normalize_word(raw_word)
            segment = vowel_in_word(word_start, word_end, segments)
            if segment is None:
                skipped["no_vowel_like_segment"] += 1
                continue
            label = segment[2].strip()
            if not looks_vowel_like(label):
                skipped["non_vowel_label"] += 1
                continue
            coverage[word][label] += 1
            examples[(word, label)].add(stem)
            if word not in mapped_words:
                counts[word][label] += 1

    suggestion_rows = []
    for word, label_counts in counts.items():
        total = sum(label_counts.values())
        candidate_counts = Counter({
            label: count for label, count in label_counts.items()
            if args.mode == "all" or label in SURFACE_TO_TARGET
        })
        candidate_total = sum(candidate_counts.values())
        if candidate_total < args.min_count:
            continue
        dominant_label, dominant_count = candidate_counts.most_common(1)[0]
        target = SURFACE_TO_TARGET.get(dominant_label, "")
        suggestion_rows.append({
            "lexical_item": word,
            "suggested_target_label": target,
            "suggested_ipa": TARGET_TO_IPA.get(target, ""),
            "dominant_surface_seg_label": dominant_label,
            "dominant_count": dominant_count,
            "candidate_count": candidate_total,
            "total_count": total,
            "candidate_surface_seg_counts": "; ".join(
                f"{label}:{count}" for label, count in candidate_counts.most_common()
            ),
            "surface_seg_counts": "; ".join(f"{label}:{count}" for label, count in label_counts.most_common()),
            "example_recordings": ", ".join(sorted(examples[(word, dominant_label)])[:8]),
        })
    suggestion_rows.sort(key=lambda row: (-int(row["candidate_count"]), -int(row["total_count"]), row["lexical_item"]))

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "lexical_item", "suggested_target_label", "suggested_ipa",
            "dominant_surface_seg_label", "dominant_count", "candidate_count", "total_count",
            "candidate_surface_seg_counts",
            "surface_seg_counts", "example_recordings",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(suggestion_rows)

    coverage_rows = []
    for word, label_counts in coverage.items():
        mapping = lexical_map.get(word)
        coverage_rows.append({
            "lexical_item": word,
            "mapped_target_label": mapping["target_label"] if mapping else "",
            "mapped_ipa": mapping["ipa"] if mapping else "",
            "is_mapped": "yes" if mapping else "no",
            "total_count": sum(label_counts.values()),
            "surface_seg_counts": "; ".join(f"{label}:{count}" for label, count in label_counts.most_common()),
        })
    coverage_rows.sort(key=lambda row: (row["is_mapped"], -int(row["total_count"]), row["lexical_item"]))
    with args.coverage_output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "lexical_item", "mapped_target_label", "mapped_ipa",
            "is_mapped", "total_count", "surface_seg_counts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(coverage_rows)

    print(f"Wrote {args.output} ({len(suggestion_rows)} suggestions with {args.mode} count >= {args.min_count})")
    print(f"Wrote {args.coverage_output} ({len(coverage_rows)} observed lexical items)")
    if skipped:
        print("Skipped: " + ", ".join(f"{key}={value}" for key, value in skipped.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
