#!/usr/bin/env python3
"""Inventory base lexical word/phone targets per SweDia speaker.

This is deliberately narrower than the formant-analysis lexical map.  It only
checks the eight current base words, and it counts an occurrence only when the
word in .ord has exactly one vowel-like .seg interval inside it.  Surface .seg
labels are summarized in notes but are not used to reject a lexical target
unless --strict-seg-label is used.

The base target list is kept in one small constant so a second pass can add
alternatives such as blöt for /ø:/ without changing the counting code.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from convert_xwaves_to_textgrids import (
    build_matches,
    choose_unique_pass_matches,
    collect_annotations,
    collect_wavs,
    parse_xwaves,
    read_places,
    read_resource_places,
    recording_parts,
    status_for_match,
)


VOWEL_INITIALS = set("aeiouyAEIOUYäöåÄÖÅ29<")
BASE_TARGETS = [
    {"word": "sot", "target": "u:", "ipa": "u:", "seg": "u:"},
    {"word": "låt", "target": "o:", "ipa": "o:", "seg": "o:", "group": "regular"},
    {"word": "lat", "target": "A:", "ipa": "ɑ:", "seg": "A:"},
    {"word": "nät", "target": "ä:", "ipa": "æ:", "seg": "ä:"},
    {"word": "leta", "target": "e:", "ipa": "e:", "seg": "e:"},
    {"word": "typ", "target": "y:", "ipa": "y:", "seg": "y:"},
    {"word": "lus", "target": "U:", "ipa": "ʉ̟:", "seg": "U:"},
    {"word": "söt", "target": "ö:", "ipa": "ø:", "seg": "ö:"},
]
EXTRA_COUNTS = [
    {"word": "låt", "target": "o2:", "ipa": "o:", "seg": "o:", "group": "second", "header": "second_låt_o"},
    {"word": "lås", "target": "o_alt:", "ipa": "o:", "seg": "o:", "group": "alternative", "header": "lås_o"},
    {"word": "blöt", "target": "ö_alt:", "ipa": "ø:", "seg": "ö:", "group": "alternative", "header": "blöt_ø"},
    {"word": "gles", "target": "e_alt:", "ipa": "e:", "seg": "e:", "group": "alternative", "header": "gles_e"},
    {"word": "lös", "target": "ö2_alt:", "ipa": "ø:", "seg": "ö:", "group": "alternative", "header": "lös_ø"},
    {"word": "läs", "target": "ä_alt:", "ipa": "æ:", "seg": "ä:", "group": "alternative", "header": "läs_æ"},
    {"word": "rot", "target": "u_alt:", "ipa": "u:", "seg": "u:", "group": "alternative", "header": "rot_u"},
    {"word": "fräs", "target": "ä2_alt:", "ipa": "æ:", "seg": "ä:", "group": "alternative", "header": "fräs_æ"},
    {"word": "gråt", "target": "o3_alt:", "ipa": "o:", "seg": "o:", "group": "alternative", "header": "gråt_o"},
    {"word": "båt", "target": "o4_alt:", "ipa": "o:", "seg": "o:", "group": "alternative", "header": "båt_o"},
]
OUTPUT_TARGETS = [
    *BASE_TARGETS,
    *EXTRA_COUNTS,
]
ALTERNATIVE_TARGET_MAP = {
    "u:": ["u_alt:"],
    "o:": ["o_alt:", "o3_alt:", "o4_alt:"],
    "ä:": ["ä_alt:", "ä2_alt:"],
    "e:": ["e_alt:"],
    "ö:": ["ö_alt:", "ö2_alt:"],
}
ALLOWED_SURFACE_TARGETS = {
    ("söt", "ø:", "o_e:"),
    ("blöt", "ø:", "o_e:"),
    ("lös", "ø:", "o_e:"),
    ("söt", "ø:", "Ö:"),
    ("nät", "æ:", "Ä:"),
    ("blöt", "ø:", "Ö:"),
    ("sot", "u:", "o:"),
    ("lat", "ɑ:", "a:"),
    ("lös", "ø:", "Ö:"),
    ("lås", "o:", "O:"),
    ("typ", "y:", "i:"),
    ("leta", "e:", "ei"),
    ("läs", "æ:", "Ä:"),
    ("typ", "y:", "U:"),
    ("lus", "ʉ̟:", "u:"),
    ("låt", "o:", "O:"),
    ("lat", "ɑ:", "O:"),
    ("leta", "e:", "ä:"),
}

SOURCE_PAIRS = {
    "media_annotations": [("Media/Annotations", Path("Media"), Path("Annotations"))],
    "sounds_sannotations": [("sounds/sannotations", Path("sounds"), Path("sannotations"))],
    "auto": [
        ("sounds/sannotations", Path("sounds"), Path("sannotations")),
        ("Media/Annotations", Path("Media"), Path("Annotations")),
    ],
}


def normalized_word(word: str) -> str:
    return word.strip().casefold()


def looks_vowel_like(label: str) -> bool:
    return bool(label) and (label[0] in VOWEL_INITIALS or label.startswith("\\}"))


def target_column(target: dict) -> str:
    if target.get("group") == "second":
        return f'second {target["word"]} - /{target["ipa"]}/'
    if target.get("group") == "alternative":
        return f'alternative {target["word"]} - /{target["ipa"]}/'
    return f'{target["word"]} - /{target["ipa"]}/'


def compact_target_column(target: dict) -> str:
    if "header" in target:
        return target["header"]
    return f'{target["word"]}_{target["target"].replace(":", "").replace("A", "ɑ").replace("ä", "æ").replace("U", "ʉ").replace("ö", "ø")}'


def read_original_codes(path: Path) -> set[str]:
    return {code for code, _ in read_places(path)}


def places_for_scope(scope: str, places_path: Path, resource_path: Path) -> list[dict]:
    original = [{"code": code, "place": place, "hints": [], "section": "Orter_SweDia"} for code, place in read_places(places_path)]
    if scope == "original":
        return original

    if not resource_path.exists():
        raise FileNotFoundError(f"{resource_path} is required for --village-scope {scope}")

    original_codes = read_original_codes(places_path)
    resource = [
        {"code": code, "place": place, "hints": [resource_stem],
         "section": "Orter_SweDia" if code in original_codes else "rest"}
        for code, place, resource_stem in read_resource_places(resource_path)
    ]
    if scope == "all":
        return resource
    if scope == "original_rest":
        return sorted(resource, key=lambda row: (row["section"] != "Orter_SweDia", row["place"]))
    raise ValueError(f"unknown village scope: {scope}")


def matches_for_places(places: list[dict], source_mode: str) -> tuple[list[dict], dict[str, str]]:
    pairs = SOURCE_PAIRS[source_mode]
    place_tuples = [(row["code"], row["place"], row["hints"]) for row in places]
    if source_mode != "auto":
        _, media_dir, annotation_dir = pairs[0]
        matches, _ = build_matches(
            place_tuples,
            collect_wavs([media_dir]),
            collect_annotations([annotation_dir], "ord"),
            collect_annotations([annotation_dir], "seg"),
        )
        return matches, {match["code"]: pairs[0][0] for match in matches}

    parsed_wavs = collect_wavs([media for _, media, _ in pairs])
    ord_annotations = collect_annotations([annotations for _, _, annotations in pairs], "ord")
    seg_annotations = collect_annotations([annotations for _, _, annotations in pairs], "seg")
    default_matches, _ = build_matches(place_tuples, parsed_wavs, ord_annotations, seg_annotations)
    source_matches = {
        label: build_matches(
            place_tuples,
            collect_wavs([media_dir]),
            collect_annotations([annotation_dir], "ord"),
            collect_annotations([annotation_dir], "seg"),
        )[0]
        for label, media_dir, annotation_dir in pairs
    }
    selected, provenance, _ = choose_unique_pass_matches(default_matches, source_matches)
    for match in selected:
        provenance.setdefault(match["code"], "")
    return selected, provenance


def source_dirs(label: str, source_mode: str) -> tuple[Path | None, Path | None]:
    if label == "Media/Annotations":
        return Path("Media"), Path("Annotations")
    if label == "sounds/sannotations":
        return Path("sounds"), Path("sannotations")
    if source_mode != "auto":
        _, media_dir, annotation_dir = SOURCE_PAIRS[source_mode][0]
        return media_dir, annotation_dir
    return None, None


def choose_existing_annotation(stem: str, extension: str, preferred: Path | None) -> Path | None:
    candidates = []
    if preferred is not None:
        candidates.append(preferred / f"{stem}.{extension}")
    candidates.extend([Path("sannotations") / f"{stem}.{extension}", Path("Annotations") / f"{stem}.{extension}"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def second_lat_window(words: list[tuple[float, float, str]]) -> tuple[float, float] | None:
    """Return a likely time window for the later length-contrast låt group.

    In these lists the second låt group generally occurs after vägg and before
    lott.  If that frame is absent, return None and count all låt as regular.
    """
    normalized = [(start, end, normalized_word(word)) for start, end, word in words]
    vagg_end = max((end for _, end, word in normalized if word == "vägg"), default=None)
    if vagg_end is None:
        return None
    lott_start = min((start for start, _, word in normalized if word == "lott" and start > vagg_end), default=None)
    if lott_start is None:
        return None
    return vagg_end, lott_start


def allowed_surface_target(word: str, target: dict, seg_label: str, enabled: bool) -> bool:
    return enabled and (normalized_word(word), target["ipa"], seg_label) in ALLOWED_SURFACE_TARGETS


def count_base_targets(
    ord_path: Path,
    seg_path: Path,
    strict_seg_label: bool = False,
    allow_surface_targets: bool = False,
) -> tuple[dict[str, int], dict[str, int], int, dict[str, str]]:
    words = parse_xwaves(ord_path).intervals
    segments = parse_xwaves(seg_path).intervals
    targets_by_word = {target["word"]: target for target in EXTRA_COUNTS}
    targets_by_word.update({target["word"]: target for target in BASE_TARGETS})
    second_låt_target = EXTRA_COUNTS[0]
    counts = {target["target"]: 0 for target in BASE_TARGETS + EXTRA_COUNTS}
    surface_counts_valid = {target["target"]: 0 for target in BASE_TARGETS + EXTRA_COUNTS}
    surface_alt_total = 0
    notes: dict[str, list[str]] = defaultdict(list)
    surface_counts: dict[str, defaultdict[str, int]] = {
        target["target"]: defaultdict(int) for target in BASE_TARGETS + EXTRA_COUNTS
    }
    second_window = second_lat_window(words)

    for word_start, word_end, word in words:
        target = targets_by_word.get(normalized_word(word))
        if target is None:
            continue
        count_target = target
        if normalized_word(word) == "låt" and second_window is not None:
            midpoint = (word_start + word_end) / 2
            if second_window[0] <= midpoint <= second_window[1]:
                count_target = second_låt_target
        vowel_segments = [
            (start, end, label)
            for start, end, label in segments
            if word_start <= (start + end) / 2 <= word_end and looks_vowel_like(label)
        ]
        if len(vowel_segments) != 1:
            notes[count_target["target"]].append(f"{word}:{len(vowel_segments)} vowel segments")
            continue
        _, _, seg_label = vowel_segments[0]
        surface_counts[count_target["target"]][seg_label] += 1
        if strict_seg_label and seg_label != count_target["seg"]:
            if allowed_surface_target(word, count_target, seg_label, allow_surface_targets):
                surface_counts_valid[count_target["target"]] += 1
                surface_alt_total += 1
                notes[count_target["target"]].append(f"{word}:{seg_label}=allowed")
            else:
                notes[count_target["target"]].append(f"{word}:{seg_label}≠{count_target['seg']}")
            continue
        counts[count_target["target"]] += 1

    for target in BASE_TARGETS + EXTRA_COUNTS:
        target_label = target["target"]
        surfaces = surface_counts[target_label]
        if surfaces:
            surface_summary = ",".join(f"{label}:{count}" for label, count in sorted(surfaces.items()))
            notes[target_label].append(f"surface={surface_summary}")

    return counts, surface_counts_valid, surface_alt_total, {target: "; ".join(values) for target, values in notes.items()}


def rating(counts: dict[str, int], min_reps: int) -> str:
    present = sum(counts[target["target"]] > 0 for target in BASE_TARGETS)
    complete = sum(counts[target["target"]] >= min_reps for target in BASE_TARGETS)
    if complete == len(BASE_TARGETS):
        return "high"
    if present == len(BASE_TARGETS):
        return "lowr"
    if present:
        return "part"
    return "none"


def alternative_rating(counts: dict[str, int], min_reps: int) -> str:
    adjusted = {}
    for target in BASE_TARGETS:
        target_label = target["target"]
        adjusted[target_label] = counts[target_label] + sum(
            counts[alternative] for alternative in ALTERNATIVE_TARGET_MAP.get(target_label, [])
        )
    return rating(adjusted, min_reps)


def surface_rating(counts: dict[str, int], surface_counts_valid: dict[str, int], min_reps: int) -> str:
    adjusted = {}
    for target in BASE_TARGETS:
        target_label = target["target"]
        adjusted[target_label] = (
            counts[target_label]
            + surface_counts_valid[target_label]
            + sum(
                counts[alternative] + surface_counts_valid[alternative]
                for alternative in ALTERNATIVE_TARGET_MAP.get(target_label, [])
            )
        )
    return rating(adjusted, min_reps)


def speaker_sort_key(speaker_id: str) -> tuple:
    parts = speaker_id.split("_")
    key = []
    for part in parts:
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(key)


def build_rows(args: argparse.Namespace) -> list[dict]:
    places = places_for_scope(args.village_scope, args.places, args.resource)
    section_by_code = {row["code"]: row["section"] for row in places}
    matches, provenance = matches_for_places(places, args.source_mode)
    rows = []

    for match in matches:
        source_label = provenance.get(match["code"], "")
        _, annotation_dir = source_dirs(source_label, args.source_mode)
        for speaker in sorted(match["speakers"], key=lambda row: speaker_sort_key(row["speaker"])):
            if speaker["wav"] == "MISSING" or speaker["ord"] == "MISSING" or speaker["seg"] == "MISSING":
                continue
            stem = Path(speaker["wav"]).stem
            ord_path = choose_existing_annotation(stem, "ord", annotation_dir)
            seg_path = choose_existing_annotation(stem, "seg", annotation_dir)
            if ord_path is None or seg_path is None:
                continue
            counts, surface_counts_valid, surface_alt_total, notes = count_base_targets(
                ord_path,
                seg_path,
                strict_seg_label=args.strict_seg_label,
                allow_surface_targets=args.allow_surface_targets,
            )
            parts = recording_parts(Path(stem))
            speaker_id = speaker["speaker"]
            if parts:
                _, kind, suffix = parts
                speaker_id = f"{kind}_{suffix[0]}"
            compact_id = f'{match["code"]}_{speaker_id}'
            row = {
                "id": compact_id,
                "|": "|",
                "section": section_by_code.get(match["code"], ""),
                "village_abbr": match["code"],
                "village": match["place"],
                "recording": stem,
                "speaker": speaker_id,
                "source": source_label,
                "rating": rating(counts, args.min_reps),
                "alt_rating": alternative_rating(counts, args.min_reps),
                "surf_rating": surface_rating(counts, surface_counts_valid, args.min_reps),
                "surface_alt": surface_alt_total,
                "complete_targets": sum(counts[target["target"]] >= args.min_reps for target in BASE_TARGETS),
                "present_targets": sum(counts[target["target"]] > 0 for target in BASE_TARGETS),
            }
            for target in BASE_TARGETS:
                row[target_column(target)] = counts[target["target"]]
            for target in EXTRA_COUNTS:
                row[target_column(target)] = counts[target["target"]]
            note_values = [
                f'{target_column(target)}: {notes[target["target"]]}'
                for target in BASE_TARGETS + EXTRA_COUNTS
                if notes.get(target["target"])
            ]
            row["notes"] = " | ".join(note_values)
            rows.append(row)

    section_order = {"Orter_SweDia": 0, "rest": 1}
    return sorted(rows, key=lambda row: (
        section_order.get(row["section"], 2),
        row["village"],
        speaker_sort_key(row["speaker"]),
        row["recording"],
    ))


def assessment_lines(rows: list[dict]) -> list[str]:
    base_high = sum(row["rating"] == "high" for row in rows)
    alt_high = sum(row["alt_rating"] == "high" for row in rows)
    surf_high = sum(row["surf_rating"] == "high" for row in rows)
    moved_high = sum(row["rating"] != "high" and row["alt_rating"] == "high" for row in rows)
    surface_moved_high = sum(row["alt_rating"] != "high" and row["surf_rating"] == "high" for row in rows)
    total_moved_high = sum(row["rating"] != "high" and row["surf_rating"] == "high" for row in rows)
    return [
        "",
        "assessment",
        f"speakers {len(rows)}",
        f"base_high {base_high}",
        f"alt_high {alt_high}",
        f"surf_high {surf_high}",
        f"moved_to_high {moved_high}",
        f"surface_moved_to_high {surface_moved_high}",
        f"total_moved_to_high {total_moved_high}",
    ]


def write_tsv(path: Path, rows: list[dict], detailed: bool = False) -> None:
    compact_fields = ["id", "rating", "alt_rating", "surf_rating"] + [
        target_column(target) for target in BASE_TARGETS
    ] + ["|"] + [target_column(target) for target in EXTRA_COUNTS] + ["surface_alt"]
    compact_header = ["speaker", "base", "alt", "surf"] + [
        compact_target_column(target) for target in BASE_TARGETS
    ] + ["|"] + [compact_target_column(target) for target in EXTRA_COUNTS] + ["surface_alt"]
    detailed_fields = [
        "section", "village_abbr", "village", "recording", "speaker", "source", "rating", "alt_rating", "surf_rating",
        "complete_targets", "present_targets",
    ] + [target_column(target) for target in OUTPUT_TARGETS] + ["surface_alt", "notes"]
    fields = detailed_fields if detailed else compact_fields
    path.parent.mkdir(parents=True, exist_ok=True)
    if detailed:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return

    with path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(compact_header) + "\n")
        for row in rows:
            handle.write(" ".join(str(row[field]) for field in fields) + "\n")
        handle.write("\n".join(assessment_lines(rows)) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--places", type=Path, default=Path("Orter_SweDia.csv"))
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument(
        "--village-scope",
        choices=["original", "original_rest", "all"],
        default="original",
        help=(
            "original = Orter_SweDia.csv only; original_rest = all resource villages with section labels; "
            "all = all resource villages without special filtering."
        ),
    )
    parser.add_argument(
        "--source-mode",
        choices=["auto", "media_annotations", "sounds_sannotations"],
        default="auto",
        help="Which wav/annotation directory pair to inspect.",
    )
    parser.add_argument("--min-reps", type=int, default=3, help="Repetitions per base target required for high rating.")
    parser.add_argument(
        "--strict-seg-label",
        action="store_true",
        help="Require the observed .seg vowel label to match the canonical base label exactly.",
    )
    parser.add_argument(
        "--allow-surface-targets",
        action="store_true",
        help=(
            "With --strict-seg-label, also count the explicitly listed word+surface-label mappings "
            "toward the separate surf rating and aggregate surface_alt count."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("Analyses/base_word_target_inventory.tsv"))
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Write the longer table with village/source metadata and surface-label notes.",
    )
    args = parser.parse_args()

    rows = build_rows(args)
    write_tsv(args.output, rows, detailed=args.detailed)
    print(f"Wrote {len(rows)} speaker rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
