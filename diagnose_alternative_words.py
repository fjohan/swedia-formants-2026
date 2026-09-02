#!/usr/bin/env python3
"""Diagnose possible alternative lexical words for the eight base vowel targets.

This script is read-only with respect to the source data.  It scans both
SweDia directory pairs directly:

    Media + Annotations
    sounds + sannotations

It does not require Analyses/base_word_target_inventory.tsv.

Candidate logic is intentionally conservative by default:

* the word must not be one of the base words;
* the first orthographic vowel letter in the word must match the base word's
  target letter, e.g. å for låt, ö for söt, a for lat;
* the word interval must contain exactly one vowel-like .seg interval;
* the .seg label must match the canonical base target label exactly, unless
  --surface-mode variants or --surface-mode any is used.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from convert_xwaves_to_textgrids import parse_xwaves, recording_parts


VOWEL_LETTERS = set("aeiouyåäöAEIOUYÅÄÖ")
VOWEL_INITIALS = set("aeiouyAEIOUYäöåÄÖÅ29<")

BASE_TARGETS = [
    {"word": "sot", "target": "u:", "ipa": "u:", "letter": "o", "seg": "u:"},
    {"word": "låt", "target": "o:", "ipa": "o:", "letter": "å", "seg": "o:"},
    {"word": "lat", "target": "A:", "ipa": "ɑ:", "letter": "a", "seg": "A:"},
    {"word": "nät", "target": "ä:", "ipa": "æ:", "letter": "ä", "seg": "ä:"},
    {"word": "leta", "target": "e:", "ipa": "e:", "letter": "e", "seg": "e:"},
    {"word": "typ", "target": "y:", "ipa": "y:", "letter": "y", "seg": "y:"},
    {"word": "lus", "target": "U:", "ipa": "ʉ̟:", "letter": "u", "seg": "U:"},
    {"word": "söt", "target": "ö:", "ipa": "ø:", "letter": "ö", "seg": "ö:"},
]

KNOWN_SURFACE_VARIANTS = {
    "u:": {"u:", "o:"},
    "o:": {"o:", "O:"},
    "A:": {"A:", "O:"},
    "ä:": {"ä:", "Ä:", "E:", "ä~:"},
    "e:": {"e:", "ä:", "ei"},
    "y:": {"y:", "U:"},
    "U:": {"U:"},
    "ö:": {"ö:", "Ö:", "o_e"},
}

SOURCE_PAIRS = [
    ("Media/Annotations", Path("Media"), Path("Annotations")),
    ("sounds/sannotations", Path("sounds"), Path("sannotations")),
]


def normalized_word(word: str) -> str:
    return word.strip().casefold()


def looks_vowel_like(label: str) -> bool:
    return bool(label) and (label[0] in VOWEL_INITIALS or label.startswith("\\}"))


def first_vowel_letter(word: str) -> str:
    for char in normalized_word(word):
        if char in VOWEL_LETTERS:
            return char
    return ""


def compatible_surface(seg_label: str, target: dict, surface_mode: str) -> bool:
    if surface_mode == "any":
        return True
    if surface_mode == "variants":
        return seg_label in KNOWN_SURFACE_VARIANTS[target["target"]]
    return seg_label == target["seg"]


def speaker_id(recording: str) -> str:
    parts = recording_parts(Path(recording))
    if parts is None:
        return ""
    _, kind, suffix = parts
    return f"{kind}_{suffix[0]}"


def scan_recording(
    source_label: str,
    recording: str,
    ord_path: Path,
    seg_path: Path,
    surface_mode: str,
    include_base: bool,
) -> list[dict]:
    words = parse_xwaves(ord_path).intervals
    segments = parse_xwaves(seg_path).intervals
    base_words = {target["word"] for target in BASE_TARGETS}
    rows = []

    for word_start, word_end, word in words:
        word_key = normalized_word(word)
        if not include_base and word_key in base_words:
            continue
        letter = first_vowel_letter(word_key)
        if not letter:
            continue
        vowel_segments = [
            (start, end, label)
            for start, end, label in segments
            if word_start <= (start + end) / 2 <= word_end and looks_vowel_like(label)
        ]
        if len(vowel_segments) != 1:
            continue
        seg_start, seg_end, seg_label = vowel_segments[0]
        for target in BASE_TARGETS:
            if letter != target["letter"]:
                continue
            if not compatible_surface(seg_label, target, surface_mode):
                continue
            rows.append({
                "source": source_label,
                "recording": recording,
                "speaker": speaker_id(recording),
                "word": word_key,
                "target": target["target"],
                "ipa": target["ipa"],
                "base_word": target["word"],
                "letter": target["letter"],
                "surface_seg_label": seg_label,
                "word_start_s": word_start,
                "word_end_s": word_end,
                "seg_start_s": seg_start,
                "seg_end_s": seg_end,
                "duration_s": seg_end - seg_start,
            })
    return rows


def scan_sources(args: argparse.Namespace) -> list[dict]:
    rows = []
    for source_label, media_dir, annotation_dir in SOURCE_PAIRS:
        if not media_dir.exists() or not annotation_dir.exists():
            continue
        wav_stems = {path.stem for path in media_dir.glob("*.wav")}
        ord_stems = {path.stem for path in annotation_dir.glob("*.ord")}
        seg_stems = {path.stem for path in annotation_dir.glob("*.seg")}
        for stem in sorted(wav_stems & ord_stems & seg_stems):
            rows.extend(scan_recording(
                source_label,
                stem,
                annotation_dir / f"{stem}.ord",
                annotation_dir / f"{stem}.seg",
                args.surface_mode,
                args.include_base,
            ))
    return rows


def summarize(rows: list[dict], min_tokens: int) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["target"], row["ipa"], row["base_word"], row["letter"], row["word"])].append(row)

    summary = []
    for (target, ipa, base_word, letter, word), values in grouped.items():
        source_counts = Counter(row["source"] for row in values)
        surface_counts = Counter(row["surface_seg_label"] for row in values)
        recordings = sorted({row["recording"] for row in values})
        speakers = sorted({f'{row["recording"].rsplit("_", 1)[0]}_{row["speaker"]}' for row in values if row["speaker"]})
        if len(values) < min_tokens:
            continue
        summary.append({
            "target": target,
            "ipa": f"/{ipa}/",
            "base_word": base_word,
            "letter": letter,
            "candidate_word": word,
            "token_count": len(values),
            "recording_count": len(recordings),
            "speaker_count": len(speakers),
            "sources": ",".join(f"{source}:{count}" for source, count in sorted(source_counts.items())),
            "surface_seg_labels": ",".join(f"{label}:{count}" for label, count in sorted(surface_counts.items())),
            "example_recordings": ",".join(recordings[:12]),
        })
    return sorted(summary, key=lambda row: (-row["recording_count"], -row["token_count"], row["candidate_word"]))


def write_summary(path: Path, rows: list[dict]) -> None:
    word_width = max((len(row["candidate_word"]) for row in rows), default=4)
    token_width = max((len(str(row["token_count"])) for row in rows), default=1)
    recording_width = max((len(str(row["recording_count"])) for row in rows), default=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f'{row["candidate_word"]:<{word_width}} -> {row["ipa"]:<5} '
                f'{row["token_count"]:>{token_width}} tokens, '
                f'{row["recording_count"]:>{recording_width}} recordings\n'
            )


def write_table_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "target", "ipa", "base_word", "letter", "candidate_word",
        "token_count", "recording_count", "speaker_count",
        "sources", "surface_seg_labels", "example_recordings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_tokens(path: Path, rows: list[dict]) -> None:
    fields = [
        "source", "recording", "speaker", "word", "target", "ipa", "base_word", "letter",
        "surface_seg_label", "word_start_s", "word_end_s", "seg_start_s", "seg_end_s", "duration_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface-mode",
        choices=["exact", "variants", "any"],
        default="exact",
        help=(
            "exact = require canonical base .seg label; variants = allow known surface variants; "
            "any = require only one vowel-like segment with the same orthographic target letter."
        ),
    )
    parser.add_argument("--include-base", action="store_true", help="Include the eight base words in the diagnosis.")
    parser.add_argument("--min-tokens", type=int, default=1, help="Minimum corpus token count for a candidate word.")
    parser.add_argument("--output", type=Path, default=Path("Analyses/alternative_word_diagnosis.tsv"))
    parser.add_argument("--table-output", type=Path, help="Optional TSV summary with metadata columns.")
    parser.add_argument(
        "--token-output",
        type=Path,
        help="Optional detailed token-level output for auditing candidate rows.",
    )
    args = parser.parse_args()

    token_rows = scan_sources(args)
    summary_rows = summarize(token_rows, args.min_tokens)
    write_summary(args.output, summary_rows)
    if args.table_output:
        write_table_summary(args.table_output, summary_rows)
    if args.token_output:
        write_tokens(args.token_output, token_rows)
    print(f"Scanned {len(token_rows)} matching tokens")
    print(f"Wrote {len(summary_rows)} candidate rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
