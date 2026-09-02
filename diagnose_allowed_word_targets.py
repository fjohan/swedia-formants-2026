#!/usr/bin/env python3
"""Diagnose observed .seg vowel targets for allowed inventory words.

This scans both SweDia source pairs directly:

    Media + Annotations
    sounds + sannotations

It uses only the unique words allowed by inventory_base_word_targets.py.  The
purpose is finer control than --strict-seg-label: for each allowed word, report
which vowel-like .seg labels actually occur inside that word.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from convert_xwaves_to_textgrids import parse_xwaves, recording_parts
from inventory_base_word_targets import BASE_TARGETS, EXTRA_COUNTS


VOWEL_INITIALS = set("aeiouyAEIOUYäöåÄÖÅ29<")
SOURCE_PAIRS = [
    ("Media/Annotations", Path("Media"), Path("Annotations")),
    ("sounds/sannotations", Path("sounds"), Path("sannotations")),
]


def normalized_word(word: str) -> str:
    return word.strip().casefold()


def looks_vowel_like(label: str) -> bool:
    return bool(label) and (label[0] in VOWEL_INITIALS or label.startswith("\\}"))


def allowed_words() -> dict[str, dict]:
    """Unique allowed words from the inventory script.

    inventory_base_word_targets.py has 18 count definitions because låt is
    counted twice positionally, but those definitions contain 17 unique words.
    For target diagnosis we want lexical words, so låt appears once.
    """
    words: dict[str, dict] = {}
    for target in BASE_TARGETS + EXTRA_COUNTS:
        words.setdefault(target["word"], target)
    return words


def speaker_id(recording: str) -> str:
    parts = recording_parts(Path(recording))
    if parts is None:
        return ""
    _, kind, suffix = parts
    return f"{kind}_{suffix[0]}"


def scan_recording(source: str, recording: str, ord_path: Path, seg_path: Path, words_by_key: dict[str, dict]) -> list[dict]:
    words = parse_xwaves(ord_path).intervals
    segments = parse_xwaves(seg_path).intervals
    rows = []
    for word_start, word_end, word in words:
        word_key = normalized_word(word)
        target = words_by_key.get(word_key)
        if target is None:
            continue
        vowel_segments = [
            (start, end, label)
            for start, end, label in segments
            if word_start <= (start + end) / 2 <= word_end and looks_vowel_like(label)
        ]
        if len(vowel_segments) != 1:
            rows.append({
                "source": source,
                "recording": recording,
                "speaker": speaker_id(recording),
                "word": word_key,
                "expected_ipa": target["ipa"],
                "expected_seg": target["seg"],
                "surface_seg_label": f"{len(vowel_segments)}_vowel_segments",
            })
            continue
        _, _, seg_label = vowel_segments[0]
        rows.append({
            "source": source,
            "recording": recording,
            "speaker": speaker_id(recording),
            "word": word_key,
            "expected_ipa": target["ipa"],
            "expected_seg": target["seg"],
            "surface_seg_label": seg_label,
        })
    return rows


def scan_sources() -> list[dict]:
    words_by_key = allowed_words()
    rows = []
    for source, media_dir, annotation_dir in SOURCE_PAIRS:
        if not media_dir.exists() or not annotation_dir.exists():
            continue
        wav_stems = {path.stem for path in media_dir.glob("*.wav")}
        ord_stems = {path.stem for path in annotation_dir.glob("*.ord")}
        seg_stems = {path.stem for path in annotation_dir.glob("*.seg")}
        for stem in sorted(wav_stems & ord_stems & seg_stems):
            rows.extend(scan_recording(
                source,
                stem,
                annotation_dir / f"{stem}.ord",
                annotation_dir / f"{stem}.seg",
                words_by_key,
            ))
    return rows


def summarize(rows: list[dict], include_canonical: bool, min_tokens: int) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not include_canonical and row["surface_seg_label"] == row["expected_seg"]:
            continue
        grouped[(row["word"], row["expected_ipa"], row["expected_seg"], row["surface_seg_label"])].append(row)

    summary = []
    for (word, expected_ipa, expected_seg, surface), values in grouped.items():
        recordings = sorted({row["recording"] for row in values})
        sources = Counter(row["source"] for row in values)
        if len(values) < min_tokens:
            continue
        summary.append({
            "word": word,
            "expected_ipa": f"/{expected_ipa}/",
            "expected_seg": expected_seg,
            "surface_seg_label": surface,
            "token_count": len(values),
            "recording_count": len(recordings),
            "sources": ",".join(f"{source}:{count}" for source, count in sorted(sources.items())),
            "example_recordings": ",".join(recordings[:12]),
        })
    return sorted(summary, key=lambda row: (-row["recording_count"], -row["token_count"], row["word"], row["surface_seg_label"]))


def write_summary(path: Path, rows: list[dict]) -> None:
    word_width = max((len(row["word"]) for row in rows), default=4)
    surface_width = max((len(row["surface_seg_label"]) for row in rows), default=3)
    token_width = max((len(str(row["token_count"])) for row in rows), default=1)
    recording_width = max((len(str(row["recording_count"])) for row in rows), default=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f'{row["word"]:<{word_width}} -> {row["expected_ipa"]:<5} '
                f'{row["surface_seg_label"]:<{surface_width}} '
                f'{row["token_count"]:>{token_width}} tokens, '
                f'{row["recording_count"]:>{recording_width}} recordings\n'
            )


def write_table(path: Path, rows: list[dict]) -> None:
    fields = [
        "word", "expected_ipa", "expected_seg", "surface_seg_label",
        "token_count", "recording_count", "sources", "example_recordings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("Analyses/allowed_word_target_diagnosis.tsv"))
    parser.add_argument("--table-output", type=Path, help="Optional TSV metadata table.")
    parser.add_argument("--include-canonical", action="store_true", help="Also show rows where surface .seg equals expected .seg.")
    parser.add_argument("--min-tokens", type=int, default=1)
    args = parser.parse_args()

    token_rows = scan_sources()
    summary_rows = summarize(token_rows, args.include_canonical, args.min_tokens)
    write_summary(args.output, summary_rows)
    if args.table_output:
        write_table(args.table_output, summary_rows)
    print(f"Scanned {len(token_rows)} allowed-word tokens")
    print(f"Wrote {len(summary_rows)} surface-target rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
