#!/usr/bin/env python3
"""Match SweDia recordings and convert xwaves .ord/.seg labels to TextGrids."""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path


SPEAKER_RE = re.compile(r"^(?P<place>.+)_(?P<kind>[^_]+)_(?P<number>\d+)$")
ALIASES = {
    "norra_rorum": "n_rorum",
    "onaset_nysatra": "nysatra",
    "vastra_vingaker": "v_vingaker",
    "v_vingaker": "v_vingaker",
    "sankt_anna": "s_anna",
}


@dataclass
class ParseResult:
    intervals: list[tuple[float, float, str]]
    warnings: list[str]


def ascii_name(value: str) -> str:
    value = value.strip().lower().replace(".", " ")
    value = re.sub(r"[()]", " ", value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def read_places(path: Path) -> list[tuple[str, str]]:
    # These historical files use the classic Macintosh Swedish character set.
    text = path.read_bytes().decode("mac_roman")
    rows = []
    for row in csv.reader(text.splitlines(), delimiter=";"):
        if not row or not row[0].strip():
            continue
        if len(row) < 2:
            raise ValueError(f"Malformed row in {path}: {row!r}")
        rows.append((row[0].strip(), row[1].strip()))
    return rows


def recording_parts(path: Path) -> tuple[str, str, int] | None:
    match = SPEAKER_RE.match(path.stem)
    if not match:
        return None
    return match["place"], match["kind"], int(match["number"])


def choose_media_stem(place: str, stems: set[str]) -> tuple[str | None, str, float]:
    normalized = ascii_name(place)
    candidates = [normalized]
    if "_" in normalized:
        candidates.extend([normalized.split("_")[0], normalized.split("_")[-1]])
    alias = ALIASES.get(normalized)
    if alias:
        candidates.insert(0, alias)
    for candidate in candidates:
        if candidate in stems:
            return candidate, "exact", 1.0

    scored = sorted(
        ((difflib.SequenceMatcher(None, candidate, stem).ratio(), stem)
         for candidate in candidates for stem in stems),
        reverse=True,
    )
    if not scored:
        return None, "no media files", 0.0
    score, stem = scored[0]
    second = next((s for s, other in scored if other != stem), 0.0)
    if score >= 0.90 and score - second >= 0.05:
        return stem, "fuzzy", score
    return None, f"no confident match (closest: {stem}, {score:.2f})", score


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def parse_xwaves(path: Path) -> ParseResult:
    warnings: list[str] = []
    boundaries: list[tuple[float, str, int]] = []
    # Unlike the site list, xlabel's annotation output is ISO-8859-1.
    for line_no, raw in enumerate(path.read_bytes().decode("latin-1").splitlines(), 1):
        fields = raw.strip().split(maxsplit=2)
        if len(fields) < 3:
            continue
        try:
            time = float(fields[0])
        except ValueError:
            continue
        boundaries.append((time, fields[2], line_no))

    # Most files use #. A minority of .ord files use x throughout instead;
    # x is also a real segment label, so only treat it as a marker when # is absent.
    closing_marker = "#" if any(label == "#" for _, label, _ in boundaries) else "x"
    intervals: list[tuple[float, float, str]] = []
    pending: tuple[float, str, int] | None = None
    for time, label, line_no in boundaries:
        if label == closing_marker:
            if pending is None:
                warnings.append(f"line {line_no}: closing # without a label")
                continue
            start, text, start_line = pending
            if time <= start:
                warnings.append(f"lines {start_line}-{line_no}: end is not after start")
            else:
                intervals.append((start, time, text))
            pending = None
        else:
            if pending is not None:
                warnings.append(f"line {line_no}: new label before previous label was closed")
            pending = (time, label, line_no)
    if pending is not None:
        warnings.append(f"line {pending[2]}: label has no closing #")
    return ParseResult(intervals, warnings)


def contiguous(intervals: list[tuple[float, float, str]], duration: float) -> tuple[list[tuple[float, float, str]], list[str]]:
    result: list[tuple[float, float, str]] = []
    warnings: list[str] = []
    cursor = 0.0
    for start, end, label in sorted(intervals):
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            warnings.append(f"discarded out-of-range interval {start:g}-{end:g} {label!r}")
            continue
        if start < cursor:
            warnings.append(f"discarded overlapping interval {start:g}-{end:g} {label!r}")
            continue
        if start > cursor:
            result.append((cursor, start, ""))
        result.append((start, end, label))
        cursor = end
    if cursor < duration:
        result.append((cursor, duration, ""))
    if not result and duration > 0:
        result.append((0.0, duration, ""))
    return result, warnings


def quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def write_textgrid(path: Path, duration: float, tiers: list[tuple[str, list[tuple[float, float, str]]]]) -> None:
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', '',
        'xmin = 0', f'xmax = {duration:.12g}', 'tiers? <exists>',
        f'size = {len(tiers)}', 'item []:'
    ]
    for tier_no, (name, intervals) in enumerate(tiers, 1):
        lines += [
            f'    item [{tier_no}]:', '        class = "IntervalTier"',
            f'        name = {quote(name)}', '        xmin = 0',
            f'        xmax = {duration:.12g}', f'        intervals: size = {len(intervals)}'
        ]
        for interval_no, (start, end, text) in enumerate(intervals, 1):
            lines += [
                f'        intervals [{interval_no}]:', f'            xmin = {start:.12g}',
                f'            xmax = {end:.12g}', f'            text = {quote(text)}'
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--places", type=Path, default=Path("Orter_SweDia.csv"))
    parser.add_argument("--media", type=Path, default=Path("Media"))
    parser.add_argument("--annotations", type=Path, default=Path("Annotations"))
    parser.add_argument("--output", type=Path, default=Path("TextGrids"))
    parser.add_argument("--summary", type=Path, default=Path("sweDia_inventory.csv"))
    parser.add_argument("--compact-summary", type=Path, default=Path("sweDia_inventory_compact.tsv"))
    args = parser.parse_args()

    wavs = sorted(args.media.glob("*.wav"))
    parsed_wavs = [(path, recording_parts(path)) for path in wavs]
    stems = {parts[0] for _, parts in parsed_wavs if parts}
    places = read_places(args.places)
    matches = []
    max_speakers = 0
    args.output.mkdir(parents=True, exist_ok=True)

    for code, place in places:
        stem, method, score = choose_media_stem(place, stems)
        selected = sorted(
            ((parts[2], parts[1], path) for path, parts in parsed_wavs if parts and parts[0] == stem),
            key=lambda item: (item[0], item[1], item[2].name),
        ) if stem else []
        max_speakers = max(max_speakers, len(selected))
        speakers = []
        for number, kind, wav in selected:
            base = wav.stem
            ord_path = args.annotations / f"{base}.ord"
            seg_path = args.annotations / f"{base}.seg"
            duration = wav_duration(wav)
            tiers = []
            warnings = []
            for tier_name, annotation in (("ord", ord_path), ("seg", seg_path)):
                parsed = parse_xwaves(annotation) if annotation.exists() else ParseResult([], [])
                intervals, extra = contiguous(parsed.intervals, duration)
                tiers.append((tier_name, intervals))
                warnings.extend(f"{tier_name}: {warning}" for warning in parsed.warnings + extra)
            output = args.output / f"{base}.TextGrid"
            write_textgrid(output, duration, tiers)
            speakers.append({
                "speaker": f"{kind}_{number}", "wav": wav.name,
                "ord": "yes" if ord_path.exists() else "MISSING",
                "seg": "yes" if seg_path.exists() else "MISSING",
                "textgrid": output.name, "warnings": " | ".join(warnings),
            })
        matches.append({"code": code, "place": place, "stem": stem or "", "method": method,
                        "score": score, "speakers": speakers})

    fields = ["code", "village", "matched_stem", "match_method", "match_score", "recording_count"]
    for i in range(1, max_speakers + 1):
        fields += [f"speaker_{i}", f"speaker_{i}_wav", f"speaker_{i}_ord", f"speaker_{i}_seg",
                   f"speaker_{i}_textgrid", f"speaker_{i}_warnings"]
    with args.summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for match in matches:
            row = {"code": match["code"], "village": match["place"],
                   "matched_stem": match["stem"], "match_method": match["method"],
                   "match_score": f'{match["score"]:.3f}', "recording_count": len(match["speakers"])}
            for i, speaker in enumerate(match["speakers"], 1):
                for key, value in speaker.items():
                    row[f"speaker_{i}" if key == "speaker" else f"speaker_{i}_{key}"] = value
            writer.writerow(row)

    # A deliberately small, human-scannable inventory. Speaker columns follow
    # the numeric suffix in names such as asby_ym_2, rather than row position.
    max_speaker_number = max(
        (int(speaker["speaker"].rsplit("_", 1)[1])
         for match in matches for speaker in match["speakers"]),
        default=3,
    )
    compact_fields = ["abbr", "full_name"] + [f"sp{i}" for i in range(1, max_speaker_number + 1)]
    with args.compact_summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fields, delimiter="\t")
        writer.writeheader()
        for match in matches:
            row = {"abbr": match["code"], "full_name": match["place"]}
            for speaker in match["speakers"]:
                number = int(speaker["speaker"].rsplit("_", 1)[1])
                available = ["w"]
                if speaker["ord"] == "yes":
                    available.append("o")
                if speaker["seg"] == "yes":
                    available.append("s")
                row[f"sp{number}"] = ",".join(available)
            writer.writerow(row)

    generated = sum(len(match["speakers"]) for match in matches)
    missing_places = sum(not match["stem"] for match in matches)
    missing_ord = sum(s["ord"] == "MISSING" for m in matches for s in m["speakers"])
    missing_seg = sum(s["seg"] == "MISSING" for m in matches for s in m["speakers"])
    print(f"Generated {generated} TextGrids in {args.output}")
    print(f"Summary: {args.summary}")
    print(f"Compact summary: {args.compact_summary}")
    print(f"Unmatched villages: {missing_places}; missing ord: {missing_ord}; missing seg: {missing_seg}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, wave.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
