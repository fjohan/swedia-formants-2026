#!/usr/bin/env python3
"""Match SweDia recordings and convert xwaves .ord/.seg labels to TextGrids."""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import shlex
import sys
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path


RECORDING_RE = re.compile(r"^(?P<place>.+)_(?P<kind>[a-z]+)_(?P<suffix>\d+(?:_\d+)*)$")
ALIASES = {
    "norra_rorum": "n_rorum",
    "onaset_nysatra": "nysatra",
    "vastra_vingaker": "v_vingaker",
    "v_vingaker": "v_vingaker",
    "sankt_anna": "st_anna",
    "nrorum": "n_rorum",
    "sodrafinnskoga": "s_finnskoga",
    "stanna": "st_anna",
    "stmellosa": "s_mellosa",
    "vingaker": "v_vingaker",
}


@dataclass
class ParseResult:
    intervals: list[tuple[float, float, str]]
    warnings: list[str]


@dataclass(frozen=True)
class Recording:
    path: Path
    place: str
    kind: str
    suffix: tuple[int, ...]
    source_priority: int

    @property
    def speaker_id(self) -> str:
        return f"{self.kind}_{'_'.join(str(part) for part in self.suffix)}"


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


def read_resource_places(path: Path) -> list[tuple[str, str, str]]:
    # Tcl-style SweDia resource file: {stem abbr full_name province x y group}
    text = path.read_bytes().decode("latin-1")
    rows = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        fields = shlex.split(line[1:-1])
        if len(fields) < 3:
            raise ValueError(f"Malformed row in {path} line {line_no}: {raw!r}")
        rows.append((fields[1], fields[2], fields[0]))
    return rows


def recording_parts(path: Path) -> tuple[str, str, tuple[int, ...]] | None:
    match = RECORDING_RE.match(path.stem)
    if not match:
        return None
    return match["place"], match["kind"], tuple(int(part) for part in match["suffix"].split("_"))


def existing_dirs(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def collect_wavs(paths: list[Path]) -> list[Recording]:
    recordings: dict[str, Recording] = {}
    for source_priority, directory in enumerate(existing_dirs(paths)):
        for path in sorted(directory.glob("*.wav")):
            parts = recording_parts(path)
            if not parts:
                continue
            place, kind, suffix = parts
            # Keep the first copy according to directory priority.
            recordings.setdefault(path.stem, Recording(path, place, kind, suffix, source_priority))
    return sorted(recordings.values(), key=lambda rec: (rec.place, rec.source_priority, rec.kind, rec.suffix, rec.path.name))


def collect_annotations(paths: list[Path], extension: str) -> dict[str, Path]:
    annotations: dict[str, Path] = {}
    for directory in existing_dirs(paths):
        for path in sorted(directory.glob(f"*.{extension}")):
            annotations.setdefault(path.stem, path)
    return annotations


def choose_media_stem(place: str, stems: set[str], hints: list[str] | None = None) -> tuple[str | None, str, float]:
    normalized = ascii_name(place)
    candidates = [ALIASES.get(ascii_name(hint), ascii_name(hint)) for hint in hints or []]
    candidates.append(normalized)
    if "_" in normalized:
        candidates.extend([normalized.split("_")[0], normalized.split("_")[-1]])
    alias = ALIASES.get(normalized)
    if alias:
        candidates.insert(0, alias)
    candidates = list(dict.fromkeys(candidate for candidate in candidates if candidate))
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


def build_matches(
    places: list[tuple[str, str, list[str]]],
    parsed_wavs: list[Recording],
    ord_annotations: dict[str, Path],
    seg_annotations: dict[str, Path],
    output: Path | None = None,
) -> tuple[list[dict], int]:
    stems = {recording.place for recording in parsed_wavs}
    matches = []
    max_speakers = 0
    for code, place, hints in places:
        stem, method, score = choose_media_stem(place, stems, hints)
        selected = sorted(
            (recording for recording in parsed_wavs if recording.place == stem),
            key=lambda recording: (recording.source_priority, recording.suffix, recording.kind, recording.path.name),
        ) if stem else []
        if selected:
            best_source_priority = min(recording.source_priority for recording in selected)
            selected = [recording for recording in selected if recording.source_priority == best_source_priority]
        max_speakers = max(max_speakers, len(selected))
        speakers = []
        for wav in selected:
            base = wav.path.stem
            ord_path = ord_annotations.get(base)
            seg_path = seg_annotations.get(base)
            warnings = []
            textgrid_name = ""
            if output is not None:
                duration = wav_duration(wav.path)
                tiers = []
                for tier_name, annotation in (("ord", ord_path), ("seg", seg_path)):
                    parsed = parse_xwaves(annotation) if annotation else ParseResult([], [])
                    intervals, extra = contiguous(parsed.intervals, duration)
                    tiers.append((tier_name, intervals))
                    warnings.extend(f"{tier_name}: {warning}" for warning in parsed.warnings + extra)
                textgrid_name = f"{base}.TextGrid"
                write_textgrid(output / textgrid_name, duration, tiers)
            speakers.append({
                "speaker": wav.speaker_id, "wav": wav.path.name,
                "ord": ord_path.name if ord_path else "MISSING",
                "seg": seg_path.name if seg_path else "MISSING",
                "textgrid": textgrid_name, "warnings": " | ".join(warnings),
            })
        matches.append({"code": code, "place": place, "stem": stem or "", "method": method,
                        "score": score, "speakers": speakers})
    return matches, max_speakers


def speaker_inventory_value(speaker: dict) -> str:
    available = ["w"]
    if speaker["ord"] != "MISSING":
        available.append("o")
    if speaker["seg"] != "MISSING":
        available.append("s")
    return ",".join(available)


def compact_speaker_values(match: dict) -> tuple[dict[int, str], bool]:
    grouped: dict[int, set[str]] = {}
    has_partial_recordings = False
    for speaker in match["speakers"]:
        parts = speaker["speaker"].split("_")[1:]
        if not parts:
            continue
        speaker_number = int(parts[0])
        has_partial_recordings = has_partial_recordings or len(parts) > 1
        available = grouped.setdefault(speaker_number, {"w"})
        if speaker["ord"] != "MISSING":
            available.add("o")
        if speaker["seg"] != "MISSING":
            available.add("s")

    order = {"w": 0, "o": 1, "s": 2}
    return {
        number: ",".join(sorted(values, key=lambda value: order[value]))
        for number, values in grouped.items()
    }, has_partial_recordings


def status_for_values(values: dict[int, str]) -> str:
    complete = sum(value == "w,o,s" for value in values.values())
    if complete >= 3:
        return "pass"
    if complete >= 1:
        return "partial pass"
    return "non-pass"


def status_for_match(match: dict) -> str:
    values, _ = compact_speaker_values(match)
    return status_for_values(values)


def write_compact_summary(
    path: Path,
    matches: list[dict],
    max_speakers: int,
    include_status: bool = False,
    provenance: dict[str, str] | None = None,
) -> None:
    compacted = [compact_speaker_values(match) for match in matches]
    max_speaker_number = max(
        (number for values, _ in compacted for number in values),
        default=max(max_speakers, 3),
    )
    compact_fields = ["abbr", "full_name"]
    if include_status:
        compact_fields.append("status")
    if provenance is not None:
        compact_fields.append("provenance")
    compact_fields += [f"sp{i}" for i in range(1, max(max_speaker_number, 3) + 1)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fields, delimiter="\t")
        writer.writeheader()
        for match, (values, has_partial_recordings) in zip(matches, compacted):
            place = f'{match["place"]}*' if has_partial_recordings else match["place"]
            row = {"abbr": match["code"], "full_name": place}
            if include_status:
                row["status"] = status_for_values(values)
            if provenance is not None:
                row["provenance"] = provenance.get(match["code"], "")
            for number, value in values.items():
                row[f"sp{number}"] = value
            writer.writerow(row)


def choose_unique_pass_matches(
    default_matches: list[dict],
    source_matches: dict[str, list[dict]],
) -> tuple[list[dict], dict[str, str], int]:
    source_by_code = {
        label: {match["code"]: match for match in matches}
        for label, matches in source_matches.items()
    }
    selected = []
    provenance = {}
    max_speakers = 0
    for default_match in default_matches:
        code = default_match["code"]
        pass_labels = [
            label for label, matches_by_code in source_by_code.items()
            if code in matches_by_code and status_for_match(matches_by_code[code]) == "pass"
        ]
        if len(pass_labels) == 1:
            label = pass_labels[0]
            match = source_by_code[label][code]
            provenance[code] = label
        else:
            match = default_match
        selected.append(match)
        values, _ = compact_speaker_values(match)
        max_speakers = max(max_speakers, *(values.keys() or [0]))
    return selected, provenance, max_speakers


def write_complete_textgrids(
    matches: list[dict],
    wavs: list[Recording],
    output: Path,
    provenance: dict[str, str] | None = None,
) -> int:
    wav_by_name = {recording.path.name: recording.path for recording in wavs}
    output.mkdir(parents=True, exist_ok=True)
    for old_textgrid in output.glob("*.TextGrid"):
        old_textgrid.unlink()
    generated = 0
    for match in matches:
        for speaker in match["speakers"]:
            if speaker["ord"] == "MISSING" or speaker["seg"] == "MISSING":
                continue
            preferred_media = None
            preferred_annotations = None
            if provenance:
                source = provenance.get(match["code"])
                if source == "sounds/sannotations":
                    preferred_media = Path("sounds")
                    preferred_annotations = Path("sannotations")
                elif source == "Media/Annotations":
                    preferred_media = Path("Media")
                    preferred_annotations = Path("Annotations")
            wav_path = (preferred_media / speaker["wav"]) if preferred_media else wav_by_name.get(speaker["wav"])
            if wav_path is not None and not wav_path.exists():
                wav_path = wav_by_name.get(speaker["wav"])
            if wav_path is None:
                continue
            annotation_dir = preferred_annotations or Path("sannotations" if wav_path.parent.name == "sounds" else "Annotations")
            ord_path = annotation_dir / speaker["ord"]
            seg_path = annotation_dir / speaker["seg"]
            if not ord_path.exists():
                ord_path = Path("sannotations") / speaker["ord"]
            if not ord_path.exists():
                ord_path = Path("Annotations") / speaker["ord"]
            if not seg_path.exists():
                seg_path = Path("sannotations") / speaker["seg"]
            if not seg_path.exists():
                seg_path = Path("Annotations") / speaker["seg"]
            if not ord_path.exists() or not seg_path.exists():
                continue

            duration = wav_duration(wav_path)
            tiers = []
            for tier_name, annotation in (("ord", ord_path), ("seg", seg_path)):
                parsed = parse_xwaves(annotation)
                intervals, _ = contiguous(parsed.intervals, duration)
                tiers.append((tier_name, intervals))
            write_textgrid(output / f"{wav_path.stem}.TextGrid", duration, tiers)
            generated += 1
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--places", type=Path, default=Path("Orter_SweDia.csv"))
    parser.add_argument(
        "--media",
        type=Path,
        action="append",
        default=None,
        help="Directory with .wav files. Can be repeated. Default: sounds, Media.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        action="append",
        default=None,
        help="Directory with .ord/.seg files. Can be repeated. Default: sannotations, Annotations.",
    )
    parser.add_argument("--output", type=Path, default=Path("TextGrids"))
    parser.add_argument("--summary", type=Path, default=Path("sweDia_inventory.csv"))
    parser.add_argument("--compact-summary", type=Path, default=Path("sweDia_inventory_compact.tsv"))
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--all-compact-summary", type=Path, default=Path("sweDia_inventory_all_compact.tsv"))
    parser.add_argument("--all-complete-output", type=Path, default=Path("TextGrids_all_wos"))
    args = parser.parse_args()

    media_dirs = args.media or [Path("sounds"), Path("Media")]
    annotation_dirs = args.annotations or [Path("sannotations"), Path("Annotations")]
    parsed_wavs = collect_wavs(media_dirs)
    ord_annotations = collect_annotations(annotation_dirs, "ord")
    seg_annotations = collect_annotations(annotation_dirs, "seg")
    args.output.mkdir(parents=True, exist_ok=True)
    places = [(code, place, []) for code, place in read_places(args.places)]
    matches, max_speakers = build_matches(places, parsed_wavs, ord_annotations, seg_annotations, args.output)

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

    write_compact_summary(args.compact_summary, matches, max_speakers)

    if args.resource.exists():
        resource_places = [
            (code, place, [resource_stem]) for code, place, resource_stem in read_resource_places(args.resource)
        ]
        all_matches, all_max_speakers = build_matches(
            resource_places, parsed_wavs, ord_annotations, seg_annotations
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
        all_matches, provenance, provenance_max_speakers = choose_unique_pass_matches(all_matches, pair_matches)
        all_max_speakers = max(all_max_speakers, provenance_max_speakers)
        write_compact_summary(
            args.all_compact_summary,
            all_matches,
            all_max_speakers,
            include_status=True,
            provenance=provenance,
        )
        all_complete_generated = write_complete_textgrids(
            all_matches, parsed_wavs, args.all_complete_output, provenance
        )
    else:
        all_complete_generated = 0

    generated = sum(len(match["speakers"]) for match in matches)
    missing_places = sum(not match["stem"] for match in matches)
    missing_ord = sum(s["ord"] == "MISSING" for m in matches for s in m["speakers"])
    missing_seg = sum(s["seg"] == "MISSING" for m in matches for s in m["speakers"])
    print(f"Generated {generated} TextGrids in {args.output}")
    print(f"Summary: {args.summary}")
    print(f"Compact summary: {args.compact_summary}")
    if args.resource.exists():
        print(f"All-villages compact summary: {args.all_compact_summary}")
        print(f"All w,o,s TextGrids: {args.all_complete_output} ({all_complete_generated})")
    print(f"Unmatched villages: {missing_places}; missing ord: {missing_ord}; missing seg: {missing_seg}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, wave.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
