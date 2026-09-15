#!/usr/bin/env python3
"""Add Bark F1/F2 and Bark-space vector lengths to an existing formant CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


TIMEPOINTS = (20, 50, 80)
METHODS = (
    ("", "bark_"),
    ("ft_", "ft_bark_"),
)


def hz_to_bark(hz: float) -> float:
    """Traunmüller transform used by the other project scripts."""
    return 26.81 / (1.0 + 1960.0 / hz) - 0.53


def bark_fields(output_prefix: str) -> list[str]:
    return [
        *(f"{output_prefix}f{formant}_{time}" for time in TIMEPOINTS for formant in (1, 2)),
        f"{output_prefix}VL",
    ]


def output_columns(input_columns: list[str], available_methods: list[tuple[str, str]]) -> list[str]:
    generated = {field for _, output_prefix in available_methods for field in bark_fields(output_prefix)}
    columns = [field for field in input_columns if field not in generated]
    for input_prefix, output_prefix in available_methods:
        anchor = f"{input_prefix}VL"
        insertion = columns.index(anchor) + 1
        columns[insertion:insertion] = bark_fields(output_prefix)
    return columns


def convert_row(row: dict, available_methods: list[tuple[str, str]]) -> dict:
    for input_prefix, output_prefix in available_methods:
        values = {}
        try:
            for time in TIMEPOINTS:
                for formant in (1, 2):
                    values[formant, time] = hz_to_bark(float(row[f"{input_prefix}f{formant}_{time}"]))
        except (TypeError, ValueError, ZeroDivisionError):
            for field in bark_fields(output_prefix):
                row[field] = ""
            continue
        for (formant, time), value in values.items():
            row[f"{output_prefix}f{formant}_{time}"] = format(value, ".12g")
        row[f"{output_prefix}VL"] = format(
            math.hypot(
                values[1, 80] - values[1, 20],
                values[2, 80] - values[2, 20],
            ),
            ".12g",
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("formants-all.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("formants-all-bark.csv"),
        help="May equal --input for an atomic in-place replacement.",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        input_columns = list(reader.fieldnames or ())
    if not rows:
        raise SystemExit(f"No data rows in {args.input}")

    available_methods = []
    for input_prefix, output_prefix in METHODS:
        required = {
            f"{input_prefix}f{formant}_{time}"
            for time in TIMEPOINTS for formant in (1, 2)
        }
        if required.issubset(input_columns):
            available_methods.append((input_prefix, output_prefix))
    if not available_methods:
        raise SystemExit("No complete Praat or FastTrack 20/50/80 Hz column set found")

    columns = output_columns(input_columns, available_methods)
    converted = [convert_row(row, available_methods) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(converted)
    temporary.replace(args.output)
    names = ", ".join("FastTrack" if prefix else "Praat" for prefix, _ in available_methods)
    print(f"Converted {len(rows)} rows ({names}); wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
