#!/usr/bin/env python3
"""Run single-speaker tracker/ellipse plots for a SweDia corpus scope.

This is a batch wrapper around combine_single_speaker_plots.py.  It selects
complete wav+ord+seg recordings from either Orter_SweDia.csv or resource.txt,
runs the single-speaker plotting workflow, and collects the combined PNGs plus
recording vowel medians into one directory.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

from compare_formant_trackers import corpus_jobs, source_slug


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collection_default(scope: str, source_mode: str, space: str) -> Path:
    return Path("Analyses") / f"CombinedSpeakerPlots_{scope}_{source_mode}_{space}"


def copied_name(job_stem: str, source_pair: str, suffix: str) -> str:
    return f"{source_slug(source_pair)}_{job_stem}_{suffix}"


def run_one(args: argparse.Namespace, job, output_png: Path, formant_dir: Path, ellipse_dir: Path) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "combine_single_speaker_plots.py",
        "--id", job.wav_path.stem,
        "--analysis-root", str(args.analysis_root),
        "--formant-dir", str(formant_dir),
        "--ellipse-dir", str(ellipse_dir),
        "--media", str(job.wav_path.parent),
        "--annotations", str(job.ord_path.parent),
        "--output", str(output_png),
        "--tracker", args.tracker,
        "--space", args.space,
        "--fit-basis", args.fit_basis,
        "--context-mode", args.context_mode,
        "--arrows", args.arrows,
        "--min-vowels", str(args.min_vowels),
        "--token-plots", args.token_plots,
        "--height", str(args.height),
    ]
    if args.front_vowel_safe:
        command.append("--front-vowel-safe")
    else:
        command.append("--no-front-vowel-safe")
    if args.force:
        command.append("--force")
    if args.dry_run:
        print(" ".join(command))
        return subprocess.CompletedProcess(command, 0, "", "")
    print("Running: " + " ".join(command), flush=True)
    return subprocess.run(command, text=True, capture_output=True)


def copy_medians(ellipse_dir: Path, collection_dir: Path, job_stem: str, source_pair: str) -> Path | None:
    source = ellipse_dir / "recording_vowel_medians.csv"
    if not source.exists():
        return None
    target = collection_dir / copied_name(job_stem, source_pair, "recording_vowel_medians.csv")
    shutil.copy2(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-scope", choices=["original", "all"], default="original")
    parser.add_argument(
        "--source-mode",
        choices=["media_annotations", "sounds_sannotations", "both"],
        default="both",
    )
    parser.add_argument("--places", type=Path, default=Path("Orter_SweDia.csv"))
    parser.add_argument("--resource", type=Path, default=Path("resource.txt"))
    parser.add_argument("--media", type=Path, default=Path("Media"), help="Only used internally for custom-compatible selection.")
    parser.add_argument("--annotations", type=Path, default=Path("Annotations"), help="Only used internally for custom-compatible selection.")
    parser.add_argument("--analysis-root", type=Path, default=Path("Analyses"))
    parser.add_argument("--collection-dir", type=Path)
    parser.add_argument("--max-recordings", type=int, default=0, help="0 means all; useful for testing.")
    parser.add_argument("--tracker", choices=["method-agreement", "vowel-space"], default="vowel-space")
    parser.add_argument("--space", choices=["hz", "bark"], default="bark")
    parser.add_argument("--fit-basis", choices=["vowels", "boundaries", "both"], default="both")
    parser.add_argument("--context-mode", choices=["all", "non_r", "r_only"], default="non_r")
    parser.add_argument("--arrows", choices=["none", "single", "overview", "both"], default="single")
    parser.add_argument("--min-vowels", type=int, default=4)
    parser.add_argument("--token-plots", choices=["all", "flagged", "none"], default="none")
    parser.add_argument("--front-vowel-safe", action=BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed speaker.")
    args = parser.parse_args()

    collection_dir = args.collection_dir or collection_default(args.corpus_scope, args.source_mode, args.space)
    collection_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_root.mkdir(parents=True, exist_ok=True)

    jobs = corpus_jobs(
        args.corpus_scope,
        args.source_mode,
        args.places,
        args.resource,
        args.media,
        args.annotations,
    )
    if args.max_recordings:
        jobs = jobs[:args.max_recordings]
    if not jobs:
        parser.error("no complete wav+ord+seg recordings matched the requested scope")

    rows = []
    for index, job in enumerate(jobs, 1):
        source = source_slug(job.source_pair)
        stem = job.wav_path.stem
        output_png = collection_dir / copied_name(stem, job.source_pair, f"tracker_ellipse_side_by_side_{args.space}.png")
        formant_dir = args.analysis_root / "BatchFormants" / source / stem
        ellipse_dir = args.analysis_root / "BatchEllipses" / source / stem
        print(f"[{index}/{len(jobs)}] {job.source_pair} {stem}", flush=True)
        result = run_one(args, job, output_png, formant_dir, ellipse_dir)
        median_copy = None
        status = "dry_run" if args.dry_run else ("ok" if result.returncode == 0 else "failed")
        if result.returncode == 0 and not args.dry_run:
            median_copy = copy_medians(ellipse_dir, collection_dir, stem, job.source_pair)
            if median_copy is None:
                status = "missing_medians"
        elif result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            if args.stop_on_error:
                raise SystemExit(result.returncode)
        rows.append({
            "recording": stem,
            "source_pair": job.source_pair,
            "status": status,
            "combined_plot": output_png.name if output_png.exists() else "",
            "recording_vowel_medians": median_copy.name if median_copy else "",
            "formant_dir": str(formant_dir),
            "ellipse_dir": str(ellipse_dir),
            "returncode": result.returncode,
        })

    write_rows(
        collection_dir / "batch_summary.tsv",
        rows,
        [
            "recording", "source_pair", "status", "combined_plot", "recording_vowel_medians",
            "formant_dir", "ellipse_dir", "returncode",
        ],
    )
    ok = sum(row["status"] == "ok" for row in rows)
    if args.dry_run:
        print(f"Planned {len(rows)} speaker plot commands; summary: {collection_dir / 'batch_summary.tsv'}")
        return 0
    print(f"Wrote {ok}/{len(rows)} combined plots/median copies to {collection_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
