#!/usr/bin/env python3
"""Combine tracker-comparison and ellipse-fit plots for one speaker/recording."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TRACKER_PLOTS = {
    ("vowel-space", "hz"): "vowel_space_comparison.png",
    ("vowel-space", "bark"): "vowel_space_comparison_bark.png",
    ("method-agreement", "hz"): "method_agreement.png",
    ("method-agreement", "bark"): "method_agreement_bark.png",
}


def read_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_title(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=(25, 25, 25), font=font)


def default_formant_dir(root: Path, recording_id: str) -> Path:
    return root / f"Formants_{recording_id}"


def default_ellipse_dir(root: Path, recording_id: str) -> Path:
    return root / f"Ellipses_{recording_id}"


def resolve_formant_dir(root: Path, recording_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        root / f"Formants_{recording_id}",
        root / "formants" / recording_id,
    ]
    for candidate in candidates:
        if (candidate / "token_comparison.csv").exists():
            return candidate
    return default_formant_dir(root, recording_id)


def resolve_ellipse_dir(root: Path, recording_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        root / f"Ellipses_{recording_id}",
        root / "ellipses" / recording_id,
    ]
    for candidate in candidates:
        if (candidate / f"{recording_id}_ellipse_fit.png").exists():
            return candidate
    return default_ellipse_dir(root, recording_id)


def run_command(command: list[str]) -> None:
    print("Running: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def formant_run_current(formant_dir: Path) -> bool:
    token_path = formant_dir / "token_comparison.csv"
    settings_path = formant_dir / "run_settings.json"
    if not token_path.exists() or not settings_path.exists():
        return False
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings.get("target_source") != "inventory":
            return False
        if settings.get("surface_target_mode") != "allowed":
            return False
        with token_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except (OSError, json.JSONDecodeError):
        return False
    required = {"surface_target_status", "surface_target_note"}
    return required.issubset(header)


def ellipse_run_current(formant_dir: Path, ellipse_dir: Path, recording_id: str) -> bool:
    ellipse_path = ellipse_dir / f"{recording_id}_ellipse_fit.png"
    medians_path = ellipse_dir / "recording_vowel_medians.csv"
    token_path = formant_dir / "token_comparison.csv"
    if not ellipse_path.exists() or not medians_path.exists():
        return False
    if token_path.exists() and token_path.stat().st_mtime > ellipse_path.stat().st_mtime:
        return False
    try:
        with medians_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except OSError:
        return False
    return "surface_target_notes" in header


def ensure_formant_run(args: argparse.Namespace, formant_dir: Path) -> None:
    tracker_path = formant_dir / TRACKER_PLOTS[(args.tracker, args.space)]
    token_path = formant_dir / "token_comparison.csv"
    if token_path.exists() and tracker_path.exists() and formant_run_current(formant_dir) and not args.force:
        return
    command = [
        sys.executable,
        "compare_formant_trackers.py",
        "--media", str(args.media),
        "--annotations", str(args.annotations),
        "--recordings", args.id,
        "--target-source", "inventory",
        "--surface-target-mode", "allowed",
        "--plots", args.token_plots,
        "--plot-space", args.space,
        "--output", str(formant_dir),
    ]
    if args.front_vowel_safe:
        command.append("--front-vowel-safe")
    else:
        command.append("--no-front-vowel-safe")
    run_command(command)


def ensure_ellipse_run(args: argparse.Namespace, formant_dir: Path, ellipse_dir: Path) -> None:
    ellipse_path = ellipse_dir / f"{args.id}_ellipse_fit.png"
    if ellipse_path.exists() and ellipse_run_current(formant_dir, ellipse_dir, args.id) and not args.force:
        return
    command = [
        sys.executable,
        "analyze_vowel_space_ellipses.py",
        "--tokens", str(formant_dir / "token_comparison.csv"),
        "--output", str(ellipse_dir),
        "--id", args.id,
        "--level", "recording",
        "--fit-basis", args.fit_basis,
        "--context-mode", args.context_mode,
        "--arrows", args.arrows,
        "--min-vowels", str(args.min_vowels),
    ]
    run_command(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="Recording id, e.g. bara_ym_1.")
    parser.add_argument("--analysis-root", type=Path, default=Path("Analyses"))
    parser.add_argument("--formant-dir", type=Path)
    parser.add_argument("--ellipse-dir", type=Path)
    parser.add_argument("--media", type=Path, default=Path("Media"))
    parser.add_argument("--annotations", type=Path, default=Path("Annotations"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracker", choices=["method-agreement", "vowel-space"], default="vowel-space")
    parser.add_argument("--space", choices=["hz", "bark"], default="bark")
    parser.add_argument("--fit-basis", choices=["vowels", "boundaries", "both"], default="both")
    parser.add_argument("--context-mode", choices=["all", "non_r", "r_only"], default="non_r")
    parser.add_argument("--arrows", choices=["none", "single", "overview", "both"], default="single")
    parser.add_argument("--min-vowels", type=int, default=4)
    parser.add_argument("--token-plots", choices=["all", "flagged", "none"], default="none")
    parser.add_argument("--front-vowel-safe", action=BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Regenerate formant and ellipse outputs even if files exist.")
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    formant_dir = resolve_formant_dir(args.analysis_root, args.id, args.formant_dir)
    ellipse_dir = resolve_ellipse_dir(args.analysis_root, args.id, args.ellipse_dir)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    ensure_formant_run(args, formant_dir)
    ensure_ellipse_run(args, formant_dir, ellipse_dir)

    tracker_path = formant_dir / TRACKER_PLOTS[(args.tracker, args.space)]
    ellipse_path = ellipse_dir / f"{args.id}_ellipse_fit.png"
    if not tracker_path.exists():
        raise FileNotFoundError(tracker_path)
    if not ellipse_path.exists():
        raise FileNotFoundError(ellipse_path)

    tracker = resize_to_height(read_image(tracker_path), args.height)
    ellipse = resize_to_height(read_image(ellipse_path), args.height)

    margin = 34
    title_height = 58
    gutter = 26
    width = tracker.width + ellipse.width + margin * 2 + gutter
    height = args.height + margin * 2 + title_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    left_x = margin
    right_x = margin + tracker.width + gutter
    image_y = margin + title_height
    draw_title(draw, "Tracker comparison", left_x, margin)
    draw_title(draw, "Ellipse fit", right_x, margin)
    canvas.paste(tracker, (left_x, image_y))
    canvas.paste(ellipse, (right_x, image_y))

    output = args.output or (args.analysis_root / f"{args.id}_tracker_ellipse_side_by_side_{args.space}.png")
    canvas.save(output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
