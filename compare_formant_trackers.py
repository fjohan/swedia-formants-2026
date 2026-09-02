#!/usr/bin/env python3
"""Compare fixed-ceiling Praat Burg and fasttrackpy on SweDia vowels.

Run this with the project's formanttest environment, for example:

    formanttest/bin/python compare_formant_trackers.py --recordings asby_ym_1

The script writes token measurements, frame-level tracks, and diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from argparse import BooleanOptionalAction
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swidia-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from fasttrackpy import CandidateTracks

from convert_xwaves_to_textgrids import parse_xwaves


VOWELS = {
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
FRONT_VOWEL_SAFE_TARGETS = {"y:", "e:", "ä:", "ö:"}
FILE_RE = re.compile(r"^(?P<village>.+)_(?P<speaker_type>[a-z]+)_(?P<speaker_number>\d+)(?:_\d+)*$")
WORD_VOWELS = set("aeiouyåäöAEIOUYÅÄÖ")


def bark(hz: float) -> float:
    """Traunmüller Bark transform, without optional endpoint corrections."""
    if not math.isfinite(hz) or hz <= 0:
        return math.nan
    return 26.81 / (1 + 1960 / hz) - 0.53


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else math.nan


def window_values(times: np.ndarray, values: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (times >= start) & (times <= end)
    return np.asarray(values)[mask]


def praat_track(sound: parselmouth.Sound, args: argparse.Namespace):
    formant = sound.to_formant_burg(
        time_step=args.time_step,
        max_number_of_formants=5.5,
        maximum_formant=args.praat_ceiling,
        window_length=args.window_length,
        pre_emphasis_from=args.pre_emphasis,
    )
    times = formant.xs()
    f1 = np.array([formant.get_value_at_time(1, t) for t in times])
    f2 = np.array([formant.get_value_at_time(2, t) for t in times])
    b1 = np.array([formant.get_bandwidth_at_time(1, t) for t in times])
    b2 = np.array([formant.get_bandwidth_at_time(2, t) for t in times])
    return times, f1, f2, b1, b2


def find_word(midpoint: float, words: list[tuple[float, float, str]]) -> str:
    for start, end, word in words:
        if start <= midpoint <= end:
            return word
    return ""


def read_lexical_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"lexical_item", "target_label", "ipa"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
    return {row["lexical_item"].strip().lower(): row for row in rows}


def looks_vowel_like(label: str) -> bool:
    return bool(label) and (label[0] in VOWEL_INITIALS or label.startswith("\\}"))


def normalized_word(word: str) -> str:
    return word.strip().casefold()


def has_post_vowel_r(word: str) -> bool:
    """Return true when r/R occurs after the first vowel letter in a lexical word."""
    folded = word.casefold()
    first_vowel = next((idx for idx, char in enumerate(folded) if char in WORD_VOWELS), None)
    return first_vowel is not None and "r" in folded[first_vowel + 1:]


def vowel_in_word(word_start: float, word_end: float, segments: list[tuple[float, float, str]]):
    candidates = [
        interval for interval in segments
        if word_start <= (interval[0] + interval[1]) / 2 <= word_end and looks_vowel_like(interval[2])
    ]
    return max(candidates, key=lambda interval: interval[1] - interval[0]) if candidates else None


def select_one_word_per_vowel(tokens: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for token in tokens:
        grouped[token["target_label"]][token["word_normalized"]].append(token)

    selected_tokens: list[dict] = []
    selection_rows: list[dict] = []
    for target_label, by_word in sorted(grouped.items()):
        def score(item: tuple[str, list[dict]]) -> tuple[int, float]:
            word_key, values = item
            durations = [float(value["duration_s"]) for value in values]
            return (len(values), finite_median(np.array(durations, dtype=float)))

        ranked = sorted(by_word.items(), key=lambda item: (-score(item)[0], -score(item)[1], item[0]))
        selected_word, selected_values = ranked[0]
        selected_tokens.extend(selected_values)
        selection_rows.append({
            "recording": selected_values[0]["recording"],
            "village": selected_values[0]["village"],
            "speaker": selected_values[0]["speaker"],
            "target_label": target_label,
            "ipa": selected_values[0]["ipa"],
            "selected_word": selected_values[0]["word"],
            "selected_word_normalized": selected_word,
            "selected_token_count": len(selected_values),
            "selected_median_duration_s": finite_median(np.array([float(value["duration_s"]) for value in selected_values], dtype=float)),
            "candidate_words": ";".join(
                f"{values[0]['word']}:{len(values)}" for _, values in sorted(by_word.items())
            ),
        })
    return sorted(selected_tokens, key=lambda token: (token["word_start_s"], token["word_end_s"], token["target_label"])), selection_rows


def save_plot(path: Path, sound: parselmouth.Sound, vowel_start: float, vowel_end: float,
              measure_start: float, measure_end: float, praat, fast, title: str) -> None:
    spectrogram = sound.to_spectrogram(window_length=0.005, maximum_frequency=3500)
    db = 10 * np.log10(np.maximum(spectrogram.values, np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.pcolormesh(spectrogram.x_grid(), spectrogram.y_grid(), db,
                  vmin=float(np.nanmax(db)) - 55, cmap="Greys", shading="auto")
    pt, pf1, pf2, _, _ = praat
    ft, ff1, ff2 = fast
    ax.plot(pt, pf1, color="#2166ac", lw=1.2, ls="--", label="Praat F1")
    ax.plot(pt, pf2, color="#b2182b", lw=1.2, ls="--", label="Praat F2")
    ax.plot(ft, ff1, color="#2166ac", lw=2.0, label="Fast Track F1")
    ax.plot(ft, ff2, color="#b2182b", lw=2.0, label="Fast Track F2")
    ax.axvline(vowel_start, color="black", lw=0.8)
    ax.axvline(vowel_end, color="black", lw=0.8)
    ax.axvspan(measure_start, measure_end, color="#ffd92f", alpha=0.25,
               label="measurement window")
    ax.set(xlim=(sound.xmin, sound.xmax), ylim=(0, 3500), xlabel="Time (s)",
           ylabel="Frequency (Hz)", title=title)
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_word_summary(rows: list[dict], label: str, max_words: int = 3) -> str:
    counts = Counter(row["word"] for row in rows if row.get("word"))
    if not counts:
        return ""
    items = counts.most_common()
    words = [word for word, _ in items[:max_words]]
    suffix = "+" if len(items) > max_words else ""
    return f" {'/'.join(words)}{suffix}"


def save_run_plots(output: Path, rows: list[dict], spaces: list[str], show_words: bool = True) -> None:
    usable = [row for row in rows if all(math.isfinite(float(row[key])) for key in
              ("praat_f1_hz", "praat_f2_hz", "fasttrack_f1_hz", "fasttrack_f2_hz"))]
    if not usable:
        return
    colors = dict(zip(VOWELS, plt.get_cmap("tab10").colors))

    for space in spaces:
        unit = "Bark" if space == "bark" else "Hz"
        suffix = "" if space == "hz" else f"_{space}"
        required = [f"{method}_{formant}_{space}" for method in ("praat", "fasttrack") for formant in ("f1", "f2")]
        space_rows = [
            row for row in usable
            if all(math.isfinite(float(row[key])) for key in required)
        ]
        if not space_rows:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
        for ax, formant in zip(axes, ("f1", "f2")):
            p = np.array([row[f"praat_{formant}_{space}"] for row in space_rows], dtype=float)
            f = np.array([row[f"fasttrack_{formant}_{space}"] for row in space_rows], dtype=float)
            lo, hi = min(p.min(), f.min()), max(p.max(), f.max())
            for label in VOWELS:
                idx = [i for i, row in enumerate(space_rows) if row["target_label"] == label]
                if idx:
                    label_rows = [space_rows[i] for i in idx]
                    legend_label = f"/{VOWELS[label]}/"
                    if show_words:
                        legend_label += plot_word_summary(label_rows, label)
                    ax.scatter(p[idx], f[idx], color=colors[label], label=legend_label, s=24)
            ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
            ax.set(xlabel=f"Praat {formant.upper()} ({unit})", ylabel=f"Fast Track {formant.upper()} ({unit})",
                   title=f"{formant.upper()} agreement", aspect="equal")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=8, fontsize=8)
        fig.savefig(output / f"method_agreement{suffix}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        for row in space_rows:
            color = colors[row["target_label"]]
            p = (float(row[f"praat_f2_{space}"]), float(row[f"praat_f1_{space}"]))
            f = (float(row[f"fasttrack_f2_{space}"]), float(row[f"fasttrack_f1_{space}"]))
            ax.plot([p[0], f[0]], [p[1], f[1]], color=color, alpha=0.3, lw=0.8)
            ax.scatter(*p, color=color, marker="o", facecolors="none", s=28)
            ax.scatter(*f, color=color, marker="x", s=28)
        for label in VOWELS:
            subset = [row for row in space_rows if row["target_label"] == label]
            if not subset:
                continue
            for method, marker in (("praat", "o"), ("fasttrack", "x")):
                f1 = np.median([float(row[f"{method}_f1_{space}"]) for row in subset])
                f2 = np.median([float(row[f"{method}_f2_{space}"]) for row in subset])
                ax.scatter(f2, f1, color=colors[label], marker=marker, s=85,
                           facecolors="none" if marker == "o" else colors[label])
            text = f"/{VOWELS[label]}/"
            if show_words:
                text += plot_word_summary(subset, label)
            ax.annotate(text, (f2, f1), xytext=(4, 4), textcoords="offset points")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set(xlabel=f"F2 ({unit})", ylabel=f"F1 ({unit})",
               title=f"Tracker comparison: Praat ○, Fast Track × ({unit})")
        fig.savefig(output / f"vowel_space_comparison{suffix}.png", dpi=150)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=Path("Media"))
    parser.add_argument("--annotations", type=Path, default=Path("Annotations"))
    parser.add_argument("--lexical-map", type=Path, default=Path("lexical_vowel_targets.tsv"))
    parser.add_argument("--output", type=Path, default=Path("FormantComparison"))
    parser.add_argument("--recordings", default="*", help="Comma-separated stem globs")
    parser.add_argument("--max-recordings", type=int, default=0, help="0 means all")
    parser.add_argument("--max-tokens-per-vowel", type=int, default=0, help="Per recording; 0 means all")
    parser.add_argument("--limit-total", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--one-word-per-vowel",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "For each recording and target vowel, keep only the best lexical word "
            "(most valid vowel tokens; ties use longer median vowel duration)."
        ),
    )
    parser.add_argument(
        "--avoid-post-vowel-r",
        action=BooleanOptionalAction,
        default=True,
        help="Exclude lexical words where r/R occurs after the first vowel, e.g. lär, dör, rör.",
    )
    parser.add_argument("--plots", choices=["all", "flagged", "none"], default="flagged")
    parser.add_argument("--plot-space", choices=["hz", "bark", "both"], default="both")
    parser.add_argument(
        "--plot-word-labels",
        action=BooleanOptionalAction,
        default=True,
        help="Include selected lexical word information in aggregate F1/F2 vowel-space labels.",
    )
    parser.add_argument("--window-start", type=float, default=0.45)
    parser.add_argument("--window-end", type=float, default=0.55)
    parser.add_argument("--padding", type=float, default=0.03)
    parser.add_argument("--time-step", type=float, default=0.002)
    parser.add_argument("--window-length", type=float, default=0.025)
    parser.add_argument("--pre-emphasis", type=float, default=50.0)
    parser.add_argument(
        "--front-vowel-safe",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Use safer defaults for front vowels: Praat ceiling 5500 Hz, "
            "FastTrack candidate ceilings 5000-7000 Hz, and extra low-F2/min-ceiling flags."
        ),
    )
    parser.add_argument("--praat-ceiling", type=float)
    parser.add_argument("--fast-min-ceiling", type=float)
    parser.add_argument("--fast-max-ceiling", type=float)
    parser.add_argument("--fast-steps", type=int)
    parser.add_argument(
        "--front-f2-min-hz",
        type=float,
        default=1500.0,
        help="Low-F2 warning threshold used with --front-vowel-safe for /y:/, /e:/, /ä:/, /ö:/.",
    )
    parser.add_argument("--f1-warning", type=float, default=100.0)
    parser.add_argument("--f2-warning", type=float, default=200.0)
    args = parser.parse_args()
    if args.front_vowel_safe:
        if args.praat_ceiling is None:
            args.praat_ceiling = 5500.0
        if args.fast_min_ceiling is None:
            args.fast_min_ceiling = 5000.0
        if args.fast_max_ceiling is None:
            args.fast_max_ceiling = 7000.0
        if args.fast_steps is None:
            args.fast_steps = 11
    else:
        if args.praat_ceiling is None:
            args.praat_ceiling = 5000.0
        if args.fast_min_ceiling is None:
            args.fast_min_ceiling = 4000.0
        if args.fast_max_ceiling is None:
            args.fast_max_ceiling = 6500.0
        if args.fast_steps is None:
            args.fast_steps = 14
    lexical_map = read_lexical_map(args.lexical_map)

    if not 0 <= args.window_start < args.window_end <= 1:
        parser.error("measurement window fractions must satisfy 0 <= start < end <= 1")

    globs = [item.strip() for item in args.recordings.split(",") if item.strip()]
    wavs = sorted({path for glob in globs for path in args.media.glob(f"{glob}.wav")})
    if args.max_recordings:
        wavs = wavs[:args.max_recordings]
    if not wavs:
        parser.error("no recordings matched --recordings")

    args.output.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output / "diagnostics"
    plot_dir.mkdir(exist_ok=True)
    token_rows: list[dict] = []
    track_rows: list[dict] = []
    errors: list[dict] = []
    word_selection_rows: list[dict] = []
    total = 0

    for wav_no, wav_path in enumerate(wavs, 1):
        seg_path = args.annotations / f"{wav_path.stem}.seg"
        ord_path = args.annotations / f"{wav_path.stem}.ord"
        if not seg_path.exists() or not ord_path.exists():
            print(f"[{wav_no}/{len(wavs)}] skip {wav_path.stem}: missing ord or seg", file=sys.stderr)
            continue
        parts = FILE_RE.match(wav_path.stem)
        sound = parselmouth.Sound(str(wav_path))
        all_segments = parse_xwaves(seg_path).intervals
        words = parse_xwaves(ord_path).intervals
        lexical_tokens = []
        for word_start, word_end, word in words:
            mapping = lexical_map.get(word.strip().lower())
            if not mapping:
                continue
            segment = vowel_in_word(word_start, word_end, all_segments)
            if segment is None:
                errors.append({
                    "token_id": f"{wav_path.stem}_unmeasured", "recording": wav_path.stem,
                    "target_label": mapping["target_label"], "ipa": mapping["ipa"], "word": word,
                    "word_start_s": word_start, "word_end_s": word_end,
                    "error_type": "MissingVowelSegment", "error": "no vowel-like seg interval inside word",
                })
                continue
            start, end, seg_label = segment
            target_label, target_ipa = mapping["target_label"], mapping["ipa"]
            duration = end - start
            parts_dict = {
                "recording": wav_path.stem,
                "village": parts["village"] if parts else "",
                "speaker": f'{parts["speaker_type"]}_{parts["speaker_number"]}' if parts else "",
            }
            lexical_tokens.append({
                **parts_dict,
                "word_start_s": word_start, "word_end_s": word_end, "word": word,
                "word_normalized": normalized_word(word),
                "post_vowel_r": int(has_post_vowel_r(word)),
                "mapping": mapping,
                "target_label": target_label, "ipa": target_ipa,
                "surface_seg_label": seg_label,
                "start_s": start, "end_s": end, "duration_s": duration,
            })
        if args.avoid_post_vowel_r:
            lexical_tokens = [token for token in lexical_tokens if not token["post_vowel_r"]]
        if args.one_word_per_vowel:
            lexical_tokens, selected = select_one_word_per_vowel(lexical_tokens)
            word_selection_rows.extend(selected)
        if args.max_tokens_per_vowel:
            kept, count = [], Counter()
            for token in lexical_tokens:
                target = token["target_label"]
                if count[target] < args.max_tokens_per_vowel:
                    kept.append(token)
                    count[target] += 1
            lexical_tokens = kept
        print(f"[{wav_no}/{len(wavs)}] {wav_path.stem}: {len(lexical_tokens)} lexical target tokens")

        for token_no, token in enumerate(lexical_tokens, 1):
            if args.limit_total and total >= args.limit_total:
                break
            total += 1
            word = token["word"]
            word_start, word_end = token["word_start_s"], token["word_end_s"]
            start, end = token["start_s"], token["end_s"]
            seg_label = token["surface_seg_label"]
            target_label, target_ipa = token["target_label"], token["ipa"]
            token_id = f"{wav_path.stem}_{token_no:04d}_{target_label.replace(':', 'L')}"
            duration = token["duration_s"]
            measure_start = start + args.window_start * duration
            measure_end = start + args.window_end * duration
            clip_start = max(sound.xmin, start - args.padding)
            clip_end = min(sound.xmax, end + args.padding)
            clip = sound.extract_part(clip_start, clip_end, preserve_times=True)
            base = {
                "token_id": token_id, "recording": wav_path.stem,
                "village": token["village"],
                "speaker": token["speaker"],
                "target_label": target_label, "ipa": target_ipa,
                "surface_seg_label": seg_label, "word": word,
                "word_normalized": token["word_normalized"],
                "post_vowel_r": token["post_vowel_r"],
                "word_start_s": word_start, "word_end_s": word_end,
                "start_s": start, "end_s": end, "duration_s": duration,
                "measure_start_s": measure_start, "measure_end_s": measure_end,
            }
            try:
                praat = praat_track(clip, args)
                candidates = CandidateTracks(
                    sound=clip, min_max_formant=args.fast_min_ceiling,
                    max_max_formant=args.fast_max_ceiling, nstep=args.fast_steps,
                    n_formants=4, window_length=args.window_length,
                    time_step=args.time_step, pre_emphasis_from=args.pre_emphasis,
                )
                winner = candidates.winner
                fast = (winner.time_domain, winner.smoothed_formants[0], winner.smoothed_formants[1])
                p_f1 = finite_median(window_values(praat[0], praat[1], measure_start, measure_end))
                p_f2 = finite_median(window_values(praat[0], praat[2], measure_start, measure_end))
                f_f1 = finite_median(window_values(fast[0], fast[1], measure_start, measure_end))
                f_f2 = finite_median(window_values(fast[0], fast[2], measure_start, measure_end))
                d_f1, d_f2 = f_f1 - p_f1, f_f2 - p_f2
                flags = []
                if not all(map(math.isfinite, (p_f1, p_f2, f_f1, f_f2))):
                    flags.append("missing_measurement")
                if math.isfinite(d_f1) and abs(d_f1) > args.f1_warning:
                    flags.append("F1_disagreement")
                if math.isfinite(d_f2) and abs(d_f2) > args.f2_warning:
                    flags.append("F2_disagreement")
                if duration < 0.05:
                    flags.append("short_vowel")
                if args.front_vowel_safe and target_label in FRONT_VOWEL_SAFE_TARGETS:
                    if math.isfinite(p_f2) and p_f2 < args.front_f2_min_hz:
                        flags.append("front_vowel_low_F2_praat")
                    if math.isfinite(f_f2) and f_f2 < args.front_f2_min_hz:
                        flags.append("front_vowel_low_F2_fasttrack")
                    if math.isclose(float(winner.maximum_formant), float(args.fast_min_ceiling), rel_tol=0.0, abs_tol=1e-6):
                        flags.append("front_vowel_min_ceiling_candidate")
                row = base | {
                    "praat_ceiling_hz": args.praat_ceiling,
                    "fasttrack_ceiling_hz": winner.maximum_formant,
                    "fasttrack_candidate": candidates.winner_idx + 1,
                    "praat_f1_hz": p_f1, "praat_f2_hz": p_f2,
                    "fasttrack_f1_hz": f_f1, "fasttrack_f2_hz": f_f2,
                    "praat_f1_bark": bark(p_f1), "praat_f2_bark": bark(p_f2),
                    "fasttrack_f1_bark": bark(f_f1), "fasttrack_f2_bark": bark(f_f2),
                    "f1_difference_hz": d_f1, "f2_difference_hz": d_f2,
                    "quality_flags": ";".join(flags),
                }
                token_rows.append(row)
                for method, times, f1s, f2s in (
                    ("praat", praat[0], praat[1], praat[2]),
                    ("fasttrack", fast[0], fast[1], fast[2]),
                ):
                    for time, f1, f2 in zip(times, f1s, f2s):
                        track_rows.append(base | {"method": method, "time_s": time,
                                                   "relative_time": (time-start)/duration,
                                                   "f1_hz": f1, "f2_hz": f2})
                if args.plots == "all" or (args.plots == "flagged" and flags):
                    save_plot(plot_dir / f"{token_id}.png", clip, start, end,
                              measure_start, measure_end, praat, fast,
                              f"{wav_path.stem}  {word}  /{target_ipa}/ [{seg_label}]  "
                              f"Praat {p_f1:.0f}/{p_f2:.0f}, Fast Track {f_f1:.0f}/{f_f2:.0f} Hz")
            except Exception as error:  # Keep a corpus run going and make failures auditable.
                errors.append(base | {"error_type": type(error).__name__, "error": str(error)})
                print(f"  failed {token_id}: {type(error).__name__}: {error}", file=sys.stderr)
        if args.limit_total and total >= args.limit_total:
            break

    token_fields = [
        "token_id", "recording", "village", "speaker", "target_label", "ipa",
        "surface_seg_label", "word", "word_normalized", "post_vowel_r", "word_start_s", "word_end_s",
        "start_s", "end_s", "duration_s", "measure_start_s", "measure_end_s",
        "praat_ceiling_hz", "fasttrack_ceiling_hz", "fasttrack_candidate",
        "praat_f1_hz", "praat_f2_hz", "fasttrack_f1_hz", "fasttrack_f2_hz",
        "praat_f1_bark", "praat_f2_bark", "fasttrack_f1_bark", "fasttrack_f2_bark",
        "f1_difference_hz", "f2_difference_hz", "quality_flags",
    ]
    write_rows(args.output / "token_comparison.csv", token_fields, token_rows)
    mapping_counts = Counter(
        (row["target_label"], row["ipa"], row["word"], row["surface_seg_label"])
        for row in token_rows
    )
    mapping_rows = [
        {"target_label": key[0], "ipa": key[1], "lexical_item": key[2],
         "surface_seg_label": key[3], "token_count": count}
        for key, count in sorted(mapping_counts.items())
    ]
    write_rows(args.output / "surface_label_mapping.csv",
               ["target_label", "ipa", "lexical_item", "surface_seg_label", "token_count"],
               mapping_rows)
    write_rows(
        args.output / "word_selection.csv",
        [
            "recording", "village", "speaker", "target_label", "ipa",
            "selected_word", "selected_word_normalized",
            "selected_token_count", "selected_median_duration_s", "candidate_words",
        ],
        word_selection_rows,
    )
    track_fields = ["token_id", "recording", "village", "speaker", "target_label", "ipa",
                    "surface_seg_label", "word",
                    "start_s", "end_s", "method", "time_s", "relative_time", "f1_hz", "f2_hz"]
    write_rows(args.output / "formant_tracks.csv", track_fields, track_rows)
    write_rows(args.output / "errors.csv",
               ["token_id", "recording", "target_label", "ipa", "surface_seg_label", "word",
                "word_start_s", "word_end_s", "start_s", "end_s", "error_type", "error"],
               errors)
    plot_spaces = ["hz", "bark"] if args.plot_space == "both" else [args.plot_space]
    save_run_plots(args.output, token_rows, plot_spaces, show_words=args.plot_word_labels)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings |= {"python": sys.version, "parselmouth": parselmouth.__version__}
    (args.output / "run_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flagged = sum(bool(row["quality_flags"]) for row in token_rows)
    print(f"Measured {len(token_rows)} tokens; {flagged} flagged; {len(errors)} errors")
    print(f"Results: {args.output}")
    return 1 if errors and not token_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
