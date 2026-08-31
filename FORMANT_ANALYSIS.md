# SweDia formant-tracker comparison

`compare_formant_trackers.py` compares two analyses with matched LPC settings:

- a fixed-ceiling Praat Burg analysis through Parselmouth;
- Fast Track's automatically selected analysis over several formant ceilings.

Run it with the `formanttest` environment. A small pilot that creates a plot for
every measured token is:

```bash
formanttest/bin/python compare_formant_trackers.py \
  --recordings asby_ym_1 \
  --max-tokens-per-vowel 1 \
  --plots all \
  --output FormantPilot
```

For all matching recordings and all lexical target words:

```bash
formanttest/bin/python compare_formant_trackers.py \
  --recordings '*' \
  --plots flagged \
  --output FormantComparison
```

Use `--help` for analysis settings and sampling limits. By default, a token is
flagged when tracker medians differ by more than 100 Hz for F1 or 200 Hz for F2.

## Outputs

- `token_comparison.csv`: 45–55% median measurements in Hz and Bark, ceiling,
  word, speaker, and quality flags;
- `surface_label_mapping.csv`: observed target-to-surface transcription counts;
- `formant_tracks.csv`: frame-level F1/F2 trajectories for both methods;
- `diagnostics/*.png`: spectrograms with overlaid tracks and measurement window;
- `method_agreement.png`: fixed Praat versus Fast Track measurements;
- `vowel_space_comparison.png`: paired measurements in an F1/F2 vowel space;
- `errors.csv`: tokens that could not be analysed;
- `run_settings.json`: the complete reproducibility record.

## Category rule

Category membership comes from `lexical_vowel_targets.tsv`, not from the
surface `seg` transcription. The mapping includes the eight core targets and
recurring alternative stimulus words, such as *lås* and *sova* for /oː/ and
*blöt, söt, lös,* and *dör* for /øː/. Within each word interval, the longest
vowel-like `seg` interval is measured. Both `target_label` and the unchanged
`surface_seg_label` are written to the results.

This preserves cross-dialect comparability without discarding transcriptional
variation. For example, an /eː/ target in *leta* remains /eː/ when its surface
label is `ä:`. Edit or extend `lexical_vowel_targets.tsv` to change the lexical
mapping; `--lexical-map` can select a different mapping file.
