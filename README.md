# SweDia vowel acoustics

This repository contains scripts and supporting resources for acoustic analyses of vowel variation in the SweDia dialect corpus. The analyses compare conventional formant measurements with spectral PCA and examine both vowel position and spectral movement over time.

Audio recordings are stored in `Media/` and `sounds/`, annotations in `TextGrids_all_wos/`, analysis scripts in `Scripts/`, and generated results in `Analyses/`.

## Extracting vowel measurements

`Scripts/extract_praat_fixed_target_vowels.py` extracts token-level measurements for the eight target vowels. It measures F1 and F2 at robust 20%, 50%, and 80% temporal points using fixed-ceiling Praat analysis and calculates vowel vector length (VL). Optional modes add corresponding FastTrack and Bark-spectrum PCA measurements. The resulting table also contains lexical, speaker, and geographic metadata.

### Adding Bark measurements

`Scripts/add_bark_to_formants.py` provides a fast way to add Bark-transformed F1/F2 values to an existing extraction table. It recalculates Bark-space VL from the transformed 20% and 80% coordinates for both fixed Praat and FastTrack, while preserving the original Hz and PCA measurements.

## Comparing vowel vector lengths

`Scripts/analyze_vl_method_correlations.py` compares VL estimates from fixed Praat, FastTrack, and spectral PCA. It calculates Pearson and Spearman correlations overall and separately by vowel, and produces pairwise correlation plots and a CSV summary. By default, it reads `formants-all.csv`.

## Plotting midpoint vowel spaces

`Scripts/plot_midpoint_vowel_spaces.py` constructs midpoint vowel spaces from fixed Praat, FastTrack, or PCA measurements. It can aggregate by village or speaker, optionally include incomplete speakers, and plot vowel positions, directed vowel-pair arrows, pair midpoints, and fitted ellipses. It also exports ellipse angles and geographic coordinates for subsequent spatial analyses.
