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

## Relating ellipse angles to geography

`Scripts/analyze_north_south_vs_angle.py` relates vowel-space and midpoint-ellipse angles to geographic north–south position. It produces angle maps, correlation plots, Pearson and Spearman statistics, and village-permutation tests, with clustered treatment of speaker-level observations.

## Modelling vowel trajectories across geography

`Scripts/model_vowel_axis_trajectories.py` models each vowel's curved path through two-dimensional acoustic space as position along a geographic axis changes. By default, position is projected onto the Löderup–Arjeplog axis, Tjällmo and Rimforsa define a central geographic band, and Gotland and Finland are excluded. The axis, partitioning, and exclusions are configurable, and the original raw-coordinate behavior remains available. Separate coordinated spline models are fitted to the two acoustic dimensions using village-level vowel positions, while individual speakers remain visible in the plots. It supports fixed Praat, FastTrack, and PCA measurements and exports model statistics, fitted positions, and bootstrap uncertainty.

### Modelling paired PCA trajectories

`Scripts/plot_pair_axis_trajectories.py` overlays geographic-axis trajectories for vowel pairs and draws the connecting vector for every speaker. Each five-panel figure shows the two individual vowel trajectories with model statistics and bootstrap uncertainty, their paired geometry, pair angle over the axis, and pair distance over the axis. By default, it runs all six canonical directed pairs and writes their plots into one directory; `--first` and `--second` select a single alternative pair. It supports PCA, fixed Praat, and FastTrack measurements. Its default geography uses the Löderup–Arjeplog axis, a Tjällmo–Rimforsa central band, and excludes Gotland and Finland; the original raw north–south coordinate remains available.

## Analysing directed vowel pairs

`Scripts/analyze_vowel_pair_geography.py` measures changes along the axis of each directed vowel pair. It separates movement of the two endpoints, movement of the pair midpoint, and expansion or compression of the pair, and models their potentially nonlinear relationship with north–south position.

## Summarizing regional pair midpoints

`Scripts/plot_regional_pair_midpoint_summary.py` provides a compact regional view of vowel-pair midpoints. Speakers can be divided into approximately equal north, central, and south groups or into Norrland, Svealand, and Götaland. Arrows identify pairs with a monotonic ordering across the three groups.
