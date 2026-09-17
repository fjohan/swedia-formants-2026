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

`Scripts/plot_regional_pair_midpoint_summary.py` provides a compact regional view of vowel-pair midpoints. Its default geography uses the Löderup–Arjeplog axis, the Tjällmo–Rimforsa central band, and excludes Gotland and Finland. Midpoints are calculated within speakers, aggregated within villages, and then summarized across villages; the plot distinguishes their true two-dimensional positions from projections onto the corpus pair axes and shows village-bootstrap uncertainty. Equal-sized groups, traditional lands, and the earlier centroid-based midpoint calculation remain available as legacy options. Arrows identify pairs with a monotonic ordering across the three groups.

---

## Spatial GAM analyses of vowels and vowel-pair midpoints

The following scripts provide complementary directions of spatial modelling. Both use village-balanced acoustic positions, two-dimensional tensor-product smooths, village-level permutation tests, and 10-fold cross-validation. Gotland and the Finnish regions are excluded by default. Geographic plots follow the source-map orientation, with larger `geo_y` values displayed farther down.

### Predicting acoustic position from geography

`Scripts/model_spatial_vowel_midpoint_gams.py` models each acoustic coordinate from the joint geographic plane:

```text
acoustic coordinate 1 ~ s(geo_x, geo_y)
acoustic coordinate 2 ~ s(geo_x, geo_y)
```

It can analyze all eight individual vowels and all six canonical paired-speaker vowel midpoints using spectral PCA, fixed-Praat formants, or FastTrack formants. For individual vowels, speaker measurements are aggregated to village medians. For midpoints, the two vowels are first paired within speakers and then aggregated within villages. Each target receives two geographic effect maps, an observed-versus-fitted acoustic-space plot, coordinate-wise and joint permutation tests, cross-validated R² values, village predictions, and a downloadable continuous surface.

The default command runs every method, vowel, and midpoint:

```bash
swedia-pca/bin/python Scripts/model_spatial_vowel_midpoint_gams.py
```

The scope can be reduced when developing or checking a particular comparison:

```bash
swedia-pca/bin/python Scripts/model_spatial_vowel_midpoint_gams.py \
  --methods pca fasttrack \
  --targets midpoints \
  --pairs 'uː,oː'
```

Default results are written to `Analyses/Spatial_vowel_midpoint_GAMs/`. The top-level `index.html` and `all_model_statistics.csv` compare the same spatial analysis across acoustic representations and targets.

### Projecting geography over acoustic space

`Scripts/model_reverse_acoustic_geography_gams.py` reverses the conditional direction. Acoustic coordinates form the two-dimensional predictor map, and one geographic coordinate is projected over it:

```text
geographic response ~ s(acoustic coordinate 1, acoustic coordinate 2)
```

The available geographic responses are `geo-x`, `geo-y`, and `axis`, where `axis` is position on the Löderup–Arjeplog line. The script supports three acoustic-map scopes:

- `individual`: one map for each vowel;
- `pairs`: both vowels occupy one shared acoustic plane, with paired village observations kept together during testing and validation;
- `whole-space`: all selected vowels occupy one shared acoustic plane.

Vowel identity controls marker shape in shared maps but is not included as a model predictor. The plotted surface is restricted to the union of vowel-specific observed acoustic regions to reduce unsupported interpolation between vowel clouds.

The default command runs `geo_x` and `geo_y` projections for every method and all three scopes:

```bash
swedia-pca/bin/python Scripts/model_reverse_acoustic_geography_gams.py
```

Examples of narrower runs are:

```bash
# All vowels together in PCA space, with geo_x and geo_y overlays
swedia-pca/bin/python Scripts/model_reverse_acoustic_geography_gams.py \
  --methods pca \
  --targets whole-space

# Shared /uː/–/oː/ spaces with the Löderup–Arjeplog coordinate overlaid
swedia-pca/bin/python Scripts/model_reverse_acoustic_geography_gams.py \
  --targets pairs \
  --pairs 'uː,oː' \
  --geographic-responses axis
```

Default results are written to `Analyses/Reverse_acoustic_geography_GAMs/`. Because this direction is predictive, the cross-validated R² should be considered alongside the permutation p-value: a visually smooth or statistically detectable surface may still generalize poorly to villages left out during fitting.
