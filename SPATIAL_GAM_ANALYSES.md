# Spatial GAM analyses of vowels and vowel-pair midpoints


The following scripts provide complementary directions of spatial modelling. Both use village-balanced acoustic positions, two-dimensional tensor-product smooths, village-level permutation tests, and 10-fold cross-validation. Gotland and the Finnish regions are excluded by default. Geographic plots follow the source-map orientation, with larger `geo_y` values displayed farther down.

`Scripts/add_speaker_standardized_spectral_pca.py` creates an optional speaker-relative version of the spectral PCA scores. For each complete `(village, speaker)` record, PC1 and PC2 are standardized separately using midpoint-token means and sample standard deviations; those parameters are then applied at 20%, 50%, and 80%. The resulting `pca-speaker-z` method can be passed to the forward and reverse GAM scripts. Because this changes the relative scale of the PCA axes for each speaker, distances and angles belong to a speaker-relative geometry and should be compared with, rather than substituted for, the original spectral PCA results.

```bash
swedia-pca/bin/python Scripts/add_speaker_standardized_spectral_pca.py
swedia-pca/bin/python Scripts/model_spatial_vowel_midpoint_gams.py \
  --input Analyses/Speaker_standardized_spectral_PCA/formants-pca-speaker-z.csv \
  --methods pca-speaker-z \
  --output Analyses/Spatial_vowel_midpoint_GAMs_speaker_z
```

`Scripts/add_lobanov_formant_pca.py` likewise creates speaker-wise Lobanov F1/F2 measurements for fixed Praat and FastTrack. The GAM scripts expose these without the optional PCA rotation as the `praat-lobanov` and `fasttrack-lobanov` methods:

```bash
swedia-pca/bin/python Scripts/add_lobanov_formant_pca.py
swedia-pca/bin/python Scripts/model_spatial_vowel_midpoint_gams.py \
  --input Analyses/Lobanov_formant_PCA/formants-lobanov-pca.csv \
  --methods praat-lobanov fasttrack-lobanov \
  --output Analyses/Spatial_vowel_midpoint_GAMs_lobanov
```

After running the raw, speaker-standardized PCA, and Lobanov pipelines, `Scripts/compare_spatial_gam_methods.py` produces matched cross-validation heatmaps and a compact summary for all six representations. Its report is written to `Analyses/GAM_method_comparison/index.html`.

## Predicting acoustic position from geography

`Scripts/model_spatial_vowel_midpoint_gams.py` models each acoustic coordinate from the joint geographic plane:

```text
acoustic coordinate 1 ~ s(geo_x, geo_y)
acoustic coordinate 2 ~ s(geo_x, geo_y)
```

It can analyze all eight individual vowels and all six canonical paired-speaker vowel midpoints using spectral PCA, fixed-Praat formants, or FastTrack formants. For individual vowels, speaker measurements are aggregated to village medians. For midpoints, the two vowels are first paired within speakers and then aggregated within villages. Each target receives a self-contained HTML report with two geographic effect maps, an observed-versus-fitted acoustic-space plot, coordinate-wise and joint permutation tests, cross-validated R² values, village predictions, and a downloadable continuous surface.

The report also transforms a supported geographic grid through both fitted GAMs and draws the result as a mesh in acoustic space. Solid curves follow north-to-south movement at fixed `geo_x`, while dotted curves follow `geo_x` at fixed `geo_y`. Their bending, convergence, and separation show the nonlinear path fitted for an individual vowel or midpoint without first dividing villages into broad regions. This is the most direct view of hook- or cane-like spatial geometry in the acoustic plane.

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

For a focused PCA case study paralleling `u-o-pca-axis-report.html`, run:

```bash
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py
```

The same report can use raw fixed-Praat or FastTrack formants:

```bash
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py --pair 1 --method praat-fixed
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py --pair 1 --method fasttrack
```

Add `--bark` to either formant method to run the complete report in Bark-transformed F1/F2 space:

```bash
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py --pair 1 --method praat-fixed --bark
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py --pair 1 --method fasttrack --bark
```

Use `--timepoint 20`, `--timepoint 50`, or `--timepoint 80` to select the
measurement point within the vowel. The default remains 50%. This option applies
to PCA scores, raw formants, and Bark-transformed formants throughout the report:

```bash
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py --pair 1 --method pca --timepoint 20
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py --pair 1 --method fasttrack --bark --timepoint 80
```

Non-default timepoints add `_t20` or `_t80` to the output directory, so results
from different measurement points do not overwrite one another.

Method-specific output directories are generated automatically, so these runs do not overwrite the PCA report.

This writes the report and all of its assets under `Analyses/u_o_PCA_GAM_report/`, with `index.html` as the entry point. In addition to the individual-vowel and midpoint surfaces, it models the paired `/uː→oː/` vector through separate spatial GAMs for its two Cartesian components. The displayed direction and distance surfaces are derived from those fitted components. The report also fits complementary reverse GAMs that project north–south `geo_y` over each individual vowel space and over the pair's shared acoustic space.

Any directed pair can be selected from the eight vowels. Output names are generated from the pair:

```bash
swedia-pca/bin/python Scripts/build_vowel_pair_gam_report.py --pair 5
```

The aliases are `1` = `/uː→oː/`, `2` = `/oː→ɑː/`, `3` = `/ɑː→æː/`, `4` = `/æː→eː/`, `5` = `/yː→ʉ̟ː/`, and `6` = `/ʉ̟ː→øː/`. Explicit directed pairs such as `--pair 'yː,ʉ̟ː'` remain available.

`Scripts/build_u_o_gam_report.py` is a shorthand entry point for the default `/uː→oː/` report.

## Projecting geography over acoustic space

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
