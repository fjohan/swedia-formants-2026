# Bark-filtered spectral PCA of SweDia vowels: methodology draft

## Scope and status

This document describes the exploratory analysis implemented in
`analyze_bark_pca_pilot.py` and `plot_pca_angle_maps.py`. The procedure is
inspired by whole-spectrum vowel analyses using PCA of Bark-filtered spectra,
particularly Jacobi, Pols, and Stroop (2006) and Leinonen (2010). It is not yet
an exact replication of either study. Choices that still require sensitivity
testing are identified below.

The present analysis includes 73 younger male speakers from 24 SweDia
locations. These recordings were selected by extracting the recording IDs from
the plots in `Analyses/CombinedSpeakerPlots_original_both_bark`. Consequently,
this is a convenience sample of recordings already included in an earlier
formant analysis, rather than a new independently defined sample.

Eight long-vowel categories were analyzed:

| Internal label | IPA label | Base word |
|---|---|---|
| `u:` | /uː/ | *sot* |
| `o:` | /oː/ | *låt* |
| `A:` | /ɑː/ | *lat* |
| `ä:` | /æː/ | *nät* |
| `e:` | /eː/ | *leta* |
| `y:` | /yː/ | *typ* |
| `U:` | /ʉ̟ː/ | *lus* |
| `ö:` | /øː/ | *söt* |

The larger analysis also includes the explicitly approved alternative lexical
items defined by `inventory_base_word_targets.py`: *rot* for /uː/; *lås*,
*gråt*, and *båt* for /oː/; *läs* and *fräs* for /æː/; *gles* for /eː/; and
*blöt* and *lös* for /øː/. These alternatives are assigned to the corresponding
base-vowel category. The later length-contrast group of tokens of *låt* is not
included. Surface segment labels are retained in the output but are not used to
exclude tokens in the main 24-village analysis. A separate strict-label pilot
is available for comparison.

## Segmentation and token selection

Word and segment intervals were read from the `ord` and `seg` tiers of the
TextGrid corresponding to each recording. A lexical target was retained only
if its word interval contained exactly one vowel-like segment, determined from
the midpoint of the segment interval. Six otherwise eligible lexical tokens
were excluded because this condition was not met.

The main analysis contains 3,557 retained vowel tokens. Measurements were made
at 10%, 25%, 50%, 75%, and 90% of each vowel interval, producing 17,785
token-time observations. The normalized time points allow both midpoint vowel
spaces and within-vowel spectral trajectories to be examined.

## Signal processing and auditory spectra

Audio was converted to floating-point amplitude. Recordings not sampled at 16
kHz were resampled to 16 kHz using polyphase resampling. All recordings in the
current set are mono, 16-bit PCM.

At each normalized vowel time point, a 25 ms frame centered on the measurement
time was multiplied by a Hann window. Its power spectrum was calculated with a
1,024-point real FFT. This zero-padding gives a dense frequency grid for
integrating filter energy; it does not increase the underlying spectral
resolution of the 25 ms observation.

Frequency in Hz was transformed to Bark using the Traunmüller formula:

```text
Bark(f) = 26.81 / (1 + 1960/f) - 0.53
```

Twenty overlapping triangular filters were placed at one-Bark intervals, with
centers from 1 through 20 Bark and outer edges at 0 and 21 Bark. For each filter,
weighted power was divided by the sum of its filter weights and converted to
decibels. Each 20-element spectrum was shifted by a common additive constant so
that its summed filter energy corresponded to a reference level of 80 dB. This
removes overall level while retaining the relative spectral shape. The original
frame RMS in dBFS is retained as a diagnostic variable.

This implementation is an explicit triangular-filter approximation. It does
not yet average the first two filters, use the male-specific 2–17 Bark range,
or reproduce the exact historical Praat filter implementation. These should be
treated as planned robustness analyses rather than as details of the current
method.

## Construction of the PCA

The PCA was fitted in a shared space across all 73 speakers, eight vowels, and
five time points. Before fitting, repeated lexical tokens were averaged within
each combination of recording, vowel category, and normalized time point. This
gave each available speaker–vowel–time cell one row in the PCA and prevented a
speaker or vowel with more repetitions from contributing disproportionate
weight. The resulting fitting matrix contained 2,885 rows and 20 Bark-filter
variables. Seven of the possible speaker–vowel cells were absent; missing cells
were left missing rather than imputed.

The mean of each Bark-filter variable was subtracted across fitting rows. The
variables were not divided by their standard deviations. Unrotated principal
components were then obtained by singular-value decomposition of the centered
matrix. Thus, this is covariance PCA of the dB filter levels rather than PCA of
a correlation matrix. The same variable means and component loadings were used
to project both the balanced fitting observations and every individual token
observation.

The first three components explained 45.2%, 21.4%, and 9.3% of total variance,
respectively, or 75.9% together. Component signs are mathematically arbitrary;
their signs have not been post-hoc aligned to conventional F1 or F2 axes. The
global PCA provides a common coordinate system, and the preceding level
normalization removes overall amplitude. It should not, however, be described
as removing all speaker effects: vocal-tract anatomy, F0, voice quality,
microphone response, and background noise can remain in the spectral shape.

## Village vowel positions and trajectories

For visualization, the projected observations were averaged in two stages.
First, repeated tokens had already been averaged within speaker, vowel, and
time point for the balanced PCA matrix. Second, these speaker-level values were
averaged across speakers within each village. Village midpoint plots use the
50% observations. Village trajectories connect the grand means at 10%, 25%,
50%, 75%, and 90%. This procedure gives equal weight to each available speaker
within a village rather than weighting villages by their numbers of tokens.

Individual-token and individual-speaker displays are also supported, but the
current 24-village overview shows only the grand village means.

## Vowel-space orientation

Six directed relations were defined between the village-level midpoint vowel
positions:

```text
/uː/ → /oː/
/oː/ → /ɑː/
/ɑː/ → /æː/
/æː/ → /eː/
/yː/ → /ʉ̟ː/
/ʉ̟ː/ → /øː/
```

The midpoint of each directed relation was calculated. Two covariance ellipses
were then fitted independently for every village: one to the eight vowel
positions and one to the six relation midpoints. Ellipse centroids and axis
lengths were obtained from the eigenvalues and eigenvectors of the two-dimensional
covariance matrix. The displayed outlines have radii of two standard deviations
along the major and minor axes.

Orientation is reported as signed angular deviation from the vertical PC1 axis.
An angle near 0° therefore denotes a vertical vowel space. Positive values lean
upward toward increasing PC2, and negative values lean downward toward
increasing PC2. Plot panels use equal scaling of the PC1 and PC2 axes so that
the visual slope agrees with the numerical angle.

When the major and minor axes have similar lengths, the major-axis direction is
poorly determined: a small change in the input positions can cause a large
change in its angle. Orientations with a major/minor axis ratio below 1.2 are
therefore flagged as unstable. Their numerical eigenvector angles remain in the
CSV for auditing, but they are not interpreted or mapped as zero.

## Geographic visualization

Village coordinates were read from fields 5 and 6 of `resource.txt`. The
resource Y coordinate was inverted for plotting so that northern locations
appear toward the top. Reliable signed angles were mapped using a diverging
color scale centered on 0°, with identical limits for the vowel-position and
midpoint maps. Unstable orientations were shown in neutral gray with an ×,
distinguishing an unidentified direction from a reliable angle of 0°.

The maps suggest a possible broad north–south tendency in vowel-space
orientation. At present this is a visual, exploratory observation only. The
sample contains 24 spatially uneven locations, nearby observations are not
independent, and angle precision varies with ellipse eccentricity and the number
and stability of underlying speakers. No geographical association should be
claimed until the pattern has been quantified and checked against uncertainty,
spatial autocorrelation, spectral range, lexical composition, and recording
conditions.

## Current outputs and reproducibility

The principal analysis outputs are in `Analyses/BarkPCA_24_villages`:

- `token_spectral_pca.csv`: token-level filter values and PC scores;
- `balanced_speaker_vowel_time_pca.csv`: PCA fitting observations and scores;
- `pca_loadings.csv` and `explained_variance.csv`: PCA definition;
- `village_pca_ellipses.csv`: ellipse geometry and reliability flags;
- midpoint, trajectory, loading, scree, and geographic map figures; and
- `run_settings.json`: complete command-line configuration and recording list.

The analysis can be reproduced with:

```bash
swedia-pca/bin/python analyze_bark_pca_pilot.py \
  --recordings-from-plots Analyses/CombinedSpeakerPlots_original_both_bark \
  --output Analyses/BarkPCA_24_villages

swedia-pca/bin/python plot_pca_angle_maps.py
```

## Priority checks before inferential analysis

1. Repeat the analysis with the historical low-filter averaging and with a
   male-oriented 2–17 Bark range.
2. Quantify agreement between PC1/PC2 and independently measured Bark F1/F2.
3. Bootstrap speakers within villages to obtain uncertainty intervals for vowel
   positions, ellipse angles, and axis ratios.
4. Assess whether recording source, RMS, noise, or spectral tilt predicts PC
   position or ellipse orientation.
5. Test the north–south association using a spatial model or permutation scheme
   that respects geographical dependence.
6. Examine angle together with axis ratio and ellipse scale, since compression
   or expansion of the vowel space affects the stability and interpretation of
   its orientation.

## References motivating the approach

- Jacobi, I., Pols, L. C. W., & Stroop, J. (2006). *Measuring and comparing
  vowel qualities in a Dutch spontaneous speech corpus*. Interspeech 2006,
  701–704. <https://www.isca-archive.org/interspeech_2006/jacobi06_interspeech.html>
- Leinonen, T. (2010). *An Acoustic Analysis of Vowel Pronunciation in Swedish
  Dialects*. University of Groningen. <https://research.rug.nl/en/publications/an-acoustic-analysis-of-vowel-pronunciation-in-swedish-dialects/>
