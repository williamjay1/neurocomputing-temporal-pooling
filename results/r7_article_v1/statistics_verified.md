# Post-review statistical extension of the frozen customer/time study

These analyses were added in response to review after the complete R6 outcomes were known. No forecasting fit, split, selected checkpoint or original primary interval changed.

The primary maximum-versus-average history interaction remains a separate prespecified contrast in each evaluation phase. The four planned fine-history diagnostics form one family within each phase. From the original 5,000 paired customer/calendar-week draws, we center each diagnostic replicate at its observed estimate, divide by its conditional bootstrap standard deviation, and use the 95th percentile of the largest absolute standardized deviation for simultaneous intervals. Bonferroni percentile intervals at 0.00625 and 0.99375 are retained as a cross-check. These conditional-bootstrap adjustments do not create a global family across all historical experiments.

| Phase / diagnostic | Estimate (NMAE) | Four-contrast simultaneous 95% interval | Bonferroni percentile interval |
| --- | --- | --- | --- |
| calibration / Maximum − magnitude, fine | 0.000336 | [-0.000792, 0.001463] | [-0.000692, 0.001659] |
| calibration / Magnitude − signed, fine | 0.000217 | [-0.000963, 0.001396] | [-0.000917, 0.001453] |
| calibration / Signed − direct, fine | -0.000658 | [-0.002224, 0.000909] | [-0.002418, 0.000788] |
| calibration / BlurPool-style − maximum, fine | 0.000749 | [-0.000784, 0.002282] | [-0.000623, 0.002458] |
| confirmation / Maximum − magnitude, fine | 0.000986 | [-0.001425, 0.003397] | [-0.000770, 0.004103] |
| confirmation / Magnitude − signed, fine | -0.000567 | [-0.001925, 0.000791] | [-0.002071, 0.000753] |
| confirmation / Signed − direct, fine | -0.002215 | [-0.005619, 0.001188] | [-0.006402, 0.000384] |
| confirmation / BlurPool-style − maximum, fine | 0.000873 | [-0.001142, 0.002887] | [-0.001217, 0.003007] |

All four simultaneous diagnostic intervals include zero in each phase. Their adjustment is applied regardless of the observed direction.

Precision is described by the standard deviation of the original matched bootstrap estimates. A local normal-shift approximation gives the effect magnitude corresponding to approximately 80% power at a two-sided 5% threshold as (1.959964 + 0.841621) × SE. This is a post-review description of current-design detectability, conditional on the three fitted models and the observed customer/week structure. It is not observed power, a prospective calculation or a claim that increasing the sample would reveal a desired effect.

| Phase | Primary bootstrap SE | Approximate 80% MDES (NMAE) | Original primary 95% interval |
| --- | --- | --- | --- |
| calibration | 0.000850 | 0.002381 | [-0.003246, 0.000107] |
| confirmation | 0.000781 | 0.002187 | [-0.002971, 0.000190] |

As a secondary sensitivity, paired weekly NMAE differences were fitted to a Gaussian crossed random-intercept model with customer and calendar week as separate random components. Each observed customer/week mean has equal weight; hence the intercept targets an equal-customer/equal-week contrast, which differs from the original availability-weighted customer macro. The complete grids contain 16 × 27 and 32 × 53 responses. Restricted likelihood was maximized under nonnegative component constraints, with exact zero-variance boundaries retained; independent dense covariance fixtures validate the orthogonal likelihood and its constrained optimum.

| Phase / three-fit mean primary | Equal-week estimate | Approximate model 95% interval | Customer variance | Week variance | Residual variance | Residual lag-one correlation |
| --- | --- | --- | --- | --- | --- | --- |
| calibration | -0.001448 | [-0.002912, 0.000016] | 7.52e-06 | 2.01e-07 | 3.47e-05 | 0.038 |
| confirmation | -0.001472 | [-0.002928, -0.000017] | 1.61e-05 | 0 | 8.04e-05 | 0.099 |

All five contrasts are also fitted separately for each of the three fixed seeds, retaining every fit. Three seeds are not used to estimate a seed population variance. Full tables retain zero-component flags, residual skew/kurtosis and adjacent-week dependence diagnostics. Gaussian component distributions and conditionally independent weekly residuals are working sensitivity assumptions; serial dependence may remain. The normal intervals plug in estimated variance components and do not account for their estimation uncertainty, especially at boundaries. They therefore do not replace the paired block-bootstrap primary inference or its intervals spanning zero.

The two phases change both customers and period. Neither this sensitivity nor the precision calculation attributes their differences to drift, seasonality or a causal mechanism.

Boundary accounting, calibration: 0/20 customer components and 1/20 week components are exactly zero at the constrained REML optimum.
Boundary accounting, confirmation: 0/20 customer components and 8/20 week components are exactly zero at the constrained REML optimum.

All input/output identities, complete paired weekly response CSVs, bootstrap draws, multiplicity and precision CSVs, model fits and engineering fixtures accompany this note.
