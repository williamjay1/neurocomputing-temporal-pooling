# R6 research protocol: predictive information after temporal access

Date: 2 October 2026. Status: research design fixed before newly reserved customer/period outcomes are scored; implementation identity supplements will bind the executed scripts. This is a new hypothesis arising after inspecting R4/R5, not a preregistered account of the earlier results.

Route: SCI original theoretical/empirical neural forecasting research, target Neurocomputing. The primary standards are a specific knowledge increment, sound conditional analysis, matched neural interventions, held-out customers/times and reproducible evidence. No social-science causal identification, new-codec branding or broad benchmark leadership is required or claimed. The sci-full-workflow, experiment-design and quantitative-analysis skills are used for applicable modules only; evidence and protocol are consolidated rather than proliferating mechanical gate files.

## Claim spine and closest work

Problem: finer raw histories contain more information, but a fixed nonlinear neural input operation can make that information inaccessible or combine it with nuisance variation. Input-resolution recommendations therefore cannot follow from raw-information nesting or parameter count alone.

Increment under investigation: an explicit forecast DGP for which a nonlinear maximum operation produces a nonnested observation relative to coarsening first, with an exact task-dependent Bayes paired-MSE crossover; coordinate-isolating neural interventions test amplitude mixing, discarded orientation and mere invertible reparameterization separately. This is a targeted analytical and empirical finding, not a claim to invent maximum/average pooling, Haar transforms, statistical sufficiency or anti-aliasing.

Closest work must include Blackwell's experiment comparison, Boureau/Ponce/LeCun's pooling analysis, mixed/gated pooling, Zhang's BlurPool and ICML 2026 DMANet. Fourier/Nyquist aliasing is distinct from the nonlinear absolute-detail folding analyzed here. The original primary-source neighbor audit will determine wording. DySCo/Cadence and the actual unsuccessful R5 selector remain documented but are not the proposed R6 core.

Boundary: the oracle has no bypass, independent nondegenerate centered Gaussian latents and squared loss. The full N-HiTS estimator has earlier backcasts, normalization and a last-level path. The oracle is a constructive restricted-access model, not a causal explanation of Adam, a global theorem for N-HiTS or a claim to explain the London calendar reversal. Neural MAE evidence and oracle MSE claims remain distinct.

## Analytical model

Let independent C~N(0, sigma_C^2) and A~N(0, sigma_A^2), with both variances positive. The observed pair is (C-A,C+A). Define centered magnitude D=|A|-E|A|, V=Var(D)=sigma_A^2(1-2/pi), and U=C+D. Adding the known centering constant does not change the information in the raw pair maximum.

The future pair has mean C+alpha D+epsilon_C and signed detail beta A+epsilon_A, with independent zero-mean Gaussian innovations. Restrict alpha to [0,1]; beta is any fixed real value. Denote W=E Var(D|U). Conditional projection and sign symmetry imply 0<W<V and:

- coarse mean access C: R_C=alpha^2 V+beta^2 sigma_A^2+innovation risk;
- maximum access U: R_U=(alpha-1)^2 W+beta^2 sigma_A^2+innovation risk;
- separated mean/magnitude access (C,D): R_abs=beta^2 sigma_A^2+innovation risk;
- raw or signed-Haar access (C,A): innovation risk.

Therefore R_U-R_C=(alpha-1)^2 W-alpha^2 V. The unique switch in [0,1] is alpha_star=sqrt(W)/(sqrt(W)+sqrt(V)), strictly between zero and one half. This is an application of established Bayes projection, not a new general Blackwell theorem. Negative alpha is outside the unique-switch statement. No nonnegative clipping is applied to this centered synthetic DGP.

Numerical W is obtained from the truncated-normal posterior and quadrature at two resolutions. It is not advertised as a closed-form constant. Parameter grid: alpha 0/.25/.5/.75/1; beta 0/.5; sigma_C=1 and sigma_A .25/1/2, with innovations SD .1. Save integration precision and all risk values.

## Controlled synthetic neural validation

Two locked scenarios: alpha=0 or 1, beta=.5, sigma_C=sigma_A=1, innovation SD .1. Independent fixed generation draws supply 20,000 training, 10,000 validation and 100,000 test pairs. Generation seeds are recorded before test metrics, with no selection based on test outcomes.

Five adapters have the same two-coordinate output: direct, repeated maximum, repeated average, orthonormal mean/absolute-detail and orthonormal mean/signed-detail. A two-hidden-layer 64-unit ReLU MLP maps to two future endpoints. The signed-Haar first affine is rotated to give the same initial direct function; the absolute variant uses the same canonical rotation, then discards sign deliberately. Every learned tensor shape, batch-row sequence, update budget and checkpoint rule is held constant within each seed.

Three seeds (20261001/2/3), 2,000 Adam updates at .001, batch 256, validation-MSE checkpoint every 200 updates: 5x2x3=30 small fits. Primary outcome is paired-endpoint MSE; MAE is secondary. Compare fitted risks with actual oracle floors and report approximation/finite-fit gaps without asserting Bayes attainment. Use the same test draws across adapters. Synthetic observations are always identified as synthetic.

## Matched Portuguese customer/time experiment

Use the immutable, previously prepared UCI ElectricityLoadDiagrams20112014 cache and versioned data manifest. Official source is CC BY 4.0, native fifteen-minute kW, 370 customers. The frozen 80-client roster uses 2012 eligibility; activation/DST masks and valid postactivation zeros remain unchanged. No new dataset or changed roster is selected for favorable performance.

Fit to the same first 24 development clients, January–September 2012, with held-time 2012 checkpoint selection. Evaluate all 16 reserved calibration clients in July–December 2013. The separate 32-client 2014 confirmation partition remains unopened until the full family, implementation, metrics, checkpoint rule and analysis are locked. There is no test-time tuning or client filtering by errors. A full 2014 replay, if advanced, uses the same already fitted models and all declared adapters rather than a winning subset.

Six final residual adapters: direct, max-repeat, average-repeat, orthonormal mean/absolute-detail, orthonormal mean/signed-detail, and a precisely documented BlurPool-style fixed low-pass maximum reference. The latter is a standard reference, not a native reproduction of DMANet. Replace only the final input adapter of the same three-block 672-to-96 width-64 N-HiTS model initialized with final kernel one. Learned parameter count and shapes stay 240,036. Signed-Haar initialization preserves the direct initial function within declared float32 tolerance. Noninvertible adapters intentionally change accessible information; equal learned parameters are not asserted to make their function classes equal.

Cross all six adapters with fine history versus observed-only adjacent means/counts, and three seeds 20261001/2/3: 36 fits. Fine/coarse completion and normalization read only that representation's past. Common complete future-pair targets and masks, causal filling, row sampling, Adam .001, 2,000-update budget, batch 64 and source-only checkpoint rule apply. Physical predictions are not clipped, consistently with the actual R5 Portuguese forward. This corrects a design-text mismatch found during source-only preflight, before calibration inference or scoring. Per-window means/scales and existing model bypasses are disclosed as part of the whole forecasting map.

Primary real endpoint: customer-macro prior-scale-normalized endpoint MAE. Physical MAE and paired mean/detail MSE are secondary. The fixed evaluation scale is an observed pre-period support mean floored at one kW, identical across adapters/history for each customer; its exact support dates are locked in the implementation record. Operational MAE is not inferred from the squared-risk theorem.

Primary contrast: the max-repeat fine-minus-coarse MAE difference minus the corresponding average-repeat difference, at common targets. Planned diagnostic contrasts are max versus mean/absolute on fine history, absolute versus signed on fine history, and signed versus direct on fine history. Blur versus maximum is a standard-filter boundary. Report all six-by-two cells and all three seeds. Orthogonal coordinate changes are interpreted as finite-training effects; no new information advantage is attributed to signed versus direct.

Uncertainty: paired resampling of whole customers and calendar-week blocks, using a joint draw across all cells and averaging the three fixed fitted seeds. Preserve seed-specific contrasts separately. Confidence intervals describe sampling variation conditional on these fits, not infinitely many training seeds or millions of independent windows. Report magnitude and interval, not only a p value. The modest number of held-out clients is retained in interpretation.

## Preflight, resource and stopping conditions

Before calibration scoring, verify adapter shapes, masks, causal chronology, exact learned-tensor counts, signed/direct initial-function alignment, coarse-pair values, all-positive/negative examples and finite losses. Bind code and data hashes. Save source validation decisions before reading calibration outcomes. At most two Torch CPU threads per worker; all training, arrays, checkpoints and results go to D. D free space checked at 140.2 GB; no new raw dataset is needed. Source papers use new read-only F snapshots when collected. Existing C/E/F archives are untouched.

The failed R5 ridge selector remains stopped. R6 does not rehabilitate it by changing metrics, discarding adverse baselines or reusing 2014 labels. The new core must be supported by the proof, synthetic fitted responses and honest whole-network matched evidence. If the real effects are negligible or bypasses defeat the simple restriction, state that boundary; do not relabel an invertible Haar coordinate change as a novel architecture. No acceptance probability or numerical publication score is promised.

## Planned article structure

Implementation reconciliation before new held-out outcomes: synthetic generation seeds are 13220261001/13220261002/13220261003 for train/validation/test, respectively. The synthetic network has 4,482 learned coefficients. The Portuguese network has 240,036 registered learned coefficients, including 43,680 identical forecast-inactive final backcast coefficients; the structurally forecast-active count is 196,356 in every condition. Calendar-phase support for calibration is 24–30 June 2013, with daily forecasts 1 July–31 December 2013. The real implementation uses no forecast clipping; this correction is made before calibration outcomes are opened. Cheap persistence, daily and weekly forecasts are quality references on the same target mask, not parameter-matched adapter conditions or codec comparisons. Precise source-only implementation identity and additional analysis, if any, must be bound in a root-authored addendum before calibration inference. The family is not enlarged after seeing new held-out differences.

The manuscript will be rebuilt around the post-access information question rather than around the list of historical experiments. Introduction: why raw information nesting is insufficient, closest pooling work, exact scoped increment. Theory: access maps, oracle risk threshold and signed/magnitude information. Methods: synthetic and matched customer/time protocols. Results: oracle/neural threshold, full matched real comparisons and uncertainty, earlier market/period boundary as contextual evidence. Discussion: choosing accessible coordinates, role of bypasses/normalization, difference from Fourier aliasing and coding, deployment limits. R5's actual failed selector and complete-cost finding remain concise in main text and fully available in the supplement. No claims will be drafted as completed until bound outputs and independent checks exist. Formatting work is deferred; the deliverable is the complete scientific manuscript.
