# R6 implementation feasibility: separating pooled mean, amplitude and orientation

Date: 2 October 2026. Stage: **engineering GO for a bounded development pilot; scientific contribution CONDITIONAL**.

This note is read-only with respect to existing code, data, weights and article files. No model construction, forward pass, fit, new outcome calculation or reserved-period value inspection was run. The only new artifact is this note. Cost estimates use existing timing records rather than a fresh benchmark.

Route: SCI, Neurocomputing prediction/method study. The proposed knowledge increment is a controlled distinction between nonlinear pooling's mixed coordinate, amplitude-only detail and signed detail at a fixed forecasting product. Evaluation standards are fair architecture intervention, chronology, generalization and evidence matching. A Haar-style change of coordinates or another model name alone is insufficient novelty. No causal identification, acceptance estimate or full workflow file bundle is required for this feasibility decision.

## 1. What the current implementation actually does

### R4 household model

`79_revision_matched_nhits_v1.py`, `MatchedNHITS.forward`, receives completed normalized values and coverage. It reverses both; initializes the forecast from the last observed/completed normalized value; and processes three residual blocks. Each block predicts a full-length backcast and forecast knots. Its backcast is subtracted before the residual is multiplied by reversed coverage. The adapters therefore operate on **learned residual coordinates**, rather than solely on raw electricity values.

The default constructor has pools `[4,4,2]`, forecast downsampling `[8,4,1]`, input 336 and horizon 48. At width 256 it has 954,930 parameters. `95_revision_pooling_sensitivity_v1.py` changes the last pool to one, yielding 997,938 parameters. `revision2_capacity_predictor_v1.py`, `CapacityNHITS`, retains the original forward and can replace only `blocks[-1].pooling_layer`. Its `lift` condition performs MaxPool2 then repeats to length 336; its direct condition leaves all 336 coordinates accessible. Their learned tensor names, shapes and seeded initial values match. The independent exact-width control is a different structural intervention and is unnecessary for the proposed primary adapter family.

The installed NeuralForecast 3.2.2 `NHITSBlock.forward` accepts `[batch,length]`, unsqueezes to `[batch,1,length]`, applies `pooling_layer`, squeezes the singleton channel and sends the result into its MLP. There are no historical/future/static exogenous channels in these constructors. A zero-parameter length-preserving module can consequently implement all proposed final adapters without changing a head, hidden layer, backcast, interpolation or forward method.

### R5 Portuguese model

`122_upgrade5_pilot_v1.py`, `model_factory`, uses input 672, horizon 96, pools `[4,2,1]`, forecast downsampling `[8,4,1]` and three width-64 blocks. It has 240,036 parameters. **Its last block already has direct access.** A max or average comparison would be a newly refitted reference, not the unchanged published/R5 fitted forecaster.

R5 computes an observed-mask mean and population variance from each decoded past window. Its standard-deviation floor is `max(1 kW,0.05*abs(window mean))`. This shared pre-adapter normalization and the last-value forecast path must stay identical across the new adapter family. The prepared pooled scaler is unused. The failed ridge retention gate is outside this study: it should neither be retrained nor presented as the new mechanism.

In either model, a final-adapter restriction is **local**. Earlier backcasts are positional; the direct last-value forecast remains; coverage is applied to subsequent residuals; and end-to-end gradients can change earlier learned blocks. It would be incorrect to claim pair-swap invariance or complete loss of ordering for the whole network from the final pooling operation alone.

## 2. The identifiable local ambiguity

For a residual pair `(r1,r2)`, let `m=(r1+r2)/2` and `d=(r1-r2)/2`. Then

`max(r1,r2)=m+abs(d)`, whereas `average(r1,r2)=m`.

Max pooling entangles pair level and fluctuation magnitude. Mean pooling removes the fluctuation coordinate. Neither scalar distinguishes `(r1,r2)` from its swap. The identity holds for signed residuals; it should not be described as a physical appliance or optimizer mechanism.

Use orthonormal coordinates `c=(r1+r2)/sqrt(2)` and `a=(r1-r2)/sqrt(2)` for the proposed two-coordinate factorization. Scaling by `sqrt(2)` avoids an arbitrary factor-two contraction relative to the raw pair. The transform is the elementary Haar coordinate map: invertible, parameter free, and already mathematically familiar.

The clean five-level adapter factor is:

| Adapter | Two output coordinates per input pair | What remains accessible locally |
|---|---|---|
| Max-repeat | `(m+abs(d),m+abs(d))` | One mixed mean/amplitude coordinate |
| Average-repeat | `(m,m)` | Pair mean only |
| Mean/absolute detail | `(c,abs(a))` | Separate level and amplitude, without residual-pair orientation |
| Mean/signed detail | `(c,a)` | Both coordinates, invertibly |
| Direct | `(r1,r2)` | Both original coordinates, invertibly |

All adapters return the original length and contain **zero trainable parameters**. Construct the final library block with native kernel one, then replace its `pooling_layer`. Do not construct native kernel two and silently change the MLP input dimension. This gives exactly 997,938 parameters for R4 or 240,036 for R5 in every cell, with the same active learned tensors and no padding parameters. Repeated coordinates remain redundant; equal counts do not imply equal effective function classes.

Direct is essential. The signed factorization is invertible, and its first linear map can absorb the transform. If `T` is the pairwise orthogonal Haar matrix, set `W_MD=W_direct*T.T`: then `W_MD*T*r=W_direct*r`, with the remaining tensors unchanged. The full model can thus have the same initial function under the compensated transform. A gain over Direct would concern finite fitting, coordinate conditioning or initialization, rather than additional information or unrestricted representational capacity. This distinction prevents rebranding an invertible transform as a new information-preserving network theorem.

A pure max/average/signed comparison requires nine fits at three seeds, but cannot distinguish amplitude from orientation or a basis effect. The five levels and fifteen fits are the smallest convincing family for the proposed three-part interpretation. A two-adapter superiority trial would be too ambiguous.

## 3. Synthetic proof and engineering preflight to freeze before fitting

These are proposed checks, **not executed results**:

1. Check the local identity, inverse Haar transform and retained length for batches at 336 and 672, negative residuals and pair swaps. Confirm all trainable tensor names/shapes/counts across adapters.
2. Use exact counterexamples: `(0,2)` and `(1,2)` share max but differ in mean; `(0,2)` and `(1,1)` share mean but differ in amplitude; `(0,2)` and `(2,0)` share mean/absolute-detail but differ in orientation. These establish local collisions without relying on a trained model.
3. On clearly synthetic pairs with future targets depending separately on level, detail amplitude and orientation, retain all three response types. A signed-detail toy alone is insufficient to establish amplitude separation. State any synthetic generator and Bayes/oracle calculation explicitly.
4. Verify the compensated first-layer relation above and whole-network prediction agreement to a declared float32 tolerance. This is a diagnostic equivalence check, not a demand that all nonlinear adapters have the same initial function.
5. Verify finite gradients away from max ties/detail zero; test ties separately under the actual PyTorch subgradient convention. Pair layout is in **reversed residual order**; do not mistake its detail sign for forward chronological detail.
6. Use existing past-only fill and masks verbatim. Test a partial pair and leading missing slots; do not introduce a new imputation or feed a missing future target into the adapter.
7. Run a small source-only engineering smoke after code/protocol sealing. Do not select among adapters or budgets using its target results. Save/reload proof must identify the executed new code and preserved library hashes.

For the real comparison, library initialization draws and all batch indices should be generated once per seed and reused. The primary family can copy the same learned tensors across all five adapters, explicitly allowing changed initial functions because the inputs differ. The compensated Direct/MD comparison should at least be checked analytically and in the synthetic preflight. If the intended main claim is **MD superiority over Direct caused by conditioning**, freeze three extra compensated-initialization MD fits before outcome access; this raises the planned family to eighteen rather than adding a favorable control after seeing results. A mechanism claim about Adam still requires more than these fits.

## 4. Recommended first real pilot and a lower-engineering-cost alternative

### Recommended: previously unscored Portuguese development gate

Use the existing 80-column UCI cache and fixed customer groups; add no download and inspect no 2014 outcome values. Refit all five adapters, including Direct, on the same first 24 development customers and the same 2012 fitting/held-time validation origins used by R5. Use 2,000 updates, width 64, batch 64, Adam 0.001, the same gradient clipping and normalization, and one prespecified checkpoint rule applied equally. The existing history-normalized masked training loss must be explicitly retained if reused; it is not identical to raw customer-macro MAE.

Evaluate every frozen fit on **all sixteen calibration customers, 1 July–31 December 2013**, with input 672 and output 96 at the same daily origins. These outcomes are already encoded in the prepared panel but have not been scored or used to fit the earlier predictor/gate. This is a new development comparison, not untouched raw data or final confirmation. Customer columns 32–47 stay fixed; later coverage or error cannot change eligibility. Use one common observed-target mask and retain non-scoreable IDs in the roster. All future targets remain fine.

Seeds: `20261001,20261002,20261003`. Adapter is the only comparison factor; no history/coarse-encoding or auxiliary-loss search is added in this first pilot. Save and report all fifteen fits. The main metric is equal-customer endpoint MAE in physical kW; also report fixed-prior-support normalized MAE, endpoint RMSE and pair mean/detail MSE. Choose and declare the normalized support rule before scoring. Because fitting and physical evaluation have different scale weights, never describe the reused fitting loss as exactly aligned with the primary macro-kW score. The Direct cell and simple daily/weekly references anchor absolute accuracy; their storage cost is outside this adapter-only experiment.

The pilot should discriminate these questions rather than pick the smallest p-value: does separate amplitude help relative to average pooling; does signed orientation add beyond absolute detail; and does signed factorization behave differently from the full-information Direct coordinates? Report seedwise contrasts and both favorable and adverse endpoints. Local pooled-residual diagnostics can describe the implemented intervention, but full-network learning effects cannot be attributed to one information loss from the scalar identity alone.

### Alternative: established London source-only pilot

Reuse the immutable `cache/revision_matched_nhits_v1/lcl_validation_full` cache: 2,672 fitting IDs, 211,088 rows, 79 origins each; 668 held-source IDs, 667 previously scoreable. Training cutoff is before September. Each of the five adapters uses width 256, 3,000 updates, batch 256, identical Adam/scheduler/gradient clipping, the existing raw-kW complete-pair household-balanced objective, and fine history B30 only. September source evaluation and fixed source/target-like support are unchanged.

This has the best loss/mask continuity with R4 and almost no new pipeline code. However, its source outcomes are already known, so it cannot serve as newly independent confirmation of the redesigned scientific claim. It is an outcome-informed development extension; the fifteen new fits must be sealed and all reported. Existing Direct/lift results can be contextual references, but a new three-seed pilot should refit all five at the same execution configuration to avoid a reused single-seed/changed-initialization comparison.

## 5. Actual resource basis and preserved confirmation boundary

No new speed measurement has been made. Existing London lift B30 fits consume 125.25/127.6875/133.171875 process-CPU seconds for 3,000 updates; recorded wall times are 161.91/198.67/272.26 seconds under competing workloads. Fifteen comparable fits imply about **32–36 CPU minutes**, with roughly 40–75 sequential wall minutes under those historical loads. Two one-thread processes can overlap independent fits; elapsed time is not guaranteed. R4 checkpoints are approximately 4 MB, so fifteen models need about 60 MB plus source forecasts and provenance. The existing fine training cache is about 284 MB and can be opened read-only.

The R5 direct fit records 26.7147 wall seconds and 50.84375 CPU seconds for 2,000 updates on two Torch threads. Fifteen refits imply approximately **13 CPU minutes** at the recorded compute rate. Allow 13–20 CPU minutes plus preparation/inference/reporting; a one-thread implementation requires a new source-only smoke to confirm elapsed speed. Input adapters add linear-time pair arithmetic and no learned matrix cost. The larger calibration window has 2,944 nominal customer-day forecasts before any score-mask exclusions. A new cached source/validation history subset should fit comfortably below 0.2 GB; checkpoint and prediction artifacts are small relative to available D space. D had 135,181,742,080 free bytes at this inspection. API cost is zero; no new package or GPU is needed.

The preserved UCI confirmation group comprises **32 fixed customers, columns 48–79**, with 2014 starting at provider slot 105,216. Its manifest says `confirmation_consumed_for_fitting_tuning_or_scoring=false`; the existing independent empirical audit says confirmation was not scored. Preparation already converted and hashed the fixed arrays and reported availability counts, so do not call the data numerically unseen. No confirmation outcome values were newly read for this note.

Only after a sealed development decision should a separate protocol declare the 2014 question, final refitting cutoff, all retained adapters, simple references, primary contrasts and uncertainty. Past support from confirmation customers is allowed only under the declared arrived-input rule, never for outcome-dependent customer selection. A signal limited to the known London period or to one coordinate initialization does not justify consuming the confirmation set to search for a preferred winner.

## 6. Decision and next concrete artifact

The implementation is tractable. Recommend the fifteen-fit Portuguese calibration pilot if the goal is a previously unscored development result; use the London source family when immediate exact continuity of loss/masks is the priority. Seal one choice before running; do not run both and retain whichever story is favorable. A new predictor wrapper can inherit the current forward and replace only the final zero-parameter adapter. It must write an independent protocol/lock, prediction completion proof, checkpoint-reload provenance and all-cell score table under new R6 paths. None of the frozen 79/95/capacity/121/122/124 artifacts should change.

The current status is **GO for bounded execution planning; CONDITIONAL for a publishable new core**. Exact collision identities and nominal count matching are established design facts, not empirical research success. A meaningful advance would be replicated error-allocation or forecasting evidence that separates amplitude mixing from orientation access and remains valid in a later frozen period. An invertible mean/detail transform that merely matches Direct, or tiny endpoint differences without stable interpretation, should not be advertised as a new forecasting architecture. Literature differentiation remains a separate required check; this note makes no novelty-priority claim.

Forecast-aware bitplane encoding is technically possible through the existing once-only quantized state, but it adds coding, bitplane selection and complete-cost confounds before addressing the current neural alias question. It is a secondary alternative, not the proposed pilot, and the failed R5 ridge selector should not be replayed with a new label name.

## Inspected evidence

- `scripts/79_revision_matched_nhits_v1.py`: residual forward, objective, fixed masks and cache handling.
- `scripts/95_revision_pooling_sensitivity_v1.py`: direct final access and count change.
- `scripts/revision2_capacity_predictor_v1.py`: exact-shape max-repeat implementation and initial-tensor checks.
- Installed `neuralforecast/models/nhits.py`: adapter entry point and MLP construction.
- `scripts/121_upgrade5_stream_v1.py`, `122_upgrade5_pilot_v1.py`, `124_upgrade5_integer_codec_v1.py`: past-only normalization, original development fit and encoded-state boundaries.
- `cache/revision_matched_nhits_v1/lcl_validation_full/cache_manifest.json`, preserved capacity fit JSONs and `results/revision2_lcl2014_v1/staged_cost_summary_v1.csv`: existing data sizes and costs.
- `datasets/upgrade5_uci_v1/roster_v1.json`, `partitions_v1.json`, `dataset_manifest_v1.json`; preparation/empirical PASS audit records: fixed customer/time groups and unused confirmation boundary.

No above artifact was modified and no fitting/scoring process was launched for this feasibility note.
