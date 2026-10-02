# R6 independent numerical audit

Scientific-content audit, 2 October 2026. This report concerns reproducible numerical agreement and claim boundaries, not publication acceptance. The independent script is `scripts/133_r6_independent_score_v1.py`; it imports none of the fitting producers130,131,132 or their metric helpers. Read-only library blocks are used to reconstruct the neural forward for source-checkpoint verification. Formal fitting was not repeated.

## Fixed scope and identity

The final scorer SHA256 is `c7effdc77c58b4e70fa58dd98eb7eecb68a47ba9390be73bd638d6259bde6d0b`. It binds the root protocol `c9e76c448cc384c82b82c0b2f72afeca5f3cea4a25664657c1d30670cb372a04` and implementation/confirmation addendum `798a8573484cc0e0a10fb31792b7c9840146c89ebe863758bb5546f4b7123e99`. Calibration and confirmation share the same fixed scoring/uncertainty kernel. The three simple references are retained; no winner-based selection is used.

All statuses below describe executions that actually completed. Independent engineering, oracle, synthetic30, source36, calibration36+3, confirmation36+3 and all288 source residual responses have passed. The source family contains36 fitted models reused in both real periods, not72 independent real-data fits.

The original `engineering_checks.json` records the earlier scorer identity `26c2d8...` and is preserved as historical. The active identity is `engineering_checks_final.json`, SHA256 `14e77356d3356ffc1c926199ddc497dbbc834a1ce779e6093e05e972b7b57237`.

## Engineering and source provenance — PASS

Independent fixtures verified missing-input poisoning, fractional constituent coverage, observed-only coarse reconstruction, invariance of the coarse history to mean-preserving signed-detail perturbations, equal learned tensor shapes, finite forward/backward, and mean/detail squared-error decomposition. Direct versus compensated signed-Haar initial forecasts differ by at most1.90735×10⁻⁶kW in the synthetic NHITS fixture; the two-coordinate synthetic MLP difference is5.96046×10⁻⁸. These are engineering fixtures, not study results.

All six real adapters have240,036 registered coefficients. The final head's672 backcast rows contain43,680 coefficients whose gradients are exactly zero in these forecast computations. The196,356 structurally prediction-active count is identical across cells. Registration, gradient presence of a tensor and structural activity of each scalar coefficient are distinguished.

The source audit independently reconstructed all11,952 training and104 held-time row keys, past values/masks, future source labels/masks, and the decoded fine/coarse histories and coverage arrays. Every numerical source window lies beforeOctober2012. All three2000×64 minibatch-index arrays match their fixed seeded construction. All36 source-selected checkpoint hashes, tensor counts/shapes and validation traces pass. Reconstructed library models safely reload each NumPy checkpoint and recompute held-time NMAE with an independent float64 formula: maximum disagreement with producer float32 scores is1.5832727129350133×10⁻⁸.

The source completion SHA256 is `5bc798dbd36b0983f8a674b927d9449d5f5747cdc11e7e4857ae914d15e8a5f5`. Independent source audit SHA256 is `1b7e06fda197050b11a2f6032d41653658a116a47acd468f9a4f00fb9d1a590b`. The complete source audit and checkpoint-score CSV are in `results/r6_independent_v1`. No calibration or confirmation numeric outcome was read by this source command.

## Constructed Gaussian oracle and30-model scores — PASS

W was computed independently by integrating conditional variance and, separately, by subtracting the variance of the posterior mean from the marginal variance. Both identities agree within6×10⁻¹⁶; two numerical precision settings agree. The conditional signed-orientation numerator vanishes in independent quadrature fixtures. These checks support the stated Gaussian calculation and do not constitute a new statistical comparison theorem.

|sigma_A|V|W|alpha switch|
|---:|---:|---:|---:|
|0.25|0.02271126423|0.02220158907|0.4971628858|
|1|0.3633802276|0.2597310210|0.4581229353|
|2|1.453520911|0.5479547814|0.3804177030|

The switch is for paired endpoint MSE in the constructed independent-Gaussian family and nonnegative alpha; no analogous MAE threshold is asserted. Original fine versus coarsened histories remain nested. The non-nested comparison concerns observations after different nonlinear adapters. Haar/averaging, the max identity and standard-filter reference are not claimed new methods.

All30 synthetic reports, checkpoint/trace hashes, source-selected minima, test-release bindings and prediction hashes passed before latent-test values were loaded. Independent truth was reconstructed from the stored100,000 latent rows and each serialized row index. All30 test MSE/MAE/mean-MSE/detail-MSE values reproduce producer scores within2×10⁻¹². Three training seeds and fixed-test iid Monte Carlo error remain separate; finite neural scores are not claimed Bayes-optimal.

|alpha|seed|neural max−average MSE|neural max−average MAE|population MSE contrast|
|---:|---:|---:|---:|---:|
|0|20261001|0.2588994425|0.1302270671|0.2597310210|
|0|20261002|0.2594064588|0.1358863845|0.2597310210|
|0|20261003|0.2590927227|0.1328907328|0.2597310210|
|1|20261001|−0.3647662835|−0.1807116573|−0.3633802276|
|1|20261002|−0.3653311288|−0.1920880037|−0.3633802276|
|1|20261003|−0.3646517329|−0.1888175344|−0.3633802276|

The active synthetic audit SHA256 is `614e986e212da9bcd2f2d6d0768c8a7c58e330e0519e3d3a1d6f820ee8646cf1`. All adapters, unfavorable cells and per-seed outcomes are saved in the independent CSVs.

## Empirical scoring definition, fixed before reading its outcomes

Each customer's normalization is its independently reconstructed prior672-provider-slot observed mean, computed in float64 and floored at1kW. Calibration support is24–30June2013; confirmation support is25–31December2013. Complete future pairs require both original endpoints observed, with one common mask for every model. Physical endpoint MAE, fixed-scale NMAE, endpoint MSE/RMSE and pair mean/detail MSE are all computed. Pair mean/detail MSE sum to endpoint MSE. Forecasts are not clipped.

For each customer, total error is divided by its total available pairs before equal-customer averaging. Daily summaries do not replace this period-level aggregation. Three fitted-seed errors are averaged before a matched customer×Monday-calendar-week bootstrap with5000 resamples and seed13320261002. Every condition uses the same resampled customers/weeks. Intervals are conditional on these three fits and this block-resampling scheme; they do not integrate arbitrary new-training uncertainty, guarantee nominal coverage under nonstationarity, or establish a universal pooling rule. Seed-specific contrasts are saved separately. The primary contrast is(fine−coarse) under max minus(fine−coarse) under average. All fixed secondary contrasts and all five metrics are retained; secondary intervals are not multiplicity-adjusted.

Calibration is16 fixed customers/184 daily origins; confirmation is32 fixed customers/365 daily origins. Advancement to confirmation is explicitly independent of calibration direction or significance. Frozen source checkpoints are reused without fitting or reselection. Hash checks for the full prediction family and three simple references precede numeric truth loading. The scorer separately reconstructs all simple-reference predictions from causal fine histories, rather than treating their serialized labels as sufficient evidence.

## Delivery status

### Calibration — PASS, limited empirical support

All36 neural and three no-fit forecast hashes/reports passed before truth was loaded. Serialized customer IDs,184 dates, native origin indices and fixed support scales match independent reconstruction. All simple-reference arrays equal independent causal-history predictions exactly. There are16 fixed customers,141,264 common complete pairs and27 Monday-calendar-week blocks. All16 are scoreable and retained. Calibration prediction completion SHA256 is `190a3546097f38eaf78c44755c283b7352b51e6bf66560de92223eadd985af51`; independent calibration audit SHA256 is `25de86d28ae6a180a2988315f17d871b2cdf9e6adfc83d5bb2a16ffb49e86c9c`.

|adapter|fine NMAE|coarse NMAE|fine MAE kW|coarse MAE kW|
|---|---:|---:|---:|---:|
|direct|0.1048048312|0.1035118518|18.04787055|17.82180240|
|max repeat|0.1046994392|0.1047632569|17.89333805|18.02054613|
|average repeat|0.1056931883|0.1043196516|18.10469934|17.98819522|
|absolute Haar|0.1043636658|0.1040425002|17.94265991|17.95598225|
|signed Haar|0.1041471636|0.1048336262|17.86302140|17.96788201|
|standard-filter max reference|0.1054481042|0.1038050750|18.07906751|17.83370192|

Primary NMAE interaction is−0.001437354393, with conditional95% interval[−0.003245576004,+0.000106546875]. Seed-specific contrasts are−0.000378082885,−0.000494516231 and−0.003439464063. The common sign does not erase the sampling interval or the much larger third-seed contrast.

|fine-history diagnostic NMAE contrast|three-seed mean|conditional95% interval|
|---|---:|---|
|max−absolute|0.000335773427|[−0.000451643686,0.001309979778]|
|absolute−signed|0.000216502122|[−0.000676587155,0.001179776810]|
|signed−direct|−0.000657667558|[−0.001942794771,0.000487435861]|
|standard-filter max−pair max|0.000748665061|[−0.000328627541,0.002034069822]|

All four intervals cross zero. The constructed Gaussian information hierarchy is not claimed demonstrated by this calibration table. The coarse-history diagnostics and all physical/MSE contrasts remain in the full CSVs. Simple-reference NMAE/MAE kW are persistence0.3619289752/72.27045197, daily naive0.1192316546/20.17711513 and weekly naive0.1271930648/22.10273725. Neural cells outperform these quality references on the reported macro MAE/NMAE here; this does not establish general SOTA.

### Confirmation — PASS, bounded generalization evidence

All36 neural predictions and three references completed and were independently gated before truth loading. The same source checkpoints were reused; no fitting, checkpoint reselection or calibration-dependent subset was introduced. There are32 fixed customers,365 dates,560,448 common complete pairs and53 Monday-calendar-week blocks. All32 are scoreable and retained. Confirmation prediction completion SHA256 is `c2d9eeca1872b7c08fae5bc5d0c327556ecc339589c39c2ced721820ac1e67e7`; independent confirmation audit SHA256 is `bfc814c508252d0036cecdd26c1a9fbedfa8d85f5757c6dd8f857516997e8239`.

|adapter|fine NMAE|coarse NMAE|fine MAE kW|coarse MAE kW|
|---|---:|---:|---:|---:|
|direct|0.1432839431|0.1396624184|29.06038540|28.62475156|
|max repeat|0.1414877529|0.1410701180|28.85907317|28.89227844|
|average repeat|0.1420333535|0.1401730871|29.07589042|28.67850457|
|absolute Haar|0.1405016480|0.1393295709|28.87450363|28.52374634|
|signed Haar|0.1410687414|0.1401250891|28.90825076|28.74776179|
|standard-filter max reference|0.1423604374|0.1411633105|29.11073555|28.79539456|

The primary NMAE interaction is−0.001442631511, with conditional95% interval[−0.002970658063,+0.000189755143]. Individual seeds give−0.001292313243,−0.000361913551 and−0.002673667739. Both real-period primary intervals include zero despite six negative seed contrasts overall; the two periods must not be pooled as six independent inferential replicates.

|fine-history diagnostic NMAE contrast|three-seed mean|conditional95% interval|
|---|---:|---|
|max−absolute|0.000986104889|[−0.000562095476,0.003204709177]|
|absolute−signed|−0.000567093368|[−0.001650807047,0.000483883328]|
|signed−direct|−0.002215201742|[−0.005307566375,0.000041477885]|
|standard-filter max−pair max|0.000872684522|[−0.000721519065,0.002490563395]|

All fine diagnostic intervals include zero. Coarse-history secondary NMAE contrasts include max−absolute0.001740547157[0.000846891936,0.002900521090] and absolute−signed−0.000795518200[−0.001625388827,−0.000132813849]; these are unadjusted secondary intervals, and the latter direction does not match a universal signed-detail advantage. They cannot be promoted into new selected primary mechanisms.

Simple-reference NMAE/MAE kW are persistence0.4195322206/97.07290023, daily naive0.1454466389/29.64668926 and weekly naive0.1452522458/30.94338443. All reference forecasts were independently reconstructed exactly. Registered/activity counts, model-source bindings, per-household/per-date records, paired week sufficient statistics and every fixed metric/contrast are preserved.

### Necessary interpretation boundaries

- The Gaussian squared-loss reversal is verified constructively; the real-data history ranking is not a direct test of the Gaussian Bayes formula or a proven folding mechanism for the whole nonlinear network.
- Confirmation's six mean NMAEs favor coarse history. Physical MAE does not do so for max: fine28.85907317 versus coarse28.89227844. Calibration absolute-Haar physical MAE also favors fine even though its NMAE favors coarse. Statements about all adapters favoring coarse must identify NMAE.
- Calibration and confirmation have different customer groups and periods. This is joint customer/period transfer, not within-household temporal replication or a calendar-season effect.
- Noninvertible local adapter access, whole-network function classes, training-coordinate effects and finite source-trained model performance are different claims. Direct/signed initialization is matched; Adam is not rotation-invariant, and equal registration counts do not prove identical effective learning dynamics.
- Standard-filter max is a matched reference, not an author-benchmark BlurPool/DMANet reproduction. Neural-versus-naive quality comparisons do not establish SOTA. No new pooling or Haar algorithm is demonstrated.
- Both real primary intervals and all fine-history diagnostic intervals include zero. Common seed signs alone cannot establish a well-estimated nonzero population interaction. Gaussian MSE theory must not be rephrased as an MAE theorem.
- Confirmation fine history reduces detail MSE for all six adapters while raising each average fixed-scale NMAE. This is a metric-specific observation. Total MSE also decreases under fine history for max, signed Haar and standard-filter max; signed Haar and standard-filter max reduce mean MSE as well. A universal claim that detail gains are offset by mean-MSE losses would therefore be incorrect.

### All288 source residual responses — independent PASS

New script137 imports only frozen independent133, not fitting producers130/131/132. It restores all36 frozen source-selected checkpoints with the independently specified forward. Each of104 held-time source histories undergoes the same eight predeclared last-block preadapter interventions, preserving pair mean and multiplying complete-coverage signed half-difference by amplitude0,0.5,1 or2 and sign−1 or+1. Earlier blocks, original history normalization, last-value path and checkpoint weights remain fixed. No source-label accuracy, target outcome or new fit is computed.

All288 independently computed physical response RMS and mean-absolute values agree exactly with the producer CSV, with maximum discrepancy0kW. Eighteen exact original-amplitude sign-swap checks pass for max, average and absolute-Haar across two histories and three seeds. Seventy-two exact signed-direction comparisons across all four amplitudes also pass for those symmetric adapters. These checks concern the actual residual pair coordinates inside the final block; they do not assert invariance of the whole input history or prove optimizer dynamics.

Script137 SHA256 is `44a826920584ea682397a508999a3de95551fcbbf277522b781c214269ea5605`. The full independent CSV SHA256 is `f01cd6d9e7e2d009b4e29fe98e943312610ef0614f280ccb6b039c50f6a2409b`. Exact source/producer metadata and checkpoint bindings are recorded in `source_residual_response_audit.json`.

### Final audit status

**PASS / DELIVERY_GO for the numerical scientific-content identity and the bounded statements above.** All requested completed families were independently scored, with no producer metric/fitter import or fit rerun. This is not a claim that a manuscript has passed its final wording/citation audit or will be accepted by Neurocomputing. Any stronger empirical mechanism, universal ranking, new-architecture or SOTA conclusion would require revision.

Full output directory: `results/r6_independent_v1`. `oracle_checks.json`, `synthetic_audit.json`, `source_audit.json`, `uci_audit.json`, `confirmation_audit.json` and `source_residual_response_audit.json` bind their actual inputs and CSVs. The independently executed full extra root-authorization binding proof is `root_full_authorization_bindings.json` SHA256 `4f38095b79ba5657022c164dded9fa5cb9c54cfdcd0bff62212d082328b50e81`. The original preliminary engineering record and the completed pre-source-response manifest remain historical. `independent_audit_manifest_v2.json` is the active full audit inventory; frozen133 and final engineering identities remain unchanged.
