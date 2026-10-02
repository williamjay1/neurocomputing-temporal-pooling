# R6 implementation and confirmation addendum

Fixed before opening any newly reserved calibration or confirmation forecasts/outcomes, 2 October 2026. This supplements the immutable root protocol (SHA256 c9e76c448cc384c82b82c0b2f72afeca5f3cea4a25664657c1d30670cb372a04); it does not recast R4/R5 exploration as preregistration.

## Bound source implementation

Portuguese producer130 SHA256 6b434b9d806e12fef001fc7fd8ceaaa73ac5c8badf992de2e947f7d8d2f581e4; proposed protocol10978eefedd5b49ebda1c745d8ef3c1f85da6fb2458a9502427cebd0e9762a0a; preflight b9688246a3a51697931781fb594df0aa3814b1ebdecbde62efe5a52e36c674e2; source-cache manifest a240d3c4c3f240bb798006ea10cbc4eed38fcc201d9ea71e936ca58790ba325a. The cache has11,952 source training rows and104 source held-time rows. Input poisoning, partial counts, finite gradients, equal learned shapes and compensated initialization passed before source fitting. Real forecast clipping is none. Registered/forecast-active counts are240,036/196,356; neither is changed to favor a condition.

Synthetic producer132 is separately bound in experiment_lock.json to its exact code, preflight, all three fixed latent draws and root protocol. Thirty validation-selected fits precede test forecasting; all test forecasts precede scoring. Oracle uncertainty and fitted test Monte Carlo uncertainty remain distinct.

## Unconditional independent-period evaluation

The full2014 confirmation is advanced if source-family completion, code/data bindings and engineering verification pass. Advancement does not depend on calibration effect direction, statistical significance or the winning adapter. All32 previously reserved confirmation customers, all365 midnight forecast origins from1January through31December2014, all36 already fitted models and the three simple references are evaluated. No refitting, checkpoint reselection, model subset or outcome-dependent customer removal is permitted. The pre-period support scale is the observed customer mean over25–31December2013, floored at1kW. Input availability and target complete-pair masking are unchanged. Every forecast is created before its outcome is scored.

Primary and diagnostic contrasts, customer/week bootstrap and seed reporting remain identical to calibration. Results from both phases are retained. Inference is conditional on this source-trained family; two periods and customer groups do not make a universal representation rule. Confirmation implementation131 must be bound before confirmation access and may import the frozen130 model/representation but no outcome-scoring helper. The independent scorer imports neither fitting producer.

## Source-only residual response diagnostic

Use all104 fixed source held-time windows and all36 selected models, without changing checkpoints. Capture the final block's residual immediately before its adapter. For residual pairs whose two coverage entries equal1, preserve pair mean m and replace signed half-difference d by s*a*d, a in{0,.5,1,2}, s in{-1,+1}; leave incomplete pairs unchanged. Keep earlier blocks, normalization, last-value path and fitted parameters fixed. The reference is a=1,s=+1. Report physical forecast RMS and mean-absolute response. Any source-label error is descriptive and does not select settings or count as held-out confirmation.

Pair sign swaps should give exact local invariance for repeated max, repeated average and mean/absolute access; direct and mean/signed may respond. Overlapping stride-one maxima in the standard-filter reference need not be pair-swap invariant. This distinction is checked in the actual residual coordinate, whose pair order is reversed from chronological input order. The diagnostic measures the sensitivity of a trained last block. It does not identify optimization dynamics, global history invariance, a calendar cause or the full-network Bayes information ceiling.

## Reporting and decision boundary

Report all synthetic and Portuguese cells, both reserved phases, the three seed-specific contrasts and sampling intervals; retain adverse naive comparisons. The exact MSE risk condition and neural operational MAE evidence must not be conflated. No claim of a novel Haar representation, novel pooling, native DMANet reproduction, general SOTA or learned-retention success is authorized. The primary increment is a task-specific, provable prediction phenomenon together with its measured neural scope. R5's unsuccessful complete-cost upgrade remains available as a boundary result. Manuscript integration uses actual independently checked outcomes only.
