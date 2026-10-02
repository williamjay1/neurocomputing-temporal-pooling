# R6 core redesign: access-induced forecast-risk crossover

Date: 2 October 2026. Ownership: independent design/proof review and the new synthetic producer 132 only. Existing manuscripts, empirical records and other agents' code are unchanged.

Decision at the research-development stage: **CONDITIONAL**. The defensible increment is a precisely scoped forecast-risk crossover caused by nonnested observations after a fixed nonlinear access map, together with interventions distinguishing magnitude mixing from orientation loss. The increment is knowledge about neural access restrictions, not a new pooling architecture, codec, Blackwell theorem or publication guarantee. A manuscript decision requires the completed, independently checked new evidence.

## The research question and closest work

When fine histories contain coarse histories, why can a particular neural input operation prefer the coarse representation? The raw information sets are nested. The post-operation observations need not be: a pair maximum combines its mean with absolute detail and loses signed orientation, whereas coarsening before that operation exposes the mean alone. These restrictions can have different forecast utility for different future targets. A useful paper must show the exact distinction, test it under fair learned maps and disclose how model bypasses weaken it in real forecasts.

The near-neighbor primary-source review is recorded separately in `R6_NEIGHBOR_CORE_GATE_20261002.md`. It prevents generic novelty claims. [Boureau, Ponce and LeCun (ICML 2010)](https://icml.cc/Conferences/2010/papers/638.pdf) already analyze task/statistics-dependent max/average pooling. [Lee, Gallagher and Tu (AISTATS 2016)](https://proceedings.mlr.press/v51/lee16a.html) already propose mixed and gated pooling. [Zhang (ICML 2019)](https://proceedings.mlr.press/v97/zhang19a.html) treats nonlinear pooling followed by low-pass filtering and subsampling. [DMANet (ICML 2026)](https://proceedings.mlr.press/v306/zhou26aq.html) is a hard time-series neighbor about spectral folding and filtering. [WPMixer (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34156) already separates wavelet approximation/detail and reconstructs forecasts. [Blackwell (1951)](https://digicoll.lib.berkeley.edu/record/112749/files/math_s2_article-08.pdf) establishes the general decision-theoretic background.

The proposed narrow increment is the explicit future-pair risk threshold under nonlinear magnitude folding and a controlled distinction among mixed magnitude, separated magnitude and signed detail. Absence of this exact calculation from the checked papers is not proof of first discovery across all literature. The manuscript should use “we characterize” rather than “the first” or “a new anti-aliasing algorithm.”

## Scoped population proposition

Let independent C~N(0, sigma_C^2) and A~N(0, sigma_A^2), with positive variances. The fine pair is X=(C-A,C+A). Put mu=E|A|, D=|A|-mu, V=Var(D)=sigma_A^2(1-2/pi), and U=C+D. The actual pair maximum is U+mu; subtracting this known constant changes no available information.

Define a future pair Y=(M-B,M+B), where M=C+alpha D+epsilon_C and B=beta A+epsilon_A. The innovations are independent of each other and of C,A, have mean zero and finite variances. Write n=Var(epsilon_C)+Var(epsilon_A). The loss is the average squared error of the two endpoints. There is no clipping or nonnegative-load interpretation. In this constructive experiment, these are centered synthetic variables.

For an observation Z, unrestricted Bayes endpoint prediction is equivalent to predicting E[M|Z] and E[B|Z], since the exact squared-error identity is

    ((e_M-e_B)^2+(e_M+e_B)^2)/2 = e_M^2+e_B^2.

Let W=E Var(D|U). The risks under the declared access restrictions are

    R_C   = alpha^2 V + beta^2 sigma_A^2 + n,
    R_U   = (alpha-1)^2 W + beta^2 sigma_A^2 + n,
    R_abs = beta^2 sigma_A^2 + n,
    R_raw = R_signed = n.

Here C is repeated-average access; U is repeated-maximum access; “abs” observes (C,D); “raw/signed” observes the invertibly equivalent (C,A). These are population infima over all square-integrable predictors, not risks guaranteed to be reached by a finite ReLU network or Adam.

Proof. Given C, the independent centered D is unpredictable, and the symmetric A has conditional mean zero. Given U, C+alpha D=U+(alpha-1)D. Symmetry of A conditional on |A|, combined with C independence, gives E[A|U]=0. Thus the remaining conditional mean error has variance (alpha-1)^2 Var(D|U), while signed-detail error contributes beta^2 E[A^2]. Access to C,D removes deterministic mean uncertainty, but not A's sign. Access to C,A removes both. Independent innovations supply n in every case. No causal identification claim is involved.

For the nondegenerate Gaussian construction, the conditional density of |A| given C+|A| is a nondegenerate truncated normal on a continuous support, so W>0. Also Cov(D,U)=V and Var(U)=sigma_C^2+V. The best affine predictor of D from U has error V sigma_C^2/(sigma_C^2+V); the conditional expectation is at least as good. Therefore

    0 < W <= V sigma_C^2/(sigma_C^2+V) < V.

The maximum-minus-coarse risk gap is g(alpha)=(1-alpha)^2 W-alpha^2 V. It is strictly decreasing on [0,1], starts at W>0 and ends at -V<0. Its unique zero in that interval is

    alpha_star = sqrt(W)/(sqrt(W)+sqrt(V)).

Consequently 0<alpha_star<1/2, and the affine-prediction bound sharpens this to

    alpha_star <= sigma_C/(sigma_C+sqrt(sigma_C^2+V)) < 1/2.

At alpha=0, coarse access has lower risk than maximum access; at alpha=1, maximum access has lower risk. Raw access dominates both throughout. There is no contradiction with information monotonicity: C and U are two nonnested post-adapter observations of X. For all real alpha there is also a negative root, so the unique-switch statement must remain restricted to [0,1] (or nonnegative alpha). No corresponding MAE threshold is proved.

The risk formulas and affine bound extend beyond Gaussianity if C and A are independent, A is symmetric about zero, the required second moments exist, V>0 and W>0. C's mean is immaterial; a fixed translation changes no risk. W>0 is an essential restriction, not an automatic consequence of independence and nonzero variances: some discrete/separable supports allow U to identify D exactly. Gaussianity supplies nondegeneracy and an explicit posterior for numerical W. This extension is a short corollary of standard projection facts, not a new general information-order theorem.

## Numerical oracle and implementation identity

For Z=C+|A|, set S=sigma_C^2+sigma_A^2, tau^2=sigma_C^2 sigma_A^2/S and t=Z sigma_A/(sigma_C sqrt(S)). With lambda(t)=phi(t)/Phi(t),

    Var(|A||Z) = tau^2 [1-t lambda(t)-lambda(t)^2].

The producer evaluates the inverse Mills ratio using SciPy log_ndtr, integrates over independent positive half-normal and Gaussian variables at 96 and 192 nodes, and cross-checks a separate one-dimensional skew-normal-density integral. W has no claimed closed form. A 12-standard-deviation integration truncation and numerical roundoff guard are disclosed; agreement is a numerical precision check rather than a certified global quadrature-error theorem.

Source-only preflight yielded the following population calculations, before any formal synthetic test score:

| sigma_A (sigma_C=1) | V | W (192 nodes) | alpha_star |
|---|---:|---:|---:|
| 0.25 | 0.0227112642270262 | 0.0222015890657820 | 0.497162885823214 |
| 1 | 0.363380227632419 | 0.259731020962560 | 0.458122935265025 |
| 2 | 1.45352091052967 | 0.547954781386044 | 0.380417702960402 |

All two-resolution and independent-integral differences were below 3e-14. These numbers are mathematical oracle calculations for the declared synthetic family, not empirical real-data estimates.

The stable source is `scripts/132_r6_oracle_and_synthetic_v1.py`, SHA256 a22028012836a4de5a710844f1075834be8917214a81e5339e6f4c9102871a5a. The root protocol is `docs/R6_RESEARCH_PROTOCOL_20261002.md`, SHA256 c9e76c448cc384c82b82c0b2f72afeca5f3cea4a25664657c1d30670cb372a04. The experiment lock SHA256 is aed011160315340efbaeef8caa69b7baa412e6900920269fa6b9085f632a9855. It binds code, protocol, preflight, generation settings and all three draw files before formal fitting.

## Constructive neural check and real-world scope

The fixed synthetic family has five two-coordinate adapters, two scenarios alpha=0/1 with beta=.5, and three seeds: 30 models. All use 4,482 learned parameters, two 64-unit ReLU hidden layers, 2,000 Adam updates, batch 256 and the same minibatch rows per seed. Training/validation/test counts are 20,000/10,000/100,000 with independent locked generation seeds. Validation every 200 updates chooses a checkpoint by MSE, with earliest exact tie. No test selection, clipping or resource reduction occurs. All cells and seeds must remain reported, including adverse MAE or approximation gaps.

Direct and signed-Haar inputs have the same information and representable functions. A rotated first-layer weight matches their initial functions to float32 precision (preflight maximum discrepancy 1.1921e-7). Adam is not rotationally invariant, so subsequent differences can reflect finite optimization. The mean/absolute variant uses the same canonical rotation and deliberately removes orientation. Repeated maximum and average have the same learned-map dimensions and parameter count but lower input rank; equal parameters do not imply equal accessible function classes.

At the initial training handoff, all 30 formal source-selected checkpoints were complete, using about 53.46 seconds of summed fit wall time. Training completion SHA256 is ed7b8519f4a72fcdd8b63cc3abf888928f58f64d17f328503ff540f7b73b1a9a, status ALL_30_SOURCE_SELECTED_CHECKPOINTS_COMPLETE_TEST_UNREAD. At that handoff no test prediction or test score had been run. Root then released the test phase with a separately bound gate, and all 30 forecast files were completed before scoring. Agent 133 supplies the separate no-producer-import numerical audit.

The subsequent producer scoring completion SHA256 is 790210034eba6295f616835d95f79baf6a529e66d2651c87869348d9849333c6. It binds all 30 test forecasts and all output tables. The three-fit mean test pair MSEs are:

| alpha (beta=.5) | direct | maximum | average | mean/absolute | mean/signed |
|---|---:|---:|---:|---:|---:|
| 0 | 0.020260638 | 0.530259015 | 0.271126140 | 0.271239634 | 0.020256352 |
| 1 | 0.020375112 | 0.271151765 | 0.636068147 | 0.271329291 | 0.020348915 |

Maximum minus average MSE is positive for all three seeds at alpha=0 (0.258899–0.259406) and negative for all three at alpha=1 (-0.365331 to -0.364652). The population gaps are +0.259731021 and -0.363380228. Mean/absolute minus mean/signed is about 0.251 in either scenario, near but not identical to the population missing-orientation term 0.25. Direct/signed differences are only several times 1e-5 and do not justify an information or new-architecture advantage. These are scores on one common 100,000-pair synthetic test draw per scenario; the three fitted-seed range is not a confidence interval. Finite fitted risk is not declared equal to the population infimum. At this update the independent 133 audit remains pending.

The matched UCI experiment belongs to worker 130. It evaluates final residual adapters in a full N-HiTS model, with fixed learned tensor shapes and source-only fitting. Earlier backcasts, residual values, per-window normalization and naive/last-level paths can preserve information outside the replaced final adapter. Its results therefore test a restricted-access intervention in a whole forecasting map; they do not instantiate an exact Gaussian bottleneck or identify the cause of prior London temporal reversals. MAE remains the declared operational real-data outcome; mean/detail MSE diagnoses outcomes without substituting for it.

## What would justify or stop this core

Proceed if the exact proposition is independently reproduced, the controlled neural results visibly recover both opposite scenarios without claiming Bayes attainment, and real matched contrasts distinguish magnitude mixing and orientation loss with honest uncertainty and meaningful effects. An adverse real result is retained as a boundary rather than hidden or addressed by selecting a new architecture after test scores.

Stop algorithm/SOTA packaging if the claimed novelty reduces to average pooling, a mixed max/average gate or an invertible Haar rotation; if neural effects consist only of initialization advantages; or if real bypasses make the proposed access restriction practically irrelevant. A synthetic construction alone proves possibility, not prevalence, deployment benefit or a sufficient empirical contribution for Neurocomputing. Do not use this theorem to rehabilitate the unsuccessful R5 retention gate, claim Fourier anti-aliasing, identify concept drift, explain all historical market effects or promise publication. The next decision must assess the actual new calibration/confirmation evidence and the closest work, rather than inflate the mathematical calculation into an architecture claim.
