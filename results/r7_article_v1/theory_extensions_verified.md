# Verified R7 theory extensions

Status: actual exact finite-support enumeration, completed after review. These constructed mathematical checks are not the original synthetic neural fits, a new preregistered experiment, or independent empirical confirmation. The distributions and two nonlinear maps were fixed in `fixed_definitions.json` before the saved final curve calculations; they were chosen after exploratory proof work. No new neural fits, raw downloads or real-data endpoint changes are involved.

## Main-text-ready loss geometry and counterexample

For a future pair \(Y=(M-B,M+B)\), put \(e_M=\hat M-M\) and \(e_B=\hat B-B\). Average squared endpoint loss is \(e_M^2+e_B^2\), whereas average absolute endpoint loss is \(\max(|e_M|,|e_B|)\). Conditional means minimize the first loss; conditional medians of each endpoint minimize the second. The squared-loss projection ranking therefore cannot simply be transferred to MAE.

A finite-support counterexample preserves independent level and sign-symmetric detail. Let \(C\) take \(-1/10,9/10\) with probabilities \(9/10,1/10\). Independently, let \(|A|\) take \(1,2\) with the same probabilities and an independent fair sign. Define \(D=|A|-11/10\), \(U=C+D\), and the same future target \(M=C+(9/20)D,\ B=A/10\), without innovations. Exact Bayes endpoint MSE is 0.031225 under \(C\) and 0.0266125 under \(U\); exact Bayes endpoint MAE is 0.125 and 0.1325, respectively. Thus MSE prefers fine maximum access and MAE prefers coarse access for the same target. These are optimal risks under two nonnested observations: \(E\operatorname{Var}(U\mid C)=9/100\) and \(E\operatorname{Var}(C\mid U)=9/200\). Full signed raw access has zero risk under either loss. The example does not reverse the value of nested raw information and does not establish why any specific real-data contrast is small.

| Observation | Bayes endpoint MSE | Bayes endpoint MAE |
|---|---:|---:|
| Coarse \(C\) | 0.031225 | 0.125 |
| Fine maximum \(U\) | 0.0266125 | 0.1325 |
| Separated \((C,D)\) | 0.013 | 0.11 |
| Signed raw \((C,A)\) | 0 | 0 |

## Main-text-ready nonlinear boundary

With \(M=C+\alpha g(D)+\epsilon_C\), \(B=\beta A+\epsilon_A\), independent \(C\) and sign-symmetric \(A\), and square-integrable \(g(D)\), define \(W_D=E\operatorname{Var}(D\mid U)\), \(K=E\operatorname{Cov}(D,g(D)\mid U)\), and \(H_g=\operatorname{Var}(E[g(D)\mid U])\). Under the same independent centered innovations as the main construction,

\[
R_U-R_C=W_D-2\alpha K-\alpha^2 H_g.
\]

When \(W_D>0\), the threshold depends on the conditional covariance and the predictability of the nonlinear future term, rather than on \(W_D,V_D\) alone. For \(H_g>0\), its sole positive root is \(\alpha_*=(\sqrt{K^2+H_gW_D}-K)/H_g\), which need not lie in \([0,1]\). If \(H_g=0\), independence forces \(K=0\), so the gap remains \(W_D\) and no switch occurs. A fixed-map amplitude family has at most one positive switch, so this extension supplies a changed condition rather than a claim of multiple crossings.

In a fully enumerated nonlinear example, take independent uniform \(C,D\in\{-1,0,1\}\), \(|A|=D+2\), an independent fair sign, \(B=A/2\), and no innovations. With \(g(D)=D+(D^2-2/3)/2\), the gap is \(1/3-2\alpha/3-37\alpha^2/108\), switching at 0.412541148. With the folded map \(g(D)=D^2-2/3\), it is \(1/3-\alpha^2/27\), strictly positive throughout \([0,1]\); its first switch is at \(\alpha=3\). At \(\alpha=1\), the corresponding \((R_C,R_U)\) are \((17/9,131/108)\) and \((25/18,91/54)\). Nonlinearity can preserve the crossover, but future use of magnitude need not favor maximum access within the original parameter domain.

## Supplement proof and exact audit detail

### Loss identity and optimality

Let \(x=e_M\), \(y=e_B\). Expansion gives \(((x-y)^2+(x+y)^2)/2=x^2+y^2\). To prove \((|x-y|+|x+y|)/2=\max(|x|,|y|)\), assume without loss of generality \(|x|\ge|y|\). Both \(x-y\) and \(x+y\) have the sign of \(x\), allowing equality at zero, and their absolute values sum to \(2|x|\). The transformation between endpoint forecasts and level/detail forecasts is invertible, so arbitrary endpoint conditional medians are globally optimal for paired MAE. This does not generally mean separate conditional medians of \(M\) and \(B\) minimize their maximum error.

For the loss counterexample, there are eight exact states. Given \(C=c\), the conditional endpoint distributions are identical and have values \(c-29/200,c+11/200,c+41/200,c+121/200\), with weights \(9/20,9/20,1/20,1/20\). A common endpoint median is \(c+11/200\), yielding expected absolute risk \(1/8\). The conditional squared risk is \((9/20)^2(9/100)+(1/10)^2(13/10)=1249/40000\).

The maximum observation has values \(-1/5,4/5,9/5\), with masses \(81/100,18/100,1/100\). The outer values identify both \(C\) and \(D\); their conditional MAE risks are \(1/10\) and \(1/5\). At the middle value, the conditional endpoints are \(21/200,101/200,151/200,191/200\), equally weighted. Every endpoint median in \([101/200,151/200]\) is optimal, giving conditional risk \(11/40\). Thus total MAE is \((81/100)(1/10)+(18/100)(11/40)+(1/100)(1/5)=53/400\). The conditional MSE is \((11/20)^2(9/200)+(1/10)^2(13/10)=2129/80000\). The exact gaps are \(-369/80000\) under MSE and \(3/400\) under MAE.

An independent linear program minimizes \(\sum_i p_i z_i\), with free \(\hat M,\hat B\), subject to \(z_i\ge|M_i-\hat M|\) and \(z_i\ge|B_i-\hat B|\) at every conditional state. It agrees with endpoint-median risks for all four loss-case accesses and for the two nonlinear maps at five prespecified curve anchors under \(C,U\): 24 LP checks in total. All exact identities and risks use rational arithmetic; the LP is only a floating-point independent optimality check, agreeing within \(10^{-10}\).

### General nonlinear projection proof

Assume \(C\perp A\), \(A\) is sign-symmetric, \(E C^2,E A^2,E g(D)^2<\infty\), \(D=|A|-E|A|\), \(U=C+D\), and independent centered innovations with variances \(v_C,v_A\). The joint law is invariant under \(A\mapsto-A\) at fixed \(C,|A|\), so \(E[A\mid U]=E[A\mid C]=0\). Consequently the future-detail risk \(\beta^2E A^2+v_A\) is common to both observations. Under \(C\), independence gives level risk \(\alpha^2\operatorname{Var}(g(D))+v_C\). Under \(U\), write \(M=U+\alpha g(D)-D+\epsilon_C\); its level risk is

\[
W_D-2\alpha K+\alpha^2E\operatorname{Var}(g(D)\mid U)+v_C.
\]

Subtract and apply total variance to obtain the stated gap. This proof imposes no monotonicity of \(g\); only the amplitude \(\alpha\) varies while the map and support stay fixed. With \(W_D>0,H_g>0\), the product of the two quadratic roots is \(-W_D/H_g<0\), so exactly one root is positive. If \(H_g=0\), then \(E[g(D)\mid U]\) is constant. Independence gives \(\operatorname{Cov}(g(D),D)=\operatorname{Cov}(g(D),U)=0\); total covariance then yields \(K=0\). The gap is constant in this attainable degenerate branch. Hence multiple positive crossings are impossible in this fixed-map amplitude family; changing \(g\), introducing nonlinear coupling in \(\alpha\), or adding sign-asymmetric terms would be a different family.

For the fixed nonlinear support there are 18 states and five \(U\) groups. Symmetry gives \(E\operatorname{Cov}(D,D^2-2/3\mid U)=0\). Direct enumeration gives \(V_D=2/3,W_D=1/3\), and for \(h(D)=D^2-2/3\), \(\operatorname{Var}(h)=2/9\), \(\operatorname{Var}(E[h\mid U])=1/27\). Thus \(g=D+h/2\) has \(\operatorname{Var}(g)=13/18\), \(K=1/3,H_g=37/108\), while \(g=h\) has \(K=0,H_g=1/27\). The common future-detail MSE is \((1/2)^2 E A^2=7/6\). Direct conditional-mean risks equal the polynomial at each of 101 rational \(\alpha\) values per map, giving 202 complete rows. Curves also contain separately optimized MAE risks, which are not used to derive the MSE theorem.

## Short executable design diagnostic

1. Fix the native forecast endpoints, primary loss, history span, source-training split and source-validation selection rule. Evaluate fine and coarsened histories on the same observed targets and weights; retain all declared conditions. Select checkpoints using source validation, finish all fits, then score external data. Report endpoint MSE and paired MAE separately when their distinction is part of the question.
2. Compare fine maximum \(C+|A|\) with separate \((C,|A|)\) access to assess level/magnitude mixing; compare \((C,|A|)\) with \((C,A)\) to assess orientation; compare invertible signed \((C,A)\) with direct pairs after compensating the first affine to assess finite fitting and conditioning. Use common learned shapes, batches, seeds and validation criteria. Nonnested maximum/average risk differences concern pooling preference; parameter matching alone does not equate their function classes.
3. State the scope of the observation. For a final residual-block intervention, inspect earlier forecasts/backcasts, last-value additions and per-window normalization. Their paths can carry information outside the transformed block. Verify local swap and constant/pair perturbation responses, and label the result local. A separate complete-input experiment using source-fitted fixed normalization and no raw bypass can examine the whole transformed observation. The different architecture and normalization make its change a scope sensitivity, not identification of which N-HiTS path explains the difference.
4. On the fixed endpoint, report the declared fine/coarse interactions and coordinate diagnostics with paired customer/time uncertainty and the planned multiplicity scope. Interpret a signed/direct difference as finite-fit sensitivity, an uncertain external contrast as inconclusive, and a detail-MSE gain without endpoint-MAE improvement as loss-dependent utility. Neither an oracle threshold nor a secondary metric is an operation-selection rule for the external cohort. Any future selected deployment needs a fresh evaluation following source-only choice.

The diagnostic yields three distinguishable questions—mixing, orientation and invertible finite-fit effects—and makes the whole-network bypass scope explicit. It does not estimate a real-data \(\alpha\), infer a universal retention rule, attribute the current null intervals to MAE, or select the best operation on already scored customers.

## Artifacts

`scripts/140_r7_nonlinear_and_loss_boundaries_v1.py` produces the definition record, support files, `results/r7_theory_v1/nonlinear_risk_curves.csv`, and exact `verified_results.json`. The JSON preserves fractions, float convenience values, conditional optimal decisions, LP audits, and the script hash. `theory_extensions_verified.json` is the integration copy. All computation and outputs are on D; no original manuscript, existing results or raw material were edited.
