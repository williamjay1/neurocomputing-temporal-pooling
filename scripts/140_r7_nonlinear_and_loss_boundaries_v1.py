"""Exact post-review loss and nonlinear boundaries; no neural fitting or raw IO.

Every finite state uses fractions.  Bayes MSE uses conditional means; Bayes
paired MAE uses endpoint conditional medians, checked by a separately stated
linear program in level/detail coordinates.  This mathematical extension was
designed after reviewer feedback and is not an originally preregistered result.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r7_theory_v1"
ARTICLE = ROOT / "results" / "r7_article_v1"


def enc(x):
    if isinstance(x, F):
        return {"fraction": str(x), "value": float(x)}
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [enc(v) for v in x]
    return x


def save_json(path, data):
    path.write_text(json.dumps(enc(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def states(cs, ts):
    # C independent of magnitude T and independent fair sign S; A=S*T.
    et = sum(t * p for t, p in ts)
    out = []
    for c, pc in cs:
        for t, pt in ts:
            for s in (-1, 1):
                a = s * t
                d = t - et
                out.append(dict(C=c, A=a, D=d, U=c+d, p=pc*pt/F(2)))
    assert sum(x["p"] for x in out) == 1
    return out


def groups(rows, access):
    gg = defaultdict(list)
    for r in rows:
        if access == "raw":
            key = (r["C"], r["A"])
        elif access == "magnitude":
            key = (r["C"], r["D"])
        else:
            key = r[access]
        gg[key].append(r)
    return gg


def average(rows, name):
    mass = sum(r["p"] for r in rows)
    return sum(r["p"] * r[name] for r in rows) / mass


def median(rows, name):
    # First support point at which the conditional CDF reaches 1/2.
    accum = F(0)
    mass = sum(r["p"] for r in rows)
    for r in sorted(rows, key=lambda q: q[name]):
        accum += r["p"]
        if 2*accum >= mass:
            return r[name]
    raise AssertionError("empty conditional support")


def target(base, alpha, beta, g):
    rows = []
    for r0 in base:
        r = dict(r0)
        r["g"] = g(r["D"])
        r["M"] = r["C"] + alpha*r["g"]
        r["B"] = beta*r["A"]
        r["Y1"], r["Y2"] = r["M"]-r["B"], r["M"]+r["B"]
        rows.append(r)
    return rows


def oracle(rows, access, loss):
    total = F(0)
    decisions = []
    for key, rr in groups(rows, access).items():
        if loss == "mse":
            m, b = average(rr, "M"), average(rr, "B")
        elif loss == "mae":
            y1, y2 = median(rr, "Y1"), median(rr, "Y2")
            m, b = (y1+y2)/2, (y2-y1)/2
        else:
            raise ValueError(loss)
        conditional = F(0)
        for r in rr:
            em, eb = m-r["M"], b-r["B"]
            l2 = ((m-b-r["Y1"])**2+(m+b-r["Y2"])**2)/2
            l1 = (abs(m-b-r["Y1"])+abs(m+b-r["Y2"]))/2
            assert l2 == em**2+eb**2
            assert l1 == max(abs(em), abs(eb))
            conditional += r["p"] * (l2 if loss == "mse" else l1)
        total += conditional
        decisions.append(dict(observation=str(key), mass=sum(r["p"] for r in rr),
                              hat_M=m, hat_B=b, unconditional_risk_contribution=conditional))
    return total, decisions


def lp_mae(rows, access):
    # Independent LP: min E z_i subject to z_i>=|M_i-m|,|B_i-b|.
    risk = 0.0
    largest_constraint_residual = 0.0
    for _, rr in groups(rows, access).items():
        n = len(rr)
        obj = np.array([0.0, 0.0]+[float(r["p"]) for r in rr])
        lhs, rhs = [], []
        for i, r in enumerate(rr):
            for j, name in ((0, "M"), (1, "B")):
                for sign in (-1, 1):
                    row = np.zeros(n+2)
                    row[j], row[i+2] = sign, -1
                    lhs.append(row)
                    rhs.append(sign*float(r[name]))
        aa, bb = np.array(lhs), np.array(rhs)
        fit = linprog(obj, A_ub=aa, b_ub=bb,
                      bounds=[(None, None), (None, None)]+[(0, None)]*n,
                      method="highs")
        assert fit.success, fit.message
        risk += float(fit.fun)
        largest_constraint_residual = max(largest_constraint_residual,
                                          float(np.max(aa@fit.x-bb)))
    return dict(risk=risk, max_constraint_violation=largest_constraint_residual)


def moments(base, g):
    rr = [dict(r, g=g(r["D"])) for r in base]
    ed, eg = average(rr, "D"), average(rr, "g")
    vd = sum(r["p"]*(r["D"]-ed)**2 for r in rr)
    vg = sum(r["p"]*(r["g"]-eg)**2 for r in rr)
    wd, wg, k, hg = F(0), F(0), F(0), F(0)
    for _, sub in groups(rr, "U").items():
        mass = sum(r["p"] for r in sub)
        md, mg = average(sub, "D"), average(sub, "g")
        wd += sum(r["p"]*(r["D"]-md)**2 for r in sub)
        wg += sum(r["p"]*(r["g"]-mg)**2 for r in sub)
        k += sum(r["p"]*(r["D"]-md)*(r["g"]-mg) for r in sub)
        hg += mass*(mg-eg)**2
    assert wg+hg == vg
    return dict(V_D=vd, V_g=vg, W_D=wd, W_g=wg, K=k, H_g=hg)


def root(mom):
    w, k, h = map(float, (mom["W_D"], mom["K"], mom["H_g"]))
    if h > 0:
        return (-k+math.sqrt(k*k+h*w))/h
    if k > 0:
        return w/(2*k)
    return None


def main():
    assert ROOT.drive.upper() == "D:", "Computation must be on D:"
    OUT.mkdir(parents=True, exist_ok=True)
    ARTICLE.mkdir(parents=True, exist_ok=True)
    protocol = dict(status="post-review exact mathematical extension",
                    definition_recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                    selection_disclosure="Definitions were chosen after review and exploratory proof search. "
                    "They are fixed here before the saved final risk curves, not claimed as an original "
                    "preregistration or independent empirical confirmation.",
                    loss_case=dict(C="[-1/10,9/10] with probabilities [9/10,1/10]",
                                   T="[1,2] with probabilities [9/10,1/10]",
                                   independence="C,T,S mutually independent; fair S in {-1,+1}; A=S*T",
                                   D="T-11/10", U="C+D", M="C+(9/20)*D", B="A/10",
                                   innovations="zero"),
                    nonlinear_cases=dict(C="uniform {-1,0,1}", T="uniform {1,2,3}",
                                         independence="C,T,S mutually independent; fair S; A=S*T; D=T-2",
                                         U="C+D", M="C+alpha*g(D)", B="A/2", alpha="[0,1]",
                                         persistence="g(D)=D+(D^2-2/3)/2",
                                         boundary="g(D)=D^2-2/3", innovations="zero"),
                    computations="Exact finite-support enumeration; no training, test selection, raw access or download.")
    protocol_path = OUT / "fixed_definitions.json"
    if protocol_path.exists():
        prior_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        assert {k:v for k,v in prior_protocol.items() if k!="definition_recorded_at_utc"} == \
               {k:v for k,v in protocol.items() if k!="definition_recorded_at_utc"}, \
               "Definitions changed: use a new version rather than overwrite an old protocol."
        protocol = prior_protocol
    else:
        save_json(protocol_path, protocol)

    loss_base = states([(F(-1,10),F(9,10)),(F(9,10),F(1,10))],
                       [(F(1),F(9,10)),(F(2),F(1,10))])
    loss_rows = target(loss_base, F(9,20), F(1,10), lambda d: d)
    loss_results, lp_audits = {}, {}
    for access in ("C", "U", "magnitude", "raw"):
        loss_results[access] = {}
        for loss in ("mse", "mae"):
            risk, pred = oracle(loss_rows, access, loss)
            loss_results[access][loss] = dict(risk=risk, decisions=pred)
        lp = lp_mae(loss_rows, access)
        assert abs(lp["risk"]-float(loss_results[access]["mae"]["risk"])) < 1e-11
        assert lp["max_constraint_violation"] < 1e-10
        lp_audits[access] = lp
    assert loss_results["C"]["mse"]["risk"] == F(1249,40000)
    assert loss_results["U"]["mse"]["risk"] == F(2129,80000)
    assert loss_results["C"]["mae"]["risk"] == F(1,8)
    assert loss_results["U"]["mae"]["risk"] == F(53,400)
    assert loss_results["U"]["mse"]["risk"] < loss_results["C"]["mse"]["risk"]
    assert loss_results["U"]["mae"]["risk"] > loss_results["C"]["mae"]["risk"]
    assert loss_results["raw"]["mse"]["risk"] == loss_results["raw"]["mae"]["risk"] == 0
    # Nonnesting is essential and verified rather than asserted by appearance.
    mom_linear = moments(loss_base, lambda d: d)
    assert mom_linear["V_D"] == F(9,100) and mom_linear["W_D"] == F(9,200)
    save_json(OUT / "loss_case_support.json", loss_rows)

    nonlinear_base = states([(F(-1),F(1,3)),(F(0),F(1,3)),(F(1),F(1,3))],
                           [(F(1),F(1,3)),(F(2),F(1,3)),(F(3),F(1,3))])
    cases = {"nonlinear_persistence": lambda d: d+(d*d-F(2,3))/2,
             "nonlinear_boundary": lambda d: d*d-F(2,3)}
    nonlinear = {}
    csv_rows = []
    max_mse_discrepancy = 0.0
    for name, g in cases.items():
        mom = moments(nonlinear_base, g)
        anchors = {}
        for i in range(101):
            alpha = F(i,100)
            rr = target(nonlinear_base, alpha, F(1,2), g)
            risks = {a: {l: oracle(rr,a,l)[0] for l in ("mse","mae")}
                     for a in ("C","U","magnitude","raw")}
            gap = risks["U"]["mse"]-risks["C"]["mse"]
            analytical_gap = mom["W_D"]-2*alpha*mom["K"]-alpha*alpha*mom["H_g"]
            assert gap == analytical_gap
            assert risks["raw"]["mse"] == risks["raw"]["mae"] == 0
            assert all(risks["magnitude"][l] <= min(risks["C"][l],risks["U"][l])
                       for l in ("mse","mae"))
            csv_rows.append(dict(case=name, alpha=float(alpha),
                                 mse_C=float(risks["C"]["mse"]), mse_U=float(risks["U"]["mse"]),
                                 mse_gap=float(gap), mae_C=float(risks["C"]["mae"]),
                                 mae_U=float(risks["U"]["mae"]),
                                 mae_gap=float(risks["U"]["mae"]-risks["C"]["mae"]),
                                 exact_mse_gap=str(gap)))
            if i in (0,25,50,75,100):
                lp = {a: lp_mae(rr,a) for a in ("C","U")}
                for a in ("C","U"):
                    assert abs(lp[a]["risk"]-float(risks[a]["mae"])) < 1e-10
                anchors[str(alpha)] = dict(risks=risks, gap=gap, lp_audit=lp)
        nonlinear[name] = dict(moments=mom, positive_root=root(mom),
                               switches_on_declared_domain=int(root(mom) is not None and 0<root(mom)<1),
                               anchors=anchors)
    assert nonlinear["nonlinear_persistence"]["moments"]["W_D"] == F(1,3)
    assert nonlinear["nonlinear_persistence"]["moments"]["K"] == F(1,3)
    assert nonlinear["nonlinear_persistence"]["moments"]["H_g"] == F(37,108)
    assert nonlinear["nonlinear_boundary"]["moments"]["K"] == 0
    assert nonlinear["nonlinear_boundary"]["moments"]["H_g"] == F(1,27)
    assert nonlinear["nonlinear_boundary"]["positive_root"] == 3
    assert nonlinear["nonlinear_persistence"]["switches_on_declared_domain"] == 1
    assert nonlinear["nonlinear_boundary"]["switches_on_declared_domain"] == 0
    with (OUT / "nonlinear_risk_curves.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    save_json(OUT / "nonlinear_base_support.json", nonlinear_base)

    results = dict(status="verified exact finite-support oracle enumeration", protocol=protocol,
                   loss_case=dict(risks=loss_results, LP_audit=lp_audits,
                                  nonnesting=dict(E_Var_U_given_C=mom_linear["V_D"],
                                                 E_Var_C_given_U=mom_linear["W_D"]),
                                  MSE_gap=loss_results["U"]["mse"]["risk"]-loss_results["C"]["mse"]["risk"],
                                  MAE_gap=loss_results["U"]["mae"]["risk"]-loss_results["C"]["mae"]["risk"]),
                   nonlinear=nonlinear,
                   audit=dict(state_count_loss=len(loss_rows),state_count_nonlinear=len(nonlinear_base),
                              curve_rows=len(csv_rows),exact_formula_discrepancy=0,
                              MAE_LP_checks=24,geometry_identity="exact at all conditional forecasts",
                              nested_raw_risk="zero under both losses; no nested-information reversal",
                              finite_sample_or_neural_claim="none", noise="zero; adding independent common centered noise preserves MSE gaps, not automatically MAE gaps"),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    prior_path = OUT / "verified_results.json"
    if prior_path.exists():
        prior_results = json.loads(prior_path.read_text(encoding="utf-8"))
        assert {k:v for k,v in prior_results.items() if k!="script_sha256"} == \
               {k:v for k,v in enc(results).items() if k!="script_sha256"}, \
               "Computed risks changed: preserve the prior result and use a new version."
    save_json(prior_path, results)
    save_json(ARTICLE / "theory_extensions_verified.json", results)
    threshold = nonlinear["nonlinear_persistence"]["positive_root"]
    prose = PROSE.replace("{THRESHOLD}",f"{threshold:.9f}")
    (ARTICLE / "theory_extensions_verified.md").write_text(prose,encoding="utf-8")
    print(json.dumps(dict(status=results["status"],MSE_gap=float(results["loss_case"]["MSE_gap"]),
                          MAE_gap=float(results["loss_case"]["MAE_gap"]),
                          nonlinear_persistence_threshold=threshold,nonlinear_boundary_threshold=3,
                          curve_rows=len(csv_rows),MAE_LP_checks=24),indent=2))


PROSE = r"""# Verified R7 theory extensions

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

In a fully enumerated nonlinear example, take independent uniform \(C,D\in\{-1,0,1\}\), \(|A|=D+2\), an independent fair sign, \(B=A/2\), and no innovations. With \(g(D)=D+(D^2-2/3)/2\), the gap is \(1/3-2\alpha/3-37\alpha^2/108\), switching at {THRESHOLD}. With the folded map \(g(D)=D^2-2/3\), it is \(1/3-\alpha^2/27\), strictly positive throughout \([0,1]\); its first switch is at \(\alpha=3\). At \(\alpha=1\), the corresponding \((R_C,R_U)\) are \((17/9,131/108)\) and \((25/18,91/54)\). Nonlinearity can preserve the crossover, but future use of magnitude need not favor maximum access within the original parameter domain.

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
"""


if __name__ == "__main__":
    main()
