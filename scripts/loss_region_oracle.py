"""Exact loss-ordering region and independently checked finite-support oracles.

This is a post-review mathematical extension, not a preregistered discovery.
All output lives under this D: project. No raw input or old archive is modified.
Dependencies: Python standard library. Optional SciPy supplies separate LP checks.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "theory_region"


def encode(obj):
    if isinstance(obj, F):
        return {"fraction": str(obj), "value": float(obj)}
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def save_json(name, obj):
    (OUT / name).write_text(json.dumps(encode(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def support(alpha, beta, p=F(1, 10), p_level=None, centered=True, history_shift=F(0), noise=F(0)):
    """Eight noiseless states; 32 with independent coordinate Rademacher noise.

    Keeping centered=False fixes the original numerical history support and U map
    when probabilities change. centered=True is an equivalent known translation.
    history_shift moves only C's upper support, intentionally separating U collisions.
    """
    q = p if p_level is None else p_level
    center = p if centered else F(1, 10)
    rows = []
    for l, h, s in itertools.product((0, 1), (0, 1), (-1, 1)):
        c = F(l) - center + history_shift * l
        d = F(h) - center
        a = F(s) * (1 + h)
        mass = (q if l else 1-q) * (p if h else 1-p) / 2
        innovations = [(F(0), F(0), F(1))] if noise == 0 else [
            (sc * noise, sa * noise, F(1, 4))
            for sc, sa in itertools.product((-1, 1), repeat=2)
        ]
        for ec, ea, pn in innovations:
            m, b = c + alpha*d + ec, beta*a + ea
            rows.append(dict(C=c, D=d, A=a, U=c+d, M=m, B=b,
                             Y1=m-b, Y2=m+b, p=mass*pn))
    assert sum(r["p"] for r in rows) == 1
    return rows


def groups(rows, access):
    out = defaultdict(list)
    for row in rows:
        key = (row["C"], row["A"]) if access == "raw" else (
            (row["C"], row["D"]) if access == "magnitude" else row[access])
        out[key].append(row)
    return out


def conditional_mean(rr, key):
    mass = sum(r["p"] for r in rr)
    return sum(r["p"] * r[key] for r in rr) / mass


def conditional_median(rr, key):
    mass = sum(r["p"] for r in rr)
    acc = F(0)
    for r in sorted(rr, key=lambda x: x[key]):
        acc += r["p"]
        if 2*acc >= mass:
            return r[key]
    raise AssertionError("empty conditional distribution")


def endpoint_oracle(rows, access, loss, decisions=False):
    total, fitted = F(0), []
    for key, rr in groups(rows, access).items():
        chooser = conditional_mean if loss == "mse" else conditional_median
        y1, y2 = chooser(rr, "Y1"), chooser(rr, "Y2")
        risk = sum(r["p"] * (((y1-r["Y1"])**2 + (y2-r["Y2"])**2)/2
                             if loss == "mse" else
                             (abs(y1-r["Y1"]) + abs(y2-r["Y2"]))/2) for r in rr)
        total += risk
        if decisions:
            fitted.append(dict(observation=str(key), mass=sum(r["p"] for r in rr),
                               endpoint1=y1, endpoint2=y2, contribution=risk))
    return (total, fitted) if decisions else total


def independent_coordinate_minimum(rows, access):
    """Exhaustively minimize max(|M-m|,|B-b|) at all endpoint support vertices.

    This does not use a median, nor the closed formula. A conditional scalar L1
    minimum occurs at a support value. The Cartesian endpoint candidate set thus
    contains an optimizer even if the complete median set is an interval.
    """
    total = F(0)
    for rr in groups(rows, access).values():
        y1_candidates = {r["Y1"] for r in rr}
        y2_candidates = {r["Y2"] for r in rr}
        candidates = []
        for y1, y2 in itertools.product(y1_candidates, y2_candidates):
            m, b = (y1+y2)/2, (y2-y1)/2
            candidates.append(sum(r["p"] * max(abs(m-r["M"]), abs(b-r["B"])) for r in rr))
        total += min(candidates)
    return total


def formulas(alpha, beta, p=F(1, 10)):
    assert 0 <= alpha <= 1 and 0 < p < F(1, 2)
    b = abs(beta)
    return dict(C_mse=p*(1-p)*alpha**2 + (1+3*p)*b*b,
                U_mse=p*(1-p)*(1-alpha)**2/2 + (1+3*p)*b*b,
                C_mae=(1-2*p)*b + p*max(alpha, 3*b),
                U_mae=((1-p)**2 + 2*p*p)*b + p*(1-p)*max(1-alpha, 3*b))


def region_exact(alpha, beta, p):
    """Strict inequalities without evaluating the irrational MSE threshold."""
    b = abs(beta)
    return alpha*alpha + 2*alpha - 1 > 0 and 3*b < 1-alpha and (2-p)*alpha < 1-p+3*p*b


def mae_branch(alpha, beta):
    return ("C_low" if alpha < 3*abs(beta) else "C_high") + ":" + (
        "U_low" if 1-alpha < 3*abs(beta) else "U_high")


def lp_coordinate_mae(rows, access, linprog):
    """Independent continuous LP in (m,b), with per-state L-infinity slacks."""
    import numpy as np
    total, violation = 0.0, 0.0
    for rr in groups(rows, access).values():
        n = len(rr)
        objective = [0.0, 0.0] + [float(r["p"]) for r in rr]
        aa, bb = [], []
        for i, r in enumerate(rr):
            for j, name in ((0, "M"), (1, "B")):
                for sign in (-1, 1):
                    row = [0.0]*(n+2)
                    row[j], row[i+2] = sign, -1.0
                    aa.append(row)
                    bb.append(sign*float(r[name]))
        fit = linprog(objective, A_ub=aa, b_ub=bb,
                      bounds=[(None, None), (None, None)] + [(0, None)]*n, method="highs")
        assert fit.success, fit.message
        total += float(fit.fun)
        violation = max(violation, float((np.asarray(aa) @ fit.x - np.asarray(bb)).max()))
    return dict(risk=total, max_constraint_violation=violation)


def normal_cdf(x):
    return math.erfc(-x/math.sqrt(2))/2


def normal_abs(mean, sd):
    z = mean/sd
    return sd*math.sqrt(2/math.pi)*math.exp(-z*z/2) + mean*(2*normal_cdf(z)-1)


def gaussian_mae_oracle(rows, access, sigma_coordinate):
    """Numerical endpoint medians for a Gaussian-mixture target, not training.

    Independent iid coordinate noise of SD sigma gives endpoint marginal SD
    sqrt(2)*sigma. Conditional risk integrals are available in closed form.
    """
    sd = math.sqrt(2)*sigma_coordinate
    total, max_cdf_residual = 0.0, 0.0
    for rr in groups(rows, access).values():
        mass = float(sum(r["p"] for r in rr))
        for endpoint in ("Y1", "Y2"):
            def cdf(x):
                return sum(float(r["p"])*normal_cdf((x-float(r[endpoint]))/sd) for r in rr)/mass
            lo = min(float(r[endpoint]) for r in rr)-12*sd
            hi = max(float(r[endpoint]) for r in rr)+12*sd
            for _ in range(120):
                mid = (lo+hi)/2
                if cdf(mid) < 0.5:
                    lo = mid
                else:
                    hi = mid
            med = (lo+hi)/2
            max_cdf_residual = max(max_cdf_residual, abs(cdf(med)-0.5))
            total += sum(float(r["p"])*normal_abs(float(r[endpoint])-med, sd)/2 for r in rr)
    return dict(risk=total, max_median_cdf_residual=max_cdf_residual)


def main():
    assert ROOT.drive.upper() == "D:", "All computation and output must be on D:"
    OUT.mkdir(parents=True, exist_ok=True)
    save_json("definitions.json", dict(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        stage="post-review extension; fixed mathematical family, not preregistered",
        support="Independent L,H~Bernoulli(p), fair S; C=L-p, D=H-p, A=S(1+H), U=C+D",
        target="M=C+alpha*D, B=beta*A, Y=(M-B,M+B)",
        domain="0<=alpha<=1, beta real, 0<p<1/2",
        losses="average squared endpoints; average absolute endpoints",
        no_external_data=True, no_neural_training=True,
        disclosure="Parameter family and robustness region were derived after review. Enumeration checks the proof; it is not independent empirical evidence."))

    p_values = [F(1,100), F(2,25), F(1,10), F(3,25), F(1,4), F(2,5), F(49,100)]
    grid, branch_counts = [], defaultdict(int)
    for p, alpha, beta in itertools.product(p_values, [F(i,20) for i in range(21)], [F(i,20) for i in range(15)]):
        rr = support(alpha, beta, p)
        analytic = formulas(alpha, beta, p)
        exact = {f"{access}_{loss}": endpoint_oracle(rr, access, loss)
                 for access, loss in itertools.product(("C", "U"), ("mse", "mae"))}
        assert analytic == exact, (p, alpha, beta, analytic, exact)
        predicted_reversal = region_exact(alpha, beta, p)
        enumerated_reversal = exact["U_mse"] < exact["C_mse"] and exact["U_mae"] > exact["C_mae"]
        assert predicted_reversal == enumerated_reversal
        branch_counts[mae_branch(alpha,beta)] += 1
        grid.append(dict(p=float(p), alpha=float(alpha), beta=float(beta),
                         **{k:float(v) for k,v in exact.items()},
                         mse_gap_exact=str(exact["U_mse"]-exact["C_mse"]),
                         mae_gap_exact=str(exact["U_mae"]-exact["C_mae"]),
                         reversal=enumerated_reversal, mae_branch=mae_branch(alpha,beta)))
    with (OUT / "exact_parameter_grid.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)

    # Explicit ties, internal points, negative beta, and branch boundaries.
    anchor_cases = [
        (F(1,10),F(9,20),F(1,10)), (F(1,10),F(9,20),F(-1,10)),
        (F(1,10),F(43,100),F(1,10)), (F(1,10),F(47,100),F(1,10)),
        (F(1,10),F(49,100),F(3,20)), (F(1,10),F(1,2),F(1,6)),
        (F(1,10),F(3,10),F(1,10)), (F(1,10),F(7,10),F(1,10)),
        (F(1,10),F(3,5),F(3,10)), (F(1,10),F(1,5),F(3,10)),
        (F(2,5),F(9,20),F(1,10)), (F(1,10),F(12,25),F(1,25)),
        (F(1,10),F(9,19),F(0)), (F(49,100),F(9,20),F(1,10)),
    ]
    box_corners = list(itertools.product((F(2,25),F(3,25)),
                                        (F(43,100),F(47,100)),
                                        (F(2,25),F(3,25))))
    anchor_cases += box_corners
    try:
        from scipy.optimize import linprog
        lp_import_error = None
    except Exception as error:
        linprog, lp_import_error = None, f"{type(error).__name__}: {error}"
    independent_checks, lp_checks, anchors = 0, 0, []
    for p, alpha, beta in anchor_cases:
        rr = support(alpha,beta,p)
        per_access = {}
        for access in ("C", "U", "magnitude", "raw"):
            mae, decisions = endpoint_oracle(rr,access,"mae",True)
            assert independent_coordinate_minimum(rr,access) == mae
            independent_checks += 1
            mse = endpoint_oracle(rr,access,"mse")
            result = dict(mse=mse, mae=mae, mae_decisions=decisions)
            if linprog is not None:
                lp = lp_coordinate_mae(rr,access,linprog)
                assert abs(lp["risk"]-float(mae)) < 1e-11
                assert lp["max_constraint_violation"] < 1e-10
                result["lp_check"] = lp
                lp_checks += 1
            per_access[access] = result
        assert per_access["raw"]["mse"] == per_access["raw"]["mae"] == 0
        for loss in ("mse","mae"):
            assert per_access["magnitude"][loss] <= min(per_access["C"][loss],per_access["U"][loss])
        anchors.append(dict(p=p,alpha=alpha,beta=beta,risks=per_access))
    save_json("anchor_oracles.json", anchors)

    original = support(F(9,20),F(1,10),centered=False)
    save_json("original_eight_state_support.json", original)
    original_risks = formulas(F(9,20),F(1,10))
    assert original_risks == dict(C_mse=F(1249,40000),U_mse=F(2129,80000),C_mae=F(1,8),U_mae=F(53,400))
    noise_results = []
    for p,alpha,beta in [(F(1,10),F(9,20),F(1,10))] + box_corners:
        noisy = support(alpha,beta,p,noise=F(1,1000))
        noiseless = formulas(alpha,beta,p)
        noisy_risks = {f"{a}_{l}":endpoint_oracle(noisy,a,l) for a,l in itertools.product(("C","U"),("mse","mae"))}
        for a in ("C","U"):
            assert noisy_risks[f"{a}_mse"] - noiseless[f"{a}_mse"] == F(1,500000)
            assert abs(noisy_risks[f"{a}_mae"]-noiseless[f"{a}_mae"]) <= F(1,1000)
        assert noisy_risks["U_mse"] < noisy_risks["C_mse"] and noisy_risks["U_mae"] > noisy_risks["C_mae"]
        gauss = {a:gaussian_mae_oracle(support(alpha,beta,p),a,0.001) for a in ("C","U")}
        assert gauss["U"]["risk"] > gauss["C"]["risk"]
        for a in ("C","U"):
            assert abs(gauss[a]["risk"]-float(noiseless[f"{a}_mae"])) <= 0.002/math.sqrt(math.pi)+1e-12
            assert gauss[a]["max_median_cdf_residual"] < 1e-11
        noise_results.append(dict(p=p,alpha=alpha,beta=beta,rademacher_risks=noisy_risks,gaussian_mae=gauss))
    save_json("target_noise_checks.json",noise_results)

    # Ordinary independent Bernoulli probability perturbations, fixed history/map.
    probability_checks = []
    for pc,ph in itertools.product((F(997,10000),F(1003,10000)),repeat=2):
        rr = support(F(9,20),F(1,10),ph,p_level=pc,centered=False)
        risks = {f"{a}_{l}":endpoint_oracle(rr,a,l) for a,l in itertools.product(("C","U"),("mse","mae"))}
        assert risks["U_mse"] < risks["C_mse"] and risks["U_mae"] > risks["C_mae"]
        probability_checks.append(dict(p_level=pc,p_magnitude=ph,risks=risks))
    save_json("fixed_support_probability_checks.json",probability_checks)

    # Counterexample to extending target-noise stability to historical support.
    shifted = support(F(9,20),F(1,10),centered=False,history_shift=F(1,10**9))
    assert len(groups(shifted,"U")) == 4
    shift_risks = {f"{a}_{l}":endpoint_oracle(shifted,a,l) for a,l in itertools.product(("C","U"),("mse","mae"))}
    assert shift_risks["U_mse"] == F(13,1000)
    assert shift_risks["U_mae"] == F(11,100)
    assert shift_risks["U_mae"] < shift_risks["C_mae"]
    save_json("history_support_nonstability.json",dict(upper_C_shift=F(1,10**9),original_U_groups=3,shifted_U_groups=4,risks=shift_risks))

    corner_formulas = [formulas(a,b,p) for p,a,b in box_corners]
    min_mse_margin = min(x["C_mse"]-x["U_mse"] for x in corner_formulas)
    min_mae_margin = min(x["U_mae"]-x["C_mae"] for x in corner_formulas)
    assert min_mse_margin == F(10327,6250000)
    assert min_mae_margin == F(46,15625)
    report = dict(status="PASS",python=platform.python_version(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  exact_grid_points=len(grid), exact_formula_risk_checks=4*len(grid),
                  mae_branch_counts=dict(branch_counts), independent_vertex_checks=independent_checks,
                  continuous_lp_checks=lp_checks, lp_status="PASS" if lp_checks else "NOT RUN",
                  lp_import_error=lp_import_error,
                  mse_switch=math.sqrt(2)-1,
                  strict_open_region="alpha>sqrt(2)-1; 3|beta|<1-alpha; (2-p)alpha<1-p+3p|beta|; 0<p<1/2",
                  closed_box=dict(p=[F(2,25),F(3,25)],alpha=[F(43,100),F(47,100)],absolute_beta=[F(2,25),F(3,25)],
                                  min_MSE_C_minus_U=min_mse_margin,min_MAE_U_minus_C=min_mae_margin),
                  gaussian_coordinate_noise=dict(sigma=0.001,target_L1_distance=0.002/math.sqrt(math.pi),
                      worst_gap_perturbation_bound=0.004/math.sqrt(math.pi),
                      anchor_sigma_sufficient_upper=0.0075*math.sqrt(math.pi)/4,
                      box_sigma_sufficient_upper=float(min_mae_margin)*math.sqrt(math.pi)/4),
                  fixed_support_TV_sufficient_upper=min(F(369,80000)/(2*F(49,16)),F(3,400)/(2*F(7,4))),
                  rademacher_noise_cases=len(noise_results),gaussian_mixture_median_cases=2*len(noise_results),
                  independent_probability_corner_cases=len(probability_checks),
                  observations="Formula checks do not validate neural training. Gaussian mixture checks are numerical, all noiseless/Rademacher support checks use exact fractions.")
    save_json("verification.json",report)
    print(json.dumps(encode(report),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
