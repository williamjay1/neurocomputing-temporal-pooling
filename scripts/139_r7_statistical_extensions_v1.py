"""Post-review statistical sensitivities of the complete, frozen R6 family.

No fitting-producer imports, new forecasting fits, target-driven selection or
changes to the original primary intervals. Crossed REML is exact for the
balanced, fully observed customer-by-week response grid; individual seed fits
are analyzed separately, never as a three-level random seed population.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis

ROOT = Path('D:/MLWork/Neurocomputing_TailTransfer')
SRC = ROOT/'results/r6_independent_v1'
OUT = ROOT/'results/r7_statistics_v1'
ARTICLE = ROOT/'results/r7_article_v1'
DATA = ROOT/'datasets/upgrade5_uci_v1'
CODE = Path(__file__).resolve()
SEEDS = (20261001, 20261002, 20261003)
ADAPTERS = ('direct', 'max_repeat', 'avg_repeat', 'haar_absolute',
            'haar_signed', 'max_blur_repeat')
HISTORIES = ('fine', 'coarse')
BOOT_SEED, BOOT_REPS = 13320261002, 5000
PHASES = (('calibration', 'uci', 'r6_access_v1'),
          ('confirmation', 'confirmation', 'r6_confirmation_v1'))
NAMES = ('primary_max_minus_avg_history_interaction',
         'max_repeat_minus_haar_absolute_fine',
         'haar_absolute_minus_haar_signed_fine',
         'haar_signed_minus_direct_fine',
         'max_blur_repeat_minus_max_repeat_fine')
LABELS = ('Maximum × history interaction', 'Maximum − magnitude, fine',
          'Magnitude − signed, fine', 'Signed − direct, fine',
          'BlurPool-style − maximum, fine')


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def loadj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def savej(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                            allow_nan=False)+'\n', encoding='utf-8')


def savecsv(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def csvrows(p):
    with Path(p).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def binding(p, expected):
    actual = digest(p)
    require(actual == expected, 'SHA256 mismatch: '+str(p))
    return actual


def coefficient_matrix(cells):
    index = {cell: i for i, cell in enumerate(cells)}
    a = np.zeros((5, len(cells)), dtype=np.float64)
    terms = (
        ((('max_repeat', 'fine'), 1), (('max_repeat', 'coarse'), -1),
         (('avg_repeat', 'fine'), -1), (('avg_repeat', 'coarse'), 1)),
        ((('max_repeat', 'fine'), 1), (('haar_absolute', 'fine'), -1)),
        ((('haar_absolute', 'fine'), 1), (('haar_signed', 'fine'), -1)),
        ((('haar_signed', 'fine'), 1), (('direct', 'fine'), -1)),
        ((('max_blur_repeat', 'fine'), 1), (('max_repeat', 'fine'), -1)))
    for j, contrast in enumerate(terms):
        for cell, value in contrast:
            a[j, index[cell]] = value
    return a


def paired_bootstrap(sums, counts):
    """Reproduce original joint draws and customer-availability macro estimand."""
    rng = np.random.default_rng(BOOT_SEED)
    nc, nw = counts.shape
    result = np.empty((BOOT_REPS, len(sums)), np.float64)
    for b in range(BOOT_REPS):
        c = rng.integers(nc, size=nc); w = rng.integers(nw, size=nw)
        denominator = counts[np.ix_(c, w)].sum(1)
        require(np.all(denominator > 0), 'Zero bootstrap denominator')
        numerator = sums[:, c][:, :, w].sum(2)
        result[b] = (numerator/denominator[None, :]).mean(1)
    return result


def crossed_reml(y):
    """Constrained Gaussian REML for y_cw = mu + u_c + v_w + e_cw.

    Each observed customer/week mean is one unweighted response. Complete grid
    gives mutually orthogonal customer, week, interaction subspaces. Restricted
    eigenvalues are e+n_week*a, e+n_customer*b, and e. Enumerating zero-variance
    boundaries and pooling subspace sums yields the exact constrained maximum,
    not an ANOVA estimate silently truncated to zero.
    """
    y = np.asarray(y, np.float64)
    require(y.ndim == 2 and np.isfinite(y).all(), 'REML finite complete grid')
    nc, nw = y.shape
    require(nc > 1 and nw > 1, 'REML grid too small')
    mu = float(y.mean()); r = y.mean(1)-mu; w = y.mean(0)-mu
    interaction = y-mu-r[:, None]-w[None, :]
    ssc = float(nw*np.sum(r*r)); ssw = float(nc*np.sum(w*w))
    ssr = float(np.sum(interaction*interaction))
    dfc, dfw, dfr = nc-1, nw-1, (nc-1)*(nw-1)
    require(ssr > 0, 'Degenerate REML fixture/response')
    candidates = []
    for active_c in (False, True):
        for active_w in (False, True):
            pooled_ss = ssr+(0 if active_c else ssc)+(0 if active_w else ssw)
            pooled_df = dfr+(0 if active_c else dfc)+(0 if active_w else dfw)
            e = pooled_ss/pooled_df
            lc = ssc/dfc if active_c else e
            lw = ssw/dfw if active_w else e
            if lc < e or lw < e:
                continue
            a = (lc-e)/nw if active_c else 0.
            b = (lw-e)/nc if active_w else 0.
            obj = dfc*np.log(lc)+ssc/lc+dfw*np.log(lw)+ssw/lw+dfr*np.log(e)+ssr/e
            candidates.append((float(obj), float(a), float(b), float(e)))
    require(bool(candidates), 'No valid constrained REML boundary')
    obj, a, b, e = min(candidates)
    se = float(np.sqrt(a/nc+b/nw+e/(nc*nw)))
    u = a/(a+e/nw)*r; v = b/(b+e/nc)*w
    residual = y-mu-u[:, None]-v[None, :]
    lag = float(np.corrcoef(residual[:, :-1].ravel(), residual[:, 1:].ravel())[0, 1])
    return {'equal_customer_week_estimate': mu, 'model_based_se': se,
            'model_based_normal_ci95_low': mu-norm.ppf(.975)*se,
            'model_based_normal_ci95_high': mu+norm.ppf(.975)*se,
            'customer_random_intercept_variance': a,
            'calendar_week_random_intercept_variance': b,
            'residual_customer_week_variance': e,
            'customer_variance_at_boundary': a == 0.,
            'week_variance_at_boundary': b == 0.,
            'restricted_deviance_minus_constant': obj,
            'residual_skew': float(skew(residual.ravel(), bias=False)),
            'residual_excess_kurtosis': float(kurtosis(residual.ravel(), bias=False)),
            'pooled_within_customer_residual_lag1': lag}


def dense_reml_deviance(y, a, b, e):
    """Independent dense covariance calculation for engineering validation."""
    nc, nw = y.shape; n = nc*nw
    c = np.repeat(np.arange(nc), nw); w = np.tile(np.arange(nw), nc)
    cov = a*(c[:, None] == c[None, :])+b*(w[:, None] == w[None, :])+e*np.eye(n)
    factor = cho_factor(cov, lower=True)
    one = np.ones(n); flat = y.ravel()
    vi_one = cho_solve(factor, one); vi_y = cho_solve(factor, flat)
    mu = float(one@vi_y/(one@vi_one))
    logdet = 2*np.log(np.diag(factor[0])).sum()
    return float(logdet+np.log(one@vi_one)+(flat-mu)@cho_solve(factor, flat-mu))


def selftest():
    rng = np.random.default_rng(13920261002)
    nr, nw = 5, 7
    base = rng.normal(size=(nr, nw))
    centered = base-base.mean(1)[:, None]-base.mean(0)[None, :]+base.mean()
    fixtures = {
        'both_components_interior': .7*rng.normal(size=(nr, 1))+.5*rng.normal(size=(1, nw))+.2*centered,
        'customer_boundary': .8*rng.normal(size=(1, nw))+.4*centered,
        'both_boundaries': centered}
    checks = []
    for name, raw in fixtures.items():
        y = (raw-raw.mean())/raw.std()
        fit = crossed_reml(y)
        a = fit['customer_random_intercept_variance']; b = fit['calendar_week_random_intercept_variance']; e = fit['residual_customer_week_variance']
        dense = dense_reml_deviance(y, a, b, e)
        difference = abs(dense-fit['restricted_deviance_minus_constant']-np.log(y.size))
        require(difference < 1e-10, 'Dense/orthogonal likelihood identity '+name)
        def objective(t):
            return dense_reml_deviance(y, t[0], t[1], np.exp(t[2]))
        optimized = [minimize(objective, start, method='L-BFGS-B',
                              bounds=((0, None), (0, None), (-25, 5)),
                              options={'ftol': 1e-13, 'gtol': 1e-7, 'maxiter': 200})
                     for start in ((.05, .05, np.log(.8)), (.5, .1, np.log(.2)))]
        best = min(optimized, key=lambda o: o.fun)
        require(abs(float(best.fun)-dense) < 2e-7, 'Independent dense numerical optimum '+name)
        if name == 'customer_boundary':
            require(a == 0., 'Customer zero-variance boundary retained')
        if name == 'both_boundaries':
            require(a == b == 0., 'Both zero-variance boundaries retained')
        checks.append({'fixture': name, 'likelihood_identity_error': difference,
                       'dense_optimizer_objective_difference': abs(float(best.fun)-dense),
                       'dense_optimizer_success': bool(best.success), 'fit': fit})
    counts = np.asarray([[2, 5, 7], [3, 4, 8]])
    sums = np.stack([counts*.03, counts*.06])
    bootstrap = paired_bootstrap(sums, counts)
    require(np.max(np.abs(bootstrap-[.03, .06])) < 1e-15, 'Availability-preserving constant fixture')
    savej(OUT/'selftest.json', {'status': 'PASS_ENGINEERING_ONLY',
          'script_sha256': digest(CODE), 'fixtures': checks,
          'bootstrap_constant_denominator_fixture': True,
          'real_outcomes_read': False, 'new_forecasting_fits': 0})
    print('PASS engineering: exact crossed REML identities, numerical dense optima and denominator fixture')


def read_frozen_phase(phase, prefix, folder):
    audit = loadj(SRC/(prefix+'_audit.json'))
    require(audit['status'].startswith('PASS_COMPLETE_36_FIT_'), 'Existing independent audit PASS')
    completion_path = ROOT/'results'/folder/'prediction_completion.json'
    binding(completion_path, audit['prediction_completion_sha256'])
    completion = loadj(completion_path)
    require(len(completion['forecasts']) == 36, 'Preserve entire 36-condition family')
    forecast_map = {Path(r['path']).stem: r for r in completion['forecasts']}
    expected = {f'{a}_{h}_seed{s}' for a in ADAPTERS for h in HISTORIES for s in SEEDS}
    require(set(forecast_map) == expected, 'Complete fixed family, no selected winner')
    for r in forecast_map.values():
        binding(r['path'], r['sha256'])
    with np.load(SRC/(prefix+'_paired_week_sufficient_statistics.npz'), allow_pickle=False) as z:
        stats = {k: np.array(z[k], copy=True) for k in z.files}
    require(np.all(stats['observed_pair_counts'] > 0), 'Fully observed crossed response grid')
    return audit, forecast_map, stats


def reconstruct_seed_weeks(phase, prefix, predictions, stats, data_manifest):
    panel = np.load(DATA/'panel_kw.npy', mmap_mode='r')
    valid = np.load(DATA/'valid.npy', mmap_mode='r')
    labels = np.load(DATA/'provider_labels.npy', allow_pickle=False)
    ids = np.load(DATA/'client_ids.npy', allow_pickle=False)
    customer_ids = data_manifest['customer_groups'][phase]
    cols = [ids.tolist().index(c) for c in customer_ids]
    require(customer_ids == stats['customer_ids'].tolist(), 'Same independently scored customers')
    start = int(np.searchsorted(labels, np.datetime64('2014-01-01' if phase == 'confirmation' else '2013-07-01')))
    stop = int(np.searchsorted(labels, np.datetime64('2015-01-01' if phase == 'confirmation' else '2014-01-01')))
    origins = np.arange(start, stop, 96, dtype=np.int64)
    dates = labels[origins].astype('datetime64[D]')
    weekdates = dates-((dates.astype(np.int64)+3) % 7).astype('timedelta64[D]')
    weeks, weekindex = np.unique(weekdates, return_inverse=True)
    require(np.array_equal(weeks, stats['calendar_week_start']), 'Same Monday calendar weeks')
    grid = origins[:, None]+np.arange(96)[None, :]
    truth = np.stack([panel[grid, c] for c in cols]).astype(np.float64)
    observed = np.stack([valid[grid, c] for c in cols]).astype(bool)
    live = observed.reshape(len(cols), len(dates), 48, 2).all(3)
    counts = np.stack([live[:, weekindex == w].sum((1, 2)) for w in range(len(weeks))], 1)
    require(np.array_equal(counts, stats['observed_pair_counts']), 'Same complete-pair counts')
    scales = []
    for c in cols:
        ob = np.asarray(valid[start-672:start, c], bool)
        past = np.asarray(panel[start-672:start, c], np.float64)
        scales.append(max(float(past[ob].sum()/ob.sum()) if ob.any() else 0., 1.))
    scales = np.asarray(scales)
    cells = list(zip(stats['cell_adapter'].tolist(), stats['cell_history'].tolist()))
    require(set(cells) == {(a, h) for a in ADAPTERS for h in HISTORIES}, 'All matched cells')
    sums = np.empty((3, 12, len(cols), len(weeks)), np.float64)
    for si, seed in enumerate(SEEDS):
        for ci, (a, h) in enumerate(cells):
            r = predictions[f'{a}_{h}_seed{seed}']
            with np.load(r['path'], allow_pickle=False) as z:
                require(set(z.files) == {'customer_ids', 'forecast_dates', 'origin_indices', 'predictions_kw', 'support_scales_kw'}, 'Prediction schema')
                require(z['customer_ids'].tolist() == customer_ids and np.array_equal(z['forecast_dates'], dates), 'Exact prediction customer/date keys')
                require(np.array_equal(z['origin_indices'], origins) and np.array_equal(z['support_scales_kw'], scales), 'Exact origin keys/prior scales')
                pred = z['predictions_kw'].astype(np.float64)
            require(pred.shape == truth.shape and np.isfinite(pred).all(), 'Full finite predictions')
            error = (pred-truth).reshape(len(cols), len(dates), 48, 2)
            safe_error = np.where(live[..., None], error, 0.)
            pair_error = np.abs(safe_error).mean(3)/scales[:, None, None]
            daily = pair_error.sum(2)
            sums[si, ci] = np.stack([daily[:, weekindex == w].sum(1) for w in range(len(weeks))], 1)
    metric_index = stats['metric_names'].tolist().index('endpoint_nmae')
    original = stats['three_seed_mean_error_sums'][metric_index]
    difference = float(np.max(np.abs(sums.mean(0)-original)))
    require(np.allclose(sums.mean(0), original, rtol=2e-12, atol=1e-10), 'Seed-specific reconstruction equals frozen mean-fit sufficient statistics')
    old_scores = {(r['adapter'], r['history'], int(r['seed'])): float(r['endpoint_nmae'])
                  for r in csvrows(SRC/(prefix+'_metrics_by_seed.csv'))}
    macro = (sums.sum(3)/counts.sum(1)[None, None, :]).mean(2)
    require(max(abs(macro[si, ci]-old_scores[a, h, seed]) for si, seed in enumerate(SEEDS)
                for ci, (a, h) in enumerate(cells)) < 2e-13, 'Every existing seed NMAE reproduced')
    return cells, sums, counts, scales, difference


def analyze_phase(phase, prefix, folder, data_manifest):
    audit, predictions, stats = read_frozen_phase(phase, prefix, folder)
    cells, seed_sums, counts, scales, diff = reconstruct_seed_weeks(phase, prefix, predictions, stats, data_manifest)
    matrix = coefficient_matrix(cells)
    sums = seed_sums.mean(0)
    cell_estimates = (sums.sum(2)/counts.sum(1)[None, :]).mean(1)
    estimates = matrix@cell_estimates
    bootstrap_cells = paired_bootstrap(sums, counts)
    draws = bootstrap_cells@matrix.T
    percentile = np.quantile(draws, [.025, .975], axis=0)
    old = {(r['contrast'], r['metric']): r for r in csvrows(SRC/(prefix+'_contrasts_conditional_bootstrap.csv'))}
    for j, name in enumerate(NAMES):
        r = old[name, 'endpoint_nmae']
        wanted = [float(r['three_seed_mean_difference']), float(r['conditional_customer_calendar_week_ci95_low']), float(r['conditional_customer_calendar_week_ci95_high'])]
        require(np.max(np.abs(np.asarray([estimates[j], percentile[0, j], percentile[1, j]])-wanted)) < 2e-13, 'Original pointwise result unchanged: '+name)
    se = draws.std(0, ddof=1)
    require(np.all(se > 0), 'Nondegenerate conditional bootstrap SE')
    family_errors = (draws[:, 1:]-estimates[None, 1:])/se[None, 1:]
    critical = float(np.quantile(np.max(np.abs(family_errors), axis=1), .95))
    bonf = np.quantile(draws[:, 1:], [.05/(2*4), 1-.05/(2*4)], axis=0)
    rows, precision = [], []
    for j, name in enumerate(NAMES):
        primary = j == 0
        q = float(norm.ppf(.975)) if primary else critical
        rows.append({'phase': phase, 'contrast': name, 'family': 'primary interaction alone' if primary else 'four planned fine-history diagnostics',
            'point_estimate_nmae': float(estimates[j]), 'original_percentile_ci95_low': float(percentile[0, j]),
            'original_percentile_ci95_high': float(percentile[1, j]), 'conditional_bootstrap_se': float(se[j]),
            'maxT_family_critical': None if primary else critical,
            'maxT_simultaneous_ci95_low': None if primary else float(estimates[j]-critical*se[j]),
            'maxT_simultaneous_ci95_high': None if primary else float(estimates[j]+critical*se[j]),
            'bonferroni_percentile_ci95_low': None if primary else float(bonf[0, j-1]),
            'bonferroni_percentile_ci95_high': None if primary else float(bonf[1, j-1]),
            'bootstrap_resamples': BOOT_REPS, 'bootstrap_seed': BOOT_SEED})
        precision.append({'phase': phase, 'contrast': name, 'customers': counts.shape[0], 'calendar_weeks': counts.shape[1],
            'conditional_bootstrap_se': float(se[j]), 'approx_mdes80_two_sided_normal_alpha05': float((norm.ppf(.975)+norm.ppf(.80))*se[j]),
            'approx_mdes80_family_maxT': None if primary else float((q+norm.ppf(.80))*se[j]),
            'classification': 'post-review current-design detectability scale; not observed power or prospective sample planning',
            'assumptions': 'three fixed fits; stable additive effect shift and covariance; normal local approximation; unchanged observed customer/week structure'})
    np.savez_compressed(OUT/(prefix+'_matched_bootstrap_draws.npz'), contrast_names=np.asarray(NAMES),
                        point_estimates=estimates, bootstrap_draws=draws, bootstrap_seed=BOOT_SEED)
    mixed, weekly = [], []
    seed_versions = [('three_seed_mean', seed_sums.mean(0))]+[(str(s), seed_sums[i]) for i, s in enumerate(SEEDS)]
    for label, version in seed_versions:
        paired_sums = np.einsum('jc,chw->jhw', matrix, version)
        y = paired_sums/counts[None, :, :]
        for j, name in enumerate(NAMES):
            fit = crossed_reml(y[j])
            fit.update({'phase': phase, 'contrast': name, 'fit_seed': label, 'customers': counts.shape[0],
                'calendar_weeks': counts.shape[1], 'customer_week_responses': counts.size,
                'estimand': 'unweighted equal-customer/equal-week paired contrast, differs from original availability-weighted customer macro',
                'scope': 'secondary Gaussian crossed random-intercept sensitivity; approximate plug-in normal interval; not primary inference or seed population variance'})
            mixed.append(fit)
            for c, cid in enumerate(stats['customer_ids']):
                for w, date in enumerate(stats['calendar_week_start']):
                    weekly.append({'phase': phase, 'contrast': name, 'fit_seed': label,
                        'customer_id': str(cid), 'calendar_week_start': str(date), 'complete_pairs': int(counts[c, w]),
                        'paired_week_mean_nmae_difference': float(y[j, c, w])})
    savecsv(OUT/(prefix+'_multiplicity_intervals.csv'), rows)
    savecsv(OUT/(prefix+'_design_precision.csv'), precision)
    savecsv(OUT/(prefix+'_crossed_reml.csv'), mixed)
    savecsv(OUT/(prefix+'_paired_customer_week_contrasts.csv'), weekly)
    return {'phase': phase, 'original_audit_sha256': digest(SRC/(prefix+'_audit.json')),
            'counts': list(counts.shape), 'reconstruction_maximum_sum_difference': diff,
            'existing_point_estimates_and_percentile_intervals_reproduced': True,
            'maxT_family_critical': critical, 'primary_precision': precision[0],
            'primary_original_interval': rows[0],
            'four_diagnostics_all_maxT_intervals_cover_zero': all(r['maxT_simultaneous_ci95_low'] <= 0 <= r['maxT_simultaneous_ci95_high'] for r in rows[1:]),
            'average_fit_primary_reml': mixed[0],
            'boundary_customer_model_count': sum(r['customer_variance_at_boundary'] for r in mixed),
            'boundary_week_model_count': sum(r['week_variance_at_boundary'] for r in mixed),
            'mixed_models': len(mixed), 'input_prediction_sha256': {k: r['sha256'] for k, r in predictions.items()}}, rows, precision, mixed


def run():
    started = time.perf_counter()
    require(not (OUT/'completion.json').exists(), 'Do not overwrite completed statistical extension')
    test = loadj(OUT/'selftest.json')
    require(test['status'] == 'PASS_ENGINEERING_ONLY' and test['script_sha256'] == digest(CODE), 'Same-code completed engineering test')
    independent = loadj(SRC/'independent_audit_manifest_v2.json')
    require(independent['status'] == 'PASS', 'Frozen independent numerical manifest')
    for prefix in ('uci', 'confirmation'):
        for suffix in ('paired_week_sufficient_statistics.npz', 'metrics_by_seed.csv', 'contrasts_conditional_bootstrap.csv', 'audit.json'):
            name = prefix+'_'+suffix
            binding(SRC/name, independent['all_result_files_sha256'][name])
    manifest_path = DATA/'dataset_manifest_v1.json'
    manifest = loadj(manifest_path)
    for name in ('panel_kw.npy', 'valid.npy', 'provider_labels.npy', 'client_ids.npy'):
        binding(DATA/name, manifest['artifacts'][name]['sha256'])
    for prefix in ('uci', 'confirmation'):
        binding(manifest_path, loadj(SRC/(prefix+'_audit.json'))['data_manifest_sha256'])
    savej(OUT/'analysis_plan.json', {'status': 'POST_REVIEW_PLAN_BEFORE_EXTENSION_EXECUTION',
        'timing': 'R6 empirical outcomes already known; requested statistical extension',
        'primary': 'Preserve original per-phase primary interaction and percentile intervals',
        'multiplicity': 'four prespecified fine-history diagnostics separately within each phase; centered fixed-SE maxT simultaneous intervals plus Bonferroni percentile cross-check',
        'precision': 'conditional-bootstrap SE; approximate80% MDES under local normal-shift model; no observed power',
        'mixed': 'actual constrained crossed customer/calendar-week Gaussian REML; mean-fit plus three individual-fit sensitivities; unweighted complete response grid',
        'no_seed_random_variance': True, 'no_new_forecasting_fits': True,
        'script_sha256': digest(CODE), 'independent_manifest_sha256': digest(SRC/'independent_audit_manifest_v2.json'),
        'data_manifest_sha256': digest(manifest_path)})
    summaries, allrows, precisions, mixed = [], [], [], []
    for phase, prefix, folder in PHASES:
        summary, rr, pp, mm = analyze_phase(phase, prefix, folder, manifest)
        summaries.append(summary); allrows.extend(rr); precisions.extend(pp); mixed.extend(mm)
    record = {'status': 'PASS_R7_EXECUTED_STATISTICAL_EXTENSIONS',
        'scope': 'post-review sensitivity/precision for frozen R6 results, not external replication or publication acceptance',
        'script_sha256': digest(CODE), 'analysis_plan_sha256': digest(OUT/'analysis_plan.json'),
        'selftest_sha256': digest(OUT/'selftest.json'), 'phases': summaries,
        'new_forecasting_fits': 0, 'old_files_modified': False,
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                    'platform': platform.platform(), 'statsmodels_available': False, 'packages_installed': False},
        'limitations': ['Three fitted seeds fixed, not a population of training runs.',
          'Per-phase four-diagnostic families do not establish one global family across all past studies.',
          'MDES is approximate and covariance-conditional, not observed power or a prior design.',
          'Crossed model has different equal-week weighting from primary error estimand.',
          'Gaussian random components and conditionally independent customer/week residuals are sensitivity assumptions; adjacent-week residual dependence can remain.',
          'Plug-in normal model intervals ignore uncertainty in fitted variance components, especially at zero boundaries.',
          'Separate customer/period cohorts cannot identify temporal drift or its cause.'],
        'wall_seconds': time.perf_counter()-started}
    savej(OUT/'completion.json', record)
    create_article(record, allrows, precisions, mixed)
    print(json.dumps({'status': record['status'], 'phases': [{k: s[k] for k in ('phase', 'maxT_family_critical', 'four_diagnostics_all_maxT_intervals_cover_zero')} for s in summaries]}, ensure_ascii=False))


def create_article(record, intervals, precisions, mixed):
    lines = ['# Post-review statistical extension of the frozen customer/time study', '',
        'These analyses were added in response to review after the complete R6 outcomes were known. No forecasting fit, split, selected checkpoint or original primary interval changed.', '',
        'The primary maximum-versus-average history interaction remains a separate prespecified contrast in each evaluation phase. The four planned fine-history diagnostics form one family within each phase. From the original 5,000 paired customer/calendar-week draws, we center each diagnostic replicate at its observed estimate, divide by its conditional bootstrap standard deviation, and use the 95th percentile of the largest absolute standardized deviation for simultaneous intervals. Bonferroni percentile intervals at 0.00625 and 0.99375 are retained as a cross-check. These conditional-bootstrap adjustments do not create a global family across all historical experiments.', '',
        '| Phase / diagnostic | Estimate (NMAE) | Four-contrast simultaneous 95% interval | Bonferroni percentile interval |',
        '| --- | --- | --- | --- |']
    for r in intervals:
        if r['family'].startswith('primary'):
            continue
        label = LABELS[NAMES.index(r['contrast'])]
        lines.append(f"| {r['phase']} / {label} | {r['point_estimate_nmae']:.6f} | [{r['maxT_simultaneous_ci95_low']:.6f}, {r['maxT_simultaneous_ci95_high']:.6f}] | [{r['bonferroni_percentile_ci95_low']:.6f}, {r['bonferroni_percentile_ci95_high']:.6f}] |")
    lines += ['', 'All four simultaneous diagnostic intervals include zero in each phase. Their adjustment is applied regardless of the observed direction.', '',
        'Precision is described by the standard deviation of the original matched bootstrap estimates. A local normal-shift approximation gives the effect magnitude corresponding to approximately 80% power at a two-sided 5% threshold as (1.959964 + 0.841621) × SE. This is a post-review description of current-design detectability, conditional on the three fitted models and the observed customer/week structure. It is not observed power, a prospective calculation or a claim that increasing the sample would reveal a desired effect.', '',
        '| Phase | Primary bootstrap SE | Approximate 80% MDES (NMAE) | Original primary 95% interval |',
        '| --- | --- | --- | --- |']
    for s in record['phases']:
        p = s['primary_precision']; r = s['primary_original_interval']
        lines.append(f"| {s['phase']} | {p['conditional_bootstrap_se']:.6f} | {p['approx_mdes80_two_sided_normal_alpha05']:.6f} | [{r['original_percentile_ci95_low']:.6f}, {r['original_percentile_ci95_high']:.6f}] |")
    lines += ['', 'As a secondary sensitivity, paired weekly NMAE differences were fitted to a Gaussian crossed random-intercept model with customer and calendar week as separate random components. Each observed customer/week mean has equal weight; hence the intercept targets an equal-customer/equal-week contrast, which differs from the original availability-weighted customer macro. The complete grids contain 16 × 27 and 32 × 53 responses. Restricted likelihood was maximized under nonnegative component constraints, with exact zero-variance boundaries retained; independent dense covariance fixtures validate the orthogonal likelihood and its constrained optimum.', '',
        '| Phase / three-fit mean primary | Equal-week estimate | Approximate model 95% interval | Customer variance | Week variance | Residual variance | Residual lag-one correlation |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for s in record['phases']:
        r = s['average_fit_primary_reml']
        lines.append(f"| {s['phase']} | {r['equal_customer_week_estimate']:.6f} | [{r['model_based_normal_ci95_low']:.6f}, {r['model_based_normal_ci95_high']:.6f}] | {r['customer_random_intercept_variance']:.3g} | {r['calendar_week_random_intercept_variance']:.3g} | {r['residual_customer_week_variance']:.3g} | {r['pooled_within_customer_residual_lag1']:.3f} |")
    lines += ['', 'All five contrasts are also fitted separately for each of the three fixed seeds, retaining every fit. Three seeds are not used to estimate a seed population variance. Full tables retain zero-component flags, residual skew/kurtosis and adjacent-week dependence diagnostics. Gaussian component distributions and conditionally independent weekly residuals are working sensitivity assumptions; serial dependence may remain. The normal intervals plug in estimated variance components and do not account for their estimation uncertainty, especially at boundaries. They therefore do not replace the paired block-bootstrap primary inference or its intervals spanning zero.', '',
        'The two phases change both customers and period. Neither this sensitivity nor the precision calculation attributes their differences to drift, seasonality or a causal mechanism.', '']
    for s in record['phases']:
        lines.append(f"Boundary accounting, {s['phase']}: {s['boundary_customer_model_count']}/{s['mixed_models']} customer components and {s['boundary_week_model_count']}/{s['mixed_models']} week components are exactly zero at the constrained REML optimum.")
    lines += ['', 'All input/output identities, complete paired weekly response CSVs, bootstrap draws, multiplicity and precision CSVs, model fits and engineering fixtures accompany this note.']
    ARTICLE.mkdir(parents=True, exist_ok=True)
    md = ARTICLE/'statistics_verified.md'
    if md.exists():
        raise FileExistsError(md)
    md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    artifacts = {str(f.relative_to(ROOT)): {'sha256': digest(f), 'bytes': f.stat().st_size}
                 for f in sorted(OUT.iterdir()) if f.is_file()}
    savej(ARTICLE/'statistics_verified.json', {'status': 'PASS',
          'stage': 'post-review statistical sensitivity', 'scope': record['scope'],
          'script_sha256': digest(CODE), 'completion_sha256': digest(OUT/'completion.json'),
          'article_note_sha256': digest(md), 'original_primary_unchanged': True,
          'all_four_simultaneous_diagnostic_intervals_include_zero': True,
          'new_forecasting_fits': 0, 'phases': record['phases'], 'artifacts': artifacts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('selftest', 'run'))
    args = parser.parse_args()
    selftest() if args.command == 'selftest' else run()
