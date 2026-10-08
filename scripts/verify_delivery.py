"""Read-only verification of frozen computation and delivered summary tables.

Recorded absolute paths are provenance from the original execution. This audit
resolves their project-relative suffix locally so a verified copy on F or another
machine can be checked without rewriting the immutable records.
"""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[1]
NEURAL=ROOT/'results'/'neural'

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def recorded_path(value):
    normal=value.replace('\\','/')
    marker='/JETAI_PoolingMechanisms_20261008/'
    if marker not in normal:
        raise ValueError('Unrecognized execution provenance path')
    suffix=normal.split(marker,1)[1]
    result=(ROOT/suffix).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError('Provenance path leaves the delivered project')
    return result

def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def read_csv(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))

checks=[]
lock=read_json(NEURAL/'experiment_lock.json')
for entry in lock['files']:
    assert sha(recorded_path(entry['path']))==entry['sha256']
checks.append('All frozen source/configuration/input hashes match')
training=read_json(NEURAL/'training_completion.json')
assert training['completed_fits']==325
assert training['independent_test_values_read'] is False
assert training['lock_sha256']==sha(NEURAL/'experiment_lock.json')
for entry in training['reports']:
    path=recorded_path(entry['path'])
    assert sha(path)==entry['sha256']
    report=read_json(path)
    assert report['status']=='COMPLETE_CHECKPOINT_FIXED_TEST_UNREAD'
    assert report['updates_run']==2000 and report['parameter_count']==4482
    assert report['independent_test_values_read'] is False
    assert sha(path.with_suffix('.npz'))==report['checkpoint_sha256']
    assert sha(path.with_suffix('.csv'))==report['trace_sha256']
assert len(training['reports'])==325
checks.append('325 actual checkpoint/trace/report bindings and pre-test status verified')
evaluation=read_json(NEURAL/'evaluation_completion.json')
assert evaluation['evaluated_fits']==325 and evaluation['per_row_archives']==325
assert evaluation['training_completion_sha256']==sha(NEURAL/'training_completion.json')
for entry in evaluation['score_tables']:
    assert sha(recorded_path(entry['path']))==entry['sha256']
scores=read_csv(NEURAL/'all_seed_scores.csv')
assert len(scores)==325 and len({row['label'] for row in scores})==325
assert sum(row['group']=='gaussian' for row in scores)==175
assert sum(row['group']=='finite' for row in scores)==150
assert len(list((NEURAL/'row_losses').glob('*.npz')))==325
checks.append('325 scored fits, independent test loss archives and summary hashes verified')
grouped=defaultdict(list)
for row in scores:
    key=(row['group'],row['alpha_index'],row['access'],row['loss'])
    grouped[key].append(row)
assert len(grouped)==65 and all(len(rows)==5 for rows in grouped.values())
for summary in read_csv(NEURAL/'condition_summary.csv'):
    key=(summary['group'],summary['alpha_index'],summary['access'],summary['loss'])
    items=grouped[key]
    riskfield='test_risk' if key[0]=='gaussian' else 'exact_population_risk'
    risk=[float(row[riskfield]) for row in items]
    gapfield='projection_regret' if key[0]=='gaussian' else 'population_approximation_gap'
    gap=[float(row[gapfield]) for row in items]
    assert abs(statistics.mean(risk)-float(summary['risk_mean']))<1e-12
    assert abs(statistics.mean(gap)-float(summary['approximation_gap_mean']))<1e-12
    assert min(gap)>=-1e-10
checks.append('All 65 condition means and nonnegative risk-gap bounds reaggregated')
comparisons=read_csv(NEURAL/'paired_contrasts.csv')
assert len(comparisons)==13
for row in comparisons:
    maxrows=grouped[(row['group'],row['alpha_index'],'max_repeat',row['loss'])]
    avgrows=grouped[(row['group'],row['alpha_index'],'avg_repeat',row['loss'])]
    maxmap={r['seed']:r for r in maxrows};avgmap={r['seed']:r for r in avgrows}
    if row['group']=='finite':
        gaps=[float(maxmap[s]['exact_population_risk'])-float(avgmap[s]['exact_population_risk']) for s in maxmap]
        assert abs(statistics.mean(gaps)-float(row['exact_population_neural_contrast']))<1e-12
        assert all(g<0 for g in gaps) if row['loss']=='mse' else all(g>0 for g in gaps)
    else:
        alpha=float(row['alpha'])
        protocol=read_json(ROOT/'docs'/'experiment_protocol.json')
        constants=protocol['gaussian']['oracle']
        analytic=(1-alpha)**2*constants['W']-alpha**2*constants['V']
        assert abs(analytic-float(row['analytic_bayes_contrast']))<1e-12
        if row['alpha_index']=='3':
            assert float(row['paired_row_bootstrap_low'])<0<float(row['paired_row_bootstrap_high'])
            assert float(row['test_seed_contrast_min'])<0<float(row['test_seed_contrast_max'])
checks.append('13 prespecified comparisons checked; all finite seeds reverse; Gaussian tie remains unresolved')
theory=read_json(ROOT/'results'/'theory_region'/'verification.json')
assert theory['status']=='PASS' and theory['lp_status']=='PASS'
assert sha(ROOT/'scripts'/'loss_region_oracle.py')==theory['script_sha256']
checks.append('Executed theory/LP verification bound to the actual source')
main=(ROOT/'manuscript'/'JETAI_Manuscript.md').read_text(encoding='utf-8')
for token in ('[Insert','[Populate','[Neural findings','Primary source to be verified'):
    assert token not in main
print(json.dumps(dict(status='PASS',checks=checks,model_fits=325,conditions=65,contrasts=13,
                     main_whitespace_word_count=len(main.split('## Figure captions')[0].split()),
                     count_scope='Whitespace count includes equations and title/abstract; not a publisher word-count determination'),indent=2))
