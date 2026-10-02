"""Single source-only conditioning repair of the failed R7 raw-access family.

The original 138 producer and all 60 failed-source fits remain immutable.
Only the common fixed affine changes, using source training history entries.
"""
from pathlib import Path
import argparse, importlib.util, hashlib, sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')
BASE=ROOT/'scripts/138_r7_raw_access_experiment_v1.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('r7_frozen_raw',BASE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=ROOT/'results/r7_raw_access_robust_v1'
original_arrays=m.source_arrays
def robust_arrays():
    arrays,weights,_,_=original_arrays()
    observed=arrays['train_observed'].astype(bool)
    measured=arrays['train_values'][observed].astype(np.float64)
    center=float(np.median(measured))
    scale=max(float(1.4826*np.median(np.abs(measured-center))),1.)
    return arrays,weights,center,scale
m.source_arrays=robust_arrays
m.__file__=__file__
ADDENDUM=ROOT/'docs/R7_SOURCE_CONDITIONING_ADDENDUM_20261002.md'
def preflight():
    if not ADDENDUM.exists():raise ValueError('Source-only addendum must precede repair')
    m.preflight()
    p=m.load(m.OUT/'protocol.json')
    p.update({'frozen_raw_factory_sha256':sha(BASE),
        'source_only_addendum_sha256':sha(ADDENDUM),
        'normalization_source':'median and 1.4826 MAD of all observed fine-history source training entries only; one common constant affine',
        'source_only_repair':True,'original_failed_source_family':'results/r7_raw_access_v1',
        'external_forecasting_gate':'only if all six-core family mean source validation NMAE is below daily-naive benchmark 0.11557961237049463; else source-only failure retained and no external inference'})
    m.save(m.OUT/'protocol.json',p)
def check():
    p=m.load(m.OUT/'protocol.json')
    if p['frozen_raw_factory_sha256']!=sha(BASE):raise ValueError('Frozen factory changed')
    if p['source_only_addendum_sha256']!=sha(ADDENDUM):raise ValueError('Addendum changed')
def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=('preflight','train','forecast'));s=a.parse_args().stage
    if s=='preflight':preflight();return
    check()
    if s=='forecast':
        reports=[m.load(q) for q in (m.OUT/'fits').glob('*.json')]
        core=[r['source_validation_nmae'] for r in reports if r['adapter'] in m.CORE]
        if len(core)!=36 or np.mean(core)>=0.11557961237049463:raise ValueError('Source competence gate failed; do not forecast')
    {'train':m.train,'forecast':m.forecast}[s]()
if __name__=='__main__':main()
