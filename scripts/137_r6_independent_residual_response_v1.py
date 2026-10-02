"""Independent source-only response audit using frozen independent133 forward.

No producer import, no fitting, no target or source-label accuracy scoring.
Only frozen source-held histories and checkpoint tensors are numerically read.
"""
from __future__ import annotations
import csv, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode=True
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path('D:/MLWork/Neurocomputing_TailTransfer')
CODE=Path(__file__).resolve()
INDEPENDENT=ROOT/'scripts/133_r6_independent_score_v1.py'
EXPECTED133='c7effdc77c58b4e70fa58dd98eb7eecb68a47ba9390be73bd638d6259bde6d0b'
OLD=ROOT/'results/r6_access_v1'
PRODUCER=ROOT/'results/r6_confirmation_v1'
OUT=ROOT/'results/r6_independent_v1'

def run():
    import hashlib
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda:f.read(2**20),b''): h.update(b)
        return h.hexdigest()
    assert sha(INDEPENDENT)==EXPECTED133,'frozen independent forward changed'
    spec=importlib.util.spec_from_file_location('r6_independent133',INDEPENDENT)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    _,_,_,_,reports=audit.real_source_gate()
    metadata=audit.loadj(PRODUCER/'source_residual_diagnostic.json')
    assert metadata['status']=='ALL36_SOURCE_ONLY_DIAGNOSTICS_COMPLETE' and metadata['rows']==288
    assert metadata['script_sha256']==sha(ROOT/'scripts/131_r6_confirmation_and_source_diagnostic_v1.py')
    assert metadata['confirmation_protocol_sha256']==sha(PRODUCER/'confirmation_protocol.json')
    assert metadata['source_completion_sha256']==sha(OLD/'source_fit_completion.json')
    assert metadata['confirmation_or_calibration_inputs_read'] is False and metadata['source_labels_scored'] is False
    sourcecsv=PRODUCER/'source_residual_diagnostic.csv'
    assert sha(sourcecsv)==metadata['csv_sha256']
    with sourcecsv.open(encoding='utf-8',newline='') as f: original=list(csv.DictReader(f))
    def key(r): return (r['adapter'],r['history'],int(r['seed']),float(r['alpha']),int(r['sign']))
    expected={(a,h,s,alpha,sign) for a in audit.REAL_ADAPTERS for h in audit.HISTORIES
              for s in audit.SEEDS for alpha in (0.,.5,1.,2.) for sign in (-1,1)}
    source={key(r):r for r in original}
    assert len(original)==len(source)==288 and set(source)==expected
    rows=[];strict_swap_checks=0;pair_response_checks=0
    started=time.perf_counter()
    for seed in audit.SEEDS:
        for adapter in audit.REAL_ADAPTERS:
            for history in audit.HISTORIES:
                tag=audit.real_tag(adapter,history,seed);report=reports[tag]
                checkpoint=OLD/'source_fits'/f'{tag}.npz'
                net=audit.independent_real_model(adapter,seed)
                with np.load(checkpoint,allow_pickle=False) as z:
                    net.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
                net.eval()
                x=np.load(OLD/'source_cache'/f'validation_{history}_decoded.npy',allow_pickle=False)
                coverage=np.load(OLD/'source_cache'/f'validation_{history}_coverage.npy',allow_pickle=False)
                assert x.shape==coverage.shape==(104,672)
                tx=torch.from_numpy(x);cov=torch.from_numpy(coverage)
                eligible=(cov.flip(-1).reshape(104,336,2)==1).all(-1)
                with torch.no_grad(): reference=net(tx,cov).numpy().astype(np.float64)
                altered={}
                for alpha in (0.,.5,1.,2.):
                    for sign in (-1,1):
                        def intervention(module,arguments):
                            incoming=arguments[0]
                            assert tuple(incoming.shape)==(104,1,672)
                            if alpha==1. and sign==1: return None
                            pair=incoming.reshape(104,336,2)
                            if alpha==1. and sign==-1:
                                replacement=pair[:,:,[1,0]]
                            else:
                                midpoint=(pair[:,:,0]+pair[:,:,1])*.5
                                signed_halfdifference=(pair[:,:,0]-pair[:,:,1])*.5
                                movement=signed_halfdifference*(sign*alpha)
                                replacement=torch.stack((midpoint+movement,midpoint-movement),2)
                            altered_pair=torch.where(eligible[:,:,None],replacement,pair)
                            return (altered_pair.reshape(104,1,672),)
                        hook=net.blocks[-1].pooling_layer.register_forward_pre_hook(intervention)
                        try:
                            with torch.no_grad(): prediction=net(tx,cov).numpy().astype(np.float64)
                        finally: hook.remove()
                        delta=prediction-reference
                        actual_ma=float(np.abs(delta).mean());actual_rms=float(np.sqrt(np.mean(delta*delta)))
                        k=(adapter,history,seed,alpha,sign);prior=source[k]
                        assert prior['checkpoint_sha256']==sha(checkpoint)==report['checkpoint_sha256']
                        assert int(prior['source_windows'])==104
                        assert int(prior['eligible_complete_residual_pairs'])==int(eligible.sum())
                        producer_ma=float(prior['forecast_mean_absolute_change_kw'])
                        producer_rms=float(prior['forecast_rms_change_kw'])
                        assert abs(actual_ma-producer_ma)<=2e-7*max(1.,actual_ma,producer_ma),(k,'MA',actual_ma,producer_ma)
                        assert abs(actual_rms-producer_rms)<=2e-7*max(1.,actual_rms,producer_rms),(k,'RMS',actual_rms,producer_rms)
                        assert actual_rms+1e-12>=actual_ma>=0
                        if alpha==1 and sign==1: assert np.array_equal(prediction,reference)
                        if alpha==1 and sign==-1 and adapter in ('max_repeat','avg_repeat','haar_absolute'):
                            assert np.array_equal(prediction,reference),'local exact pair swap failed'
                            strict_swap_checks+=1
                        altered[alpha,sign]=prediction
                        rows.append({'adapter':adapter,'history':history,'seed':seed,'alpha':alpha,'sign':sign,
                            'independent_mean_absolute_response_kw':actual_ma,'producer_mean_absolute_response_kw':producer_ma,
                            'mean_absolute_response_disagreement_kw':abs(actual_ma-producer_ma),
                            'independent_rms_response_kw':actual_rms,'producer_rms_response_kw':producer_rms,
                            'rms_response_disagreement_kw':abs(actual_rms-producer_rms),
                            'source_windows':104,'eligible_complete_residual_pairs':int(eligible.sum()),
                            'checkpoint_sha256':sha(checkpoint)})
                if adapter in ('max_repeat','avg_repeat','haar_absolute'):
                    for alpha in (0.,.5,1.,2.):
                        assert np.array_equal(altered[alpha,-1],altered[alpha,1]),'signed direction leak after local symmetric adapter'
                        pair_response_checks+=1
    assert len(rows)==288 and strict_swap_checks==18 and pair_response_checks==72
    output=OUT/'source_residual_response_independent.csv'
    audit.savecsv(output,rows)
    proof={'status':'PASS_ALL_288_INDEPENDENT_SOURCE_RESIDUAL_RESPONSES',
        'audit_script_sha256':sha(CODE),'independent_forward_sha256':EXPECTED133,
        'producer_script_sha256':sha(ROOT/'scripts/131_r6_confirmation_and_source_diagnostic_v1.py'),
        'producer_csv_sha256':sha(sourcecsv),'producer_metadata_sha256':sha(PRODUCER/'source_residual_diagnostic.json'),
        'source_completion_sha256':sha(OLD/'source_fit_completion.json'),
        'source_cache_manifest_sha256':sha(OLD/'source_cache/manifest.json'),
        'output_csv_sha256':sha(output),'rows':288,'fixed_models':36,'source_windows_per_model':104,
        'exact_unmodified_pair_sign_swap_invariance_checks':18,
        'exact_all_amplitude_signed_direction_invariance_checks':72,
        'maximum_mean_absolute_response_disagreement_kw':max(r['mean_absolute_response_disagreement_kw'] for r in rows),
        'maximum_rms_response_disagreement_kw':max(r['rms_response_disagreement_kw'] for r in rows),
        'source_labels_or_future_target_values_scored':False,
        'producer_130_131_132_imported':False,'formal_optimization_executed':False,
        'scope':'Frozen last-block preadapter residual intervention; earlier blocks, original normalization and baseline path fixed. Not global input invariance, a learned-information ceiling, or a proof of optimizer dynamics.',
        'wall_seconds':time.perf_counter()-started}
    audit.savej(OUT/'source_residual_response_audit.json',proof)
    print(json.dumps(proof))

if __name__=='__main__': run()
