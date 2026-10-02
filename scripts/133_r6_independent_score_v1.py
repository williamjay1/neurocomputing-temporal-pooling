"""Independent R6 audit/scoring; never import the experiment producers.

Engineering/oracle checks may run before forecasts. Empirical truth is loaded
only after the entire fixed family and all its provenance bindings pass.
The separate confirmation command requires its own root gate/full-family chain.
"""
from __future__ import annotations
import argparse, csv, hashlib, inspect, json, math, os, platform, sys, time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr

ROOT = Path('D:/MLWork/Neurocomputing_TailTransfer')
OUT = ROOT/'results/r6_independent_v1'
REAL = ROOT/'results/r6_access_v1'
CONFIRM = ROOT/'results/r6_confirmation_v1'
SYN = ROOT/'results/r6_oracle_v1'
DATA = ROOT/'datasets/upgrade5_uci_v1'
CODE = Path(__file__).resolve()
REAL_ADAPTERS = ('direct','max_repeat','avg_repeat','haar_absolute','haar_signed','max_blur_repeat')
SYN_ADAPTERS = ('direct','max_repeat','avg_repeat','mean_abs','mean_signed')
HISTORIES = ('fine','coarse')
SEEDS = (20261001,20261002,20261003)
BOOTSTRAP_SEED, BOOTSTRAP_REPS = 13320261002, 5000
ROOT_PROTOCOL = ROOT/'docs/R6_RESEARCH_PROTOCOL_20261002.md'
ROOT_PROTOCOL_SHA = 'c9e76c448cc384c82b82c0b2f72afeca5f3cea4a25664657c1d30670cb372a04'
ADDENDUM = ROOT/'docs/R6_IMPLEMENTATION_ADDENDUM_20261002.md'
ADDENDUM_SHA = '798a8573484cc0e0a10fb31792b7c9840146c89ebe863758bb5546f4b7123e99'
METRICS = ('endpoint_mae_kw','endpoint_nmae','endpoint_mse_kw2','mean_mse_kw2','detail_mse_kw2')
CHECKS = []

def check(condition, name):
    if not bool(condition):
        raise AssertionError(name)
    CHECKS.append(name)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def loadj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def savej(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError('Preserve completed independent output: '+str(p))
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def savecsv(path, rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError(p)
    if not rows: raise ValueError('No rows for '+str(p))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def verify_binding(path, expected, label):
    check(Path(path).is_file(),label+' exists')
    actual=digest(path);check(actual==expected,label+' SHA256')
    return actual

def real_tag(adapter,history,seed): return f'{adapter}_{history}_seed{seed}'
def syn_tag(alpha,adapter,seed): return f'alpha{int(alpha)}_beta05_{adapter}_seed{seed}'

def own_history(raw,mask,coarse=False):
    """Independent observed-only pairing followed by causal carry-forward."""
    a=np.array(raw,dtype=np.float32,copy=True);m=np.asarray(mask,dtype=bool)
    check(a.shape==m.shape and a.ndim==2 and a.shape[1]%2==0,'history fixture/schema')
    if coarse:
        z=a.reshape(len(a),-1,2);o=m.reshape(len(a),-1,2)
        n=o.sum(2)
        totals=np.where(o,z,0).astype(np.float64).sum(2)
        means=np.full(totals.shape,np.nan,dtype=np.float64)
        np.divide(totals,n,out=means,where=n>0)
        a=np.repeat(means.astype(np.float32),2,1)
        coverage=np.repeat(n.astype(np.float32)*.5,2,1)
    else: coverage=m.astype(np.float32)
    carry=np.zeros(len(a),dtype=np.float32)
    for k in range(a.shape[1]):
        ok=coverage[:,k]>0
        carry[ok]=a[ok,k];a[:,k]=carry
    check(np.isfinite(a).all(),'decoded history finite')
    return a,coverage

def independent_real_model(adapter,seed):
    """Library blocks are used; no producer factory or metric is imported."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    from neuralforecast.models.nhits import NHITSBlock,_IdentityBasis
    torch.set_num_threads(2);torch.manual_seed(seed)
    class Adapter(nn.Module):
        def forward(self,x):
            if adapter=='direct': return x
            if adapter=='max_blur_repeat':
                y=F.max_pool1d(F.pad(x,(0,1),mode='replicate'),2,1)
                filt=x.new_tensor([.25,.5,.25]).reshape(1,1,3)
                return F.conv1d(F.pad(y,(1,1),mode='replicate'),filt,stride=2).repeat_interleave(2,-1)
            z=x.reshape(*x.shape[:-1],336,2)
            if adapter=='max_repeat': return z.amax(-1).repeat_interleave(2,-1)
            if adapter=='avg_repeat': return z.mean(-1).repeat_interleave(2,-1)
            mean=(z[...,0]+z[...,1])/math.sqrt(2)
            detail=(z[...,0]-z[...,1])/math.sqrt(2)
            if adapter=='haar_absolute': detail=detail.abs()
            return torch.stack((mean,detail),-1).flatten(-2)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks=nn.ModuleList([NHITSBlock(input_size=672,h=96,n_theta=672+96//down,
                mlp_units=[[64,64],[64,64]],basis=_IdentityBasis(672,96,'linear',1),
                futr_input_size=0,hist_input_size=0,stat_input_size=0,
                n_pool_kernel_size=pool,pooling_mode='MaxPool1d',dropout_prob=0.,activation='ReLU')
                for down,pool in ((8,4),(4,2),(1,1))])
            self.blocks[-1].pooling_layer=Adapter()
            if adapter in ('haar_signed','haar_absolute'):
                with torch.no_grad():
                    w=self.blocks[-1].layers[0].weight
                    old=w.reshape(64,336,2).clone()
                    w.copy_(torch.stack(((old[:,:,0]+old[:,:,1])/math.sqrt(2),
                                        (old[:,:,0]-old[:,:,1])/math.sqrt(2)),-1).reshape(64,672))
        def forward(self,x,coverage):
            count=coverage.sum(-1,keepdim=True).clamp_min(1)
            center=(coverage*x).sum(-1,keepdim=True)/count
            variance=((x-center)**2*coverage).sum(-1,keepdim=True)/count
            scale=torch.maximum(variance.sqrt(),torch.maximum(.05*center.abs(),torch.ones_like(center)))
            normal=(x-center)/scale
            residual=normal.flip(-1);mask=coverage.flip(-1)
            prediction=normal[:,-1:].expand(-1,96)
            for block in self.blocks:
                back,future=block(residual,None,None,None)
                residual=(residual-back)*mask
                prediction=prediction+future[:,:,0]
            return prediction*scale+center
    return Model()

def engineering():
    """Deterministic engineering fixtures, never study outcomes."""
    import torch
    from neuralforecast.models.nhits import NHITSBlock
    rng=np.random.default_rng(13301)
    x=rng.normal(20,4,(6,672)).astype(np.float32);mask=np.ones_like(x,np.uint8)
    mask[0,:7]=0;mask[1,10]=0;x[mask==0]=np.nan
    bad=x.copy();bad[mask==0]=1e25
    for coarse in (False,True):
        a=own_history(x,mask,coarse);b=own_history(bad,mask,coarse)
        check(all(np.array_equal(i,j) for i,j in zip(a,b)),'missing-history poison invariant '+str(coarse))
    xc,mc=own_history(x,mask,True)
    check(mc[1,10]==mc[1,11]==.5,'partial-pair coverage .5')
    p=rng.integers(10,30,(4,336)).astype(np.float32)
    p=np.repeat(p,2,1);q=p.copy();q[:,::2]+=2;q[:,1::2]-=2
    check(all(np.array_equal(a,b) for a,b in zip(own_history(p,np.ones_like(p),True),
         own_history(q,np.ones_like(q),True))),'coarse independent of signed pair detail')
    xf,mf=own_history(x,mask)
    outputs={};shapes=None
    for a in REAL_ADAPTERS:
        net=independent_real_model(a,SEEDS[0])
        check(sum(p.numel() for p in net.parameters())==240036,'registered parameters '+a)
        s={k:tuple(v.shape) for k,v in net.named_parameters()}
        if shapes is None: shapes=s
        check(s==shapes,'identical learned shapes '+a)
        xx=torch.tensor(xf);cm=torch.tensor(mf);out=net(xx,cm)
        check(tuple(out.shape)==(6,96) and torch.isfinite(out).all(),'finite 96-output '+a)
        out.mean().backward()
        active=sum(p.numel() for p in net.parameters() if p.grad is not None)
        # Backcast/forecast rows share the same Linear parameter object. Count
        # structurally inactive individual coefficients from final head rows.
        check(active==240036,'registered tensor gradient reachability '+a)
        check(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()),'finite gradients '+a)
        last=net.blocks[-1].layers[-1]
        check(tuple(last.weight.shape)==(768,64),'last theta head 672 backcast plus96 forecast '+a)
        check(torch.count_nonzero(last.weight.grad[:672]).item()==0 and
              torch.count_nonzero(last.bias.grad[:672]).item()==0,'final backcast gradient exactly zero '+a)
        outputs[a]=out.detach().numpy()
    alignment=float(np.abs(outputs['direct']-outputs['haar_signed']).max())
    check(np.allclose(outputs['direct'],outputs['haar_signed'],rtol=3e-5,atol=5e-5),'direct/signed NHITS initial function alignment')
    # Orthogonal two-coordinate MLP initialization independent of producer.
    torch.manual_seed(SEEDS[0]);direct=torch.nn.Sequential(torch.nn.Linear(2,64),torch.nn.ReLU(),
        torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
    torch.manual_seed(SEEDS[0]);signed=torch.nn.Sequential(torch.nn.Linear(2,64),torch.nn.ReLU(),
        torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
    T=torch.tensor([[1.,1.],[-1.,1.]])/math.sqrt(2)
    with torch.no_grad(): signed[0].weight.copy_(signed[0].weight@T.T)
    xx=torch.tensor(rng.normal(size=(128,2)),dtype=torch.float32)
    with torch.no_grad(): align2=float((direct(xx)-signed(xx@T.T)).abs().max())
    check(align2<2e-6,'direct/signed synthetic initial alignment')
    e=rng.normal(size=(100,2));m=e.mean(1);d=(e[:,1]-e[:,0])/2
    check(np.allclose((e*e).mean(1),m*m+d*d,atol=1e-12),'endpoint MSE = mean MSE + detail MSE')
    # A constant paired loss must remain unchanged under joint resampling.
    toycounts=np.asarray([[2,3,4],[3,2,5]],dtype=np.int64)
    toysums=np.stack((toycounts*2.,toycounts*4.),0)[None,:,:,:]
    toyreps=bootstrap_macro(toysums,toycounts)
    check(np.array_equal(toyreps[:,0,0],np.full(BOOTSTRAP_REPS,2.)) and
          np.array_equal(toyreps[:,0,1],np.full(BOOTSTRAP_REPS,4.)),
          'matched customer/week bootstrap constant-loss denominator fixture')
    report={'status':'PASS_ENGINEERING_FIXTURES_ONLY','scorer_sha256':digest(CODE),
        'checks':list(CHECKS),'real_or_synthetic_test_values_loaded':False,
        'formal_fits_executed':0,'direct_signed_real_max_difference':alignment,
        'direct_signed_synthetic_max_difference':align2,
        'registered_parameters':240036,'structurally_prediction_active_parameters':196356,
        'NHITS_library_path':inspect.getfile(NHITSBlock),'NHITS_library_sha256':digest(inspect.getfile(NHITSBlock)),
        'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
        'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_resamples':BOOTSTRAP_REPS,
        'bootstrap_definition':'joint customer and Monday-calendar-week resampling; average errors over three fixed fits first; identical draws across all cells'}
    savej(OUT/'engineering_checks_final.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},ensure_ascii=False))

def posterior_moments(z,sa):
    """|A| conditional on C+|A|=z, computed from truncated normal moments."""
    total=1+sa*sa;tau=sa/math.sqrt(total);t=z*sa/math.sqrt(total)
    ratio=math.exp(-.5*t*t-.5*math.log(2*math.pi)-float(log_ndtr(t)))
    mean=z*sa*sa/total+tau*ratio
    variance=tau*tau*(1-t*ratio-ratio*ratio)
    if variance<-2e-10: raise ArithmeticError('negative conditional variance')
    return mean,max(variance,0.)

def own_oracle():
    """W from posterior-variance and total-variance independent integrals."""
    rows=[];checks=[]
    for sa in (.25,1.,2.):
        s=math.sqrt(1+sa*sa);meanabs=sa*math.sqrt(2/math.pi);V=sa*sa*(1-2/math.pi)
        def density(z): return 2*math.exp(-.5*(z/s)**2)/(s*math.sqrt(2*math.pi))*float(ndtr(z*sa/s))
        def iv(z): return density(z)*posterior_moments(z,sa)[1]
        def im(z): return density(z)*(posterior_moments(z,sa)[0]-meanabs)**2
        wlo,elo=quad(iv,-14*s,14*s,epsabs=2e-10,epsrel=2e-10,limit=400)
        W,err=quad(iv,-14*s,14*s,epsabs=2e-12,epsrel=2e-12,limit=400)
        explained,emean=quad(im,-14*s,14*s,epsabs=2e-12,epsrel=2e-12,limit=400)
        cross=V-explained
        mass,_=quad(density,-14*s,14*s,epsabs=2e-12,epsrel=2e-12,limit=400)
        check(0<W<V,'strict 0<W<V sa'+str(sa))
        check(max(abs(W-wlo),abs(W-cross))<2e-9,'two W identities and tolerances sa'+str(sa))
        check(abs(mass-1)<2e-10,'skew density mass sa'+str(sa))
        switch=math.sqrt(W)/(math.sqrt(W)+math.sqrt(V))
        gap=lambda a:(a-1)**2*W-a*a*V
        check(0<switch<.5 and abs(gap(switch))<1e-12,'unique nonnegative MSE switch sa'+str(sa))
        check(gap(0)>0 and gap(1)<0,'opposite endpoint MSE rankings sa'+str(sa))
        check(np.all(np.diff([gap(a) for a in np.linspace(0,1,501)])<0),'strictly decreasing MSE gap on0:1 sa'+str(sa))
        # Odd signed-detail numerator vanishes under orientation symmetry.
        numerators=[]
        for z in (-2.,0.,2.):
            def oddjoint(a):
                return a*math.exp(-.5*(a/sa)**2-.5*(z-abs(a))**2)/(2*math.pi*sa)
            numerator,_=quad(oddjoint,-12*sa,12*sa,epsabs=1e-12,limit=400)
            numerators.append(numerator)
        check(max(map(abs,numerators))<1e-12,'conditional signed orientation numerator zero sa'+str(sa))
        checks.append({'sigma_c':1.,'sigma_a':sa,'V':V,'W_direct_conditional_variance':W,
            'W_total_variance_identity':cross,'W_looser_quadrature':wlo,'estimated_quad_error':err,
            'alpha_switch':switch,'integrated_density_mass':mass,
            'maximum_orientation_numerator':max(map(abs,numerators)),
            'scope':'Bayes squared loss only; alpha in[0,1]; centered synthetic coordinates; Gaussian independent C/A; no MAE theorem'})
        for alpha in (0.,.25,.5,.75,1.):
            for beta in (0.,.5):
                for adapter in SYN_ADAPTERS:
                    risk=.02
                    if adapter in ('max_repeat','avg_repeat','mean_abs'): risk+=beta*beta*sa*sa
                    if adapter=='max_repeat': risk+=(alpha-1)**2*W
                    if adapter=='avg_repeat': risk+=alpha*alpha*V
                    rows.append({'sigma_c':1.,'sigma_a':sa,'alpha':alpha,'beta':beta,'adapter':adapter,
                        'bayes_pair_mse':risk,'V':V,'W':W,'alpha_switch':switch,
                        'max_minus_avg_bayes_pair_mse':gap(alpha)})
    return rows,checks

def oracle():
    rows,checks=own_oracle();savecsv(OUT/'oracle_grid_independent.csv',rows)
    savej(OUT/'oracle_checks.json',{'status':'PASS_INDEPENDENT_THEOREM_NUMERICAL_CHECKS',
        'scorer_sha256':digest(CODE),'checks':checks,'assertions':list(CHECKS),
        'test_predictions_or_values_read':False,'new_theory_proof_claimed':False})
    print(json.dumps({'status':'PASS_ORACLE_ONLY','checks':checks}))

def synthetic_gate():
    release=loadj(SYN/'root_test_gate.json')
    check(release['status']=='ROOT_APPROVED_ALL30_SYNTHETIC_TEST_FORECAST_AND_SCORE' and
          release['family_count']==30 and release['selection_on_test_results'] is False,'synthetic root all30 release')
    for relative,expected_sha in release['bindings'].items():
        verify_binding(ROOT/relative,expected_sha,'synthetic root release '+relative)
    lockpath=SYN/'experiment_lock.json';lock=loadj(lockpath);lh=digest(lockpath)
    check(lock['status']=='LOCKED_BEFORE_FORMAL_FITTING_AND_TEST_SCORING','synthetic lock status')
    for name,entry in lock['files'].items(): verify_binding(entry['path'],entry['sha256'],'synthetic locked '+name)
    settings=lock['settings']
    check(settings['parameter_count']==4482 and settings['updates']==2000,'synthetic fixed budget')
    check(settings['data_counts']=={'train':20000,'validation':10000,'test':100000},'synthetic sample sizes')
    check(settings['adapters']==list(SYN_ADAPTERS) and settings['model_seeds']==list(SEEDS),'synthetic full declared family')
    completion=loadj(SYN/'training_completion.json');th=digest(SYN/'training_completion.json')
    check(completion['status']=='ALL_30_SOURCE_SELECTED_CHECKPOINTS_COMPLETE_TEST_UNREAD' and completion['jobs']==30,'synthetic all30 training completion')
    check(completion['lock_sha256']==lh,'synthetic training lock binding')
    expected={syn_tag(a,b,s) for a in (0,1) for b in SYN_ADAPTERS for s in SEEDS}
    check(set(completion['reports'])==expected,'synthetic all30 report keys')
    reports={}
    for name,b in completion['reports'].items():
        verify_binding(b['report_path'],b['report_sha256'],'synthetic report '+name)
        r=loadj(b['report_path']);reports[name]=r
        check(r['parameter_count']==4482 and r['updates_run']==2000 and r['test_values_read'] is False,'synthetic source-only fit '+name)
        check(r['lock_sha256']==lh,'synthetic fit lock '+name)
        verify_binding(SYN/'models'/f'{name}.npz',r['checkpoint_sha256'],'synthetic checkpoint '+name)
        tracepath=SYN/'models'/f'{name}_trace.csv';verify_binding(tracepath,r['trace_sha256'],'synthetic trace '+name)
        with tracepath.open(encoding='utf-8',newline='') as f: trace=list(csv.DictReader(f))
        check([int(t['update']) for t in trace]==list(range(200,2001,200)),'synthetic validation schedule '+name)
        chosen=min(trace,key=lambda t:float(t['validation_pair_mse']))
        check(int(chosen['update'])==r['selected_update'] and float(chosen['validation_pair_mse'])==r['best_validation_pair_mse'],'synthetic source-selected checkpoint '+name)
        with np.load(SYN/'models'/f'{name}.npz',allow_pickle=False) as z:
            check(set(z.files)=={'0.weight','0.bias','2.weight','2.bias','4.weight','4.bias'},'synthetic checkpoint tensor keys '+name)
            check(sum(z[k].size for k in z.files)==4482,'synthetic checkpoint parameter count '+name)
    predpath=SYN/'prediction_completion.json';pc=loadj(predpath)
    check(pc['status']=='ALL_30_FIXED_FAMILY_TEST_FORECASTS_COMPLETE_UNSCORED' and pc['jobs']==30,'synthetic all30 forecasts gate')
    check(pc['lock_sha256']==lh and pc['training_completion_sha256']==th,'synthetic forecast chain')
    check(set(pc['predictions'])==expected,'synthetic all30 forecast keys')
    for name,b in pc['predictions'].items(): verify_binding(b['path'],b['sha256'],'synthetic forecast '+name)
    return lock,completion,pc,reports

def synthetic():
    lock,tc,pc,reports=synthetic_gate()  # No test values above this line.
    latent=np.load(SYN/'latent_test.npy',allow_pickle=False)
    check(latent.shape==(100000,4) and latent.dtype==np.float64,'actual latent test schema')
    rows,checks=own_oracle();risks={(r['alpha'],r['adapter']):r['bayes_pair_mse'] for r in rows if r['sigma_a']==1 and r['beta']==.5}
    scores=[];contrasts=[]
    c,a,ec,ea=latent.T
    for alpha in (0,1):
        mean=c+alpha*(np.abs(a)-math.sqrt(2/math.pi))+ec;detail=.5*a+ea
        target=np.stack((mean-detail,mean+detail),1)
        for adapter in SYN_ADAPTERS:
            for seed in SEEDS:
                tag=syn_tag(alpha,adapter,seed);r=reports[tag]
                with np.load(pc['predictions'][tag]['path'],allow_pickle=False) as z:
                    check(set(z.files)=={'predictions','row_indices'},'synthetic prediction schema '+tag)
                    check(np.array_equal(z['row_indices'],np.arange(100000,dtype=np.int64)),'synthetic row alignment '+tag)
                    pred=z['predictions'].astype(np.float64)
                check(pred.shape==target.shape and np.isfinite(pred).all(),'synthetic actual forecast finite '+tag)
                error=pred-target;m=error.mean(1);d=(error[:,1]-error[:,0])*.5;squared=(error*error).mean(1)
                identity=float(np.max(np.abs(squared-m*m-d*d)))
                check(identity<2e-12,'synthetic independent decomposition identity '+tag)
                scores.append({'alpha':alpha,'beta':.5,'adapter':adapter,'seed':seed,'test_pairs':100000,
                    'test_pair_mse':float(squared.mean()),'test_endpoint_mae':float(np.abs(error).mean()),
                    'test_mean_mse':float(np.mean(m*m)),'test_detail_mse':float(np.mean(d*d)),
                    'iid_test_score_mc_standard_error':float(squared.std(ddof=1)/math.sqrt(100000)),
                    'population_bayes_pair_mse':risks[alpha,adapter],
                    'measured_minus_population_bayes_pair_mse':float(squared.mean())-risks[alpha,adapter],
                    'selected_update':r['selected_update'],'fit_wall_seconds':r['fit_wall_seconds'],
                    'fit_cpu_seconds':r['fit_cpu_seconds'],'parameter_count':4482,'pair_identity_max_error':identity})
        for seed in SEEDS:
            g={r['adapter']:r for r in scores if r['alpha']==alpha and r['seed']==seed}
            contrasts.append({'alpha':alpha,'seed':seed,
                'max_minus_avg_pair_mse':g['max_repeat']['test_pair_mse']-g['avg_repeat']['test_pair_mse'],
                'max_minus_avg_endpoint_mae':g['max_repeat']['test_endpoint_mae']-g['avg_repeat']['test_endpoint_mae'],
                'mean_abs_minus_signed_pair_mse':g['mean_abs']['test_pair_mse']-g['mean_signed']['test_pair_mse'],
                'signed_minus_direct_pair_mse':g['mean_signed']['test_pair_mse']-g['direct']['test_pair_mse'],
                'oracle_max_minus_avg_pair_mse':risks[alpha,'max_repeat']-risks[alpha,'avg_repeat']})
    summary=[]
    for alpha in (0,1):
        for adapter in SYN_ADAPTERS:
            g=[r for r in scores if r['alpha']==alpha and r['adapter']==adapter]
            summary.append({'alpha':alpha,'beta':.5,'adapter':adapter,'fits':3,
                **{metric+'_three_seed_mean':float(np.mean([r[metric] for r in g])) for metric in
                   ('test_pair_mse','test_endpoint_mae','test_mean_mse','test_detail_mse')},
                'test_pair_mse_seed_min':min(r['test_pair_mse'] for r in g),
                'test_pair_mse_seed_max':max(r['test_pair_mse'] for r in g),
                'population_bayes_pair_mse':risks[alpha,adapter]})
    savecsv(OUT/'synthetic_metrics_by_seed.csv',scores);savecsv(OUT/'synthetic_metrics_summary.csv',summary)
    savecsv(OUT/'synthetic_contrasts_by_seed.csv',contrasts)
    comparisons=[]
    if (SYN/'metrics_by_seed.csv').exists():
        with (SYN/'metrics_by_seed.csv').open(encoding='utf-8',newline='') as f: producer=list(csv.DictReader(f))
        pmap={(int(float(r['alpha'])),r['adapter'],int(r['seed'])):r for r in producer}
        check(len(pmap)==30,'producer synthetic metric row count')
        for r in scores:
            p=pmap[r['alpha'],r['adapter'],r['seed']]
            for metric in ('test_pair_mse','test_endpoint_mae','test_mean_mse','test_detail_mse'):
                diff=abs(r[metric]-float(p[metric]));check(diff<2e-12,'synthetic producer comparison '+metric)
                comparisons.append(diff)
    outputs={p.name:digest(p) for p in OUT.glob('synthetic_*.csv')}
    savej(OUT/'synthetic_audit.json',{'status':'PASS_COMPLETE_30_FIT_SYNTHETIC_FAMILY',
        'scorer_sha256':digest(CODE),'experiment_lock_sha256':digest(SYN/'experiment_lock.json'),
        'training_completion_sha256':digest(SYN/'training_completion.json'),
        'prediction_completion_sha256':digest(SYN/'prediction_completion.json'),
        'oracle_checks':checks,'assertions':list(CHECKS),'output_sha256':outputs,
        'producer_metrics_present_and_compared':bool(comparisons),'maximum_metric_disagreement':max(comparisons,default=None),
        'bounds':'Constructed Gaussian squared-loss example; trained models not claimed Bayes-optimal; three-seed variation and fixed-test Monte Carlo uncertainty distinct.'})
    print(json.dumps({'status':'PASS_SYNTHETIC','fits':30,'contrasts':contrasts}))

def real_source_gate():
    verify_binding(ROOT_PROTOCOL,ROOT_PROTOCOL_SHA,'immutable root R6 protocol')
    verify_binding(ADDENDUM,ADDENDUM_SHA,'immutable implementation/confirmation addendum')
    pf=loadj(REAL/'preflight.json');protocolpath=REAL/'proposed_protocol.json';protocol=loadj(protocolpath)
    check(pf['status']=='PASS' and pf['synthetic_only'] is True,'real engineering preflight')
    for p,b in pf['bindings'].items(): verify_binding(p,b['sha256'],'real preflight '+Path(p).name)
    verify_binding(protocolpath,pf['protocol_sha256'],'real proposed protocol')
    check(protocol['adapters']==list(REAL_ADAPTERS) and protocol['histories']==list(HISTORIES) and protocol['seeds']==list(SEEDS),'real full declared family')
    check(protocol['formal_fit_count']==36 and protocol['parameter_count']==240036 and protocol['prediction_active_parameter_count']==196356,'real equal parameter counts')
    check(protocol['prediction_clipping'].startswith('none'),'actual forecast has no clipping')
    manifest=loadj(DATA/'dataset_manifest_v1.json')
    for name in ('panel_kw.npy','valid.npy','provider_labels.npy','client_ids.npy'):
        verify_binding(DATA/name,manifest['artifacts'][name]['sha256'],'UCI frozen '+name)
    cm=loadj(REAL/'source_cache/manifest.json')
    check(cm['status']=='SOURCE_CACHE_COMPLETE' and cm['all_numeric_samples_strictly_pre2012_october'] is True,'real source cache chronology declaration')
    verify_binding(DATA/'dataset_manifest_v1.json',cm['data_manifest_sha256'],'source cache data manifest')
    verify_binding(ROOT/'scripts/130_r6_access_experiment_v1.py',cm['production_script_sha256'],'source cache producer')
    verify_binding(protocolpath,cm['proposed_protocol_sha256'],'source cache protocol')
    for name,b in cm['files'].items(): verify_binding(REAL/'source_cache'/name,b['sha256'],'source cache '+name)
    complete=loadj(REAL/'source_fit_completion.json')
    check(complete['status']=='ALL36_SOURCE_FITS_COMPLETE_CALIBRATION_GATED' and len(complete['jobs'])==36,'all36 source fit completion')
    verify_binding(ROOT/'scripts/130_r6_access_experiment_v1.py',complete['script_sha256'],'source completion producer')
    verify_binding(protocolpath,complete['protocol_sha256'],'source completion protocol')
    expected={real_tag(a,h,s) for a in REAL_ADAPTERS for h in HISTORIES for s in SEEDS}
    check({Path(j['report_path']).stem for j in complete['jobs']}==expected,'all36 source fit report identities')
    reports={};cp_shapes=None
    for j in complete['jobs']:
        p=Path(j['report_path']);tag=p.stem;verify_binding(p,j['report_sha256'],'real fit report '+tag)
        r=loadj(p);reports[tag]=r
        check(tag==real_tag(r['adapter'],r['history'],r['seed']),'real fit declared identity '+tag)
        check(r['status']=='SOURCE_FIT_COMPLETE_NO_CALIBRATION_PREDICTION' and r['calibration_or_confirmation_inference_performed'] is False,'real source-only fit status '+tag)
        check(r['updates']==2000 and r['parameter_count']==240036 and r['prediction_active_parameter_count']==196356,'real fit budget '+tag)
        check(r['script_sha256']==complete['script_sha256'] and r['protocol_sha256']==complete['protocol_sha256'],'real fit protocol/code '+tag)
        verify_binding(REAL/'source_cache/manifest.json',r['source_cache_manifest_sha256'],'real fit cache '+tag)
        verify_binding(REAL/'source_cache'/f"draws_seed{r['seed']}.npy",r['draws_sha256'],'real fit shared draw '+tag)
        cp=REAL/'source_fits'/f'{tag}.npz';verify_binding(cp,r['checkpoint_sha256'],'real checkpoint '+tag)
        check(j['checkpoint_sha256']==r['checkpoint_sha256'],'real completion checkpoint binding '+tag)
        t=r['trace'];check([i['update'] for i in t]==list(range(200,2001,200)),'real validation schedule '+tag)
        selected=min(t,key=lambda i:i['held_time_complete_pair_nmae'])
        check(selected['update']==r['selected_update'] and selected['held_time_complete_pair_nmae']==r['best_source_held_time_complete_pair_nmae'],'real source-selected checkpoint '+tag)
        check(abs(r['checkpoint_reload_source_nmae']-r['best_source_held_time_complete_pair_nmae'])<2e-7*max(1,abs(r['best_source_held_time_complete_pair_nmae'])),'real reported checkpoint reload '+tag)
        with np.load(cp,allow_pickle=False) as z:
            shapes={k:z[k].shape for k in z.files}
            if cp_shapes is None: cp_shapes=shapes
            check(shapes==cp_shapes,'real all checkpoint tensor shapes '+tag)
            check(sum(z[k].size for k in z.files)==240036,'real serialized coefficient count '+tag)
    return manifest,cm,protocol,complete,reports

def source_audit():
    manifest,cm,protocol,complete,reports=real_source_gate()
    # Only pre-October2012 numerical rows are loaded in this command.
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False);ids=np.load(DATA/'client_ids.npy',allow_pickle=False)
    check(ids.tolist()==manifest['client_id_column_order'],'source exact frozen ID order')
    panel=np.load(DATA/'panel_kw.npy',mmap_mode='r');valid=np.load(DATA/'valid.npy',mmap_mode='r')
    index=lambda date:int(np.searchsorted(labels,np.datetime64(date)))
    expected={
        'train':[(c,o) for c in range(24) for o in range(index('2012-01-08'),index('2012-09-15')-96,48)
            if np.mean(valid[o:o+96,c])>=.95 and np.mean(valid[o-672:o,c])>=.95],
        'validation':[(c,o) for c in range(8) for o in range(index('2012-09-16'),index('2012-09-30')-96,96)
            if np.mean(valid[o:o+96,c])>=.95]}
    for split,keys in expected.items():
        folder=REAL/'source_cache'
        arrays={k:np.load(folder/f'{split}_{k}.npy',mmap_mode='r') for k in ('values','observed','labels','label_observed','row_keys')}
        check(np.array_equal(arrays['row_keys'],np.asarray(keys,np.int64)),'independent exact '+split+' source row keys')
        check(all(o+96<=index('2012-10-01') and c<24 for c,o in keys),'actual source numeric chronology '+split)
        check(len(keys)==cm[split+'_rows'],'source row count '+split)
        harrays={(h,k):np.load(folder/f'{split}_{h}_{k}.npy',mmap_mode='r') for h in HISTORIES for k in ('decoded','coverage')}
        for start in range(0,len(keys),256):
            part=keys[start:start+256]
            raw=np.stack([panel[o-672:o,c] for c,o in part]).astype(np.float32)
            ob=np.stack([valid[o-672:o,c] for c,o in part]).astype(np.uint8)
            y=np.stack([panel[o:o+96,c] for c,o in part]).astype(np.float32)
            ym=np.stack([valid[o:o+96,c] for c,o in part]).astype(np.uint8)
            sl=slice(start,start+len(part))
            for key,actual in (('values',raw),('observed',ob),('labels',np.nan_to_num(y,nan=0.)),('label_observed',ym)):
                check(np.array_equal(arrays[key][sl],actual,equal_nan=True),'raw source cache reconstruction '+split+'/'+key+f'/{start}')
            for h in HISTORIES:
                decoded,coverage=own_history(raw,ob,h=='coarse')
                check(np.array_equal(decoded,harrays[h,'decoded'][sl]),'native decoded reconstruction '+split+'/'+h+f'/{start}')
                check(np.array_equal(coverage,harrays[h,'coverage'][sl]),'native coverage reconstruction '+split+'/'+h+f'/{start}')
    for seed in SEEDS:
        draws=np.load(REAL/'source_cache'/f'draws_seed{seed}.npy',allow_pickle=False)
        own=np.random.default_rng(seed).integers(len(expected['train']),size=(2000,64),dtype=np.int32)
        check(np.array_equal(draws,own),'complete shared minibatch draw reconstruction '+str(seed))
    # Restore the selected snapshots with the independently specified forward
    # and recompute source-held-time scores; this performs no optimization.
    import torch
    held_label=np.load(REAL/'source_cache/validation_labels.npy',allow_pickle=False).astype(np.float64)
    held_observed=np.load(REAL/'source_cache/validation_label_observed.npy',allow_pickle=False).astype(bool)
    pairmask=held_observed.reshape(-1,48,2).all(2)
    reload_scores=[]
    for tag,r in reports.items():
        x=np.load(REAL/'source_cache'/f"validation_{r['history']}_decoded.npy",allow_pickle=False)
        coverage=np.load(REAL/'source_cache'/f"validation_{r['history']}_coverage.npy",allow_pickle=False)
        net=independent_real_model(r['adapter'],r['seed'])
        with np.load(REAL/'source_fits'/f'{tag}.npz',allow_pickle=False) as z:
            net.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files})
        net.eval()
        with torch.no_grad(): predicted=net(torch.from_numpy(x),torch.from_numpy(coverage)).numpy().astype(np.float64)
        abs_pair=np.abs(predicted-held_label).reshape(-1,48,2).mean(2)
        perrow=np.where(pairmask,abs_pair,0.).sum(1)/np.maximum(pairmask.sum(1),1)
        ownscale=np.maximum(np.sum(x.astype(np.float64)*coverage,1)/np.maximum(coverage.sum(1,dtype=np.float64),1),1)
        ownscore=float(np.mean(perrow/ownscale))
        difference=abs(ownscore-r['best_source_held_time_complete_pair_nmae'])
        check(difference<3e-6*max(1,abs(ownscore)),'independent source checkpoint validation '+tag)
        reload_scores.append({'adapter':r['adapter'],'history':r['history'],'seed':r['seed'],
            'independent_source_validation_nmae':ownscore,'producer_source_validation_nmae':r['best_source_held_time_complete_pair_nmae'],
            'absolute_float64_vs_float32_disagreement':difference})
    savecsv(OUT/'source_checkpoint_validation_independent.csv',reload_scores)
    savej(OUT/'source_audit.json',{'status':'PASS_SOURCE_ONLY_DATA_AND_36_FIT_PROVENANCE',
        'scorer_sha256':digest(CODE),'source_completion_sha256':digest(REAL/'source_fit_completion.json'),
        'source_cache_manifest_sha256':digest(REAL/'source_cache/manifest.json'),
        'data_manifest_sha256':digest(DATA/'dataset_manifest_v1.json'),
        'train_rows':len(expected['train']),'validation_rows':len(expected['validation']),
        'maximum_independent_checkpoint_score_disagreement':max(r['absolute_float64_vs_float32_disagreement'] for r in reload_scores),
        'checks':list(CHECKS),'calibration_or_confirmation_numeric_values_read':False,
        'bounds':'Source cache rebuilt independently; actual checkpoint tensors/selection trace verified. No re-fit or producer helper import.'})
    print(json.dumps({'status':'PASS_SOURCE_ONLY','train_rows':len(expected['train']),'validation_rows':len(expected['validation'])}))

def real_prediction_gate(phase='calibration'):
    manifest,cm,protocol,sc,reports=real_source_gate()
    isconfirm=phase=='confirmation';folder=CONFIRM if isconfirm else REAL
    producer=ROOT/'scripts'/('131_r6_confirmation_and_source_diagnostic_v1.py' if isconfirm else '130_r6_access_experiment_v1.py')
    gatepath=folder/('root_confirmation_gate.json' if isconfirm else 'root_calibration_gate.json');gate=loadj(gatepath)
    check(gate['status']=='ROOT_APPROVED_'+phase.upper()+'_INFERENCE' and gate['near_neighbor_gate_complete'] is True,'root '+phase+' authorization')
    binds=[('script_sha256',producer),('source_completion_sha256',REAL/'source_fit_completion.json'),
           ('data_manifest_sha256',DATA/'dataset_manifest_v1.json')]
    if isconfirm:
        binds.extend([('confirmation_protocol_sha256',folder/'confirmation_protocol.json'),
          ('learner_script_sha256',ROOT/'scripts/130_r6_access_experiment_v1.py'),
          ('learner_protocol_sha256',REAL/'proposed_protocol.json')])
        check(gate['independent_engineering_and_data_checks_passed'] is True and
              gate['all36_confirmation_precommitted_independent_of_calibration_scores'] is True,'unconditional confirmation advancement')
        fp=loadj(folder/'preflight.json');plan=loadj(folder/'confirmation_protocol.json')
        check(fp['status']=='PASS' and fp['synthetic_only'] is True,'confirmation source-only preflight')
        verify_binding(producer,fp['script_sha256'],'confirmation preflight producer')
        verify_binding(folder/'confirmation_protocol.json',fp['confirmation_protocol_sha256'],'confirmation preflight protocol')
        for field,path in (('learner_protocol_sha256',REAL/'proposed_protocol.json'),('learner_preflight_sha256',REAL/'preflight.json'),
            ('source_cache_manifest_sha256',REAL/'source_cache/manifest.json'),('data_manifest_sha256',DATA/'dataset_manifest_v1.json')):
            verify_binding(path,plan[field],'confirmation plan '+field)
    else: binds.append(('protocol_sha256',REAL/'proposed_protocol.json'))
    for field,path in binds:
        verify_binding(path,gate[field],'root gate '+field)
    # Root adds protocol/chronology addenda as declared path/hash bindings.
    for field,value in gate.items():
        if field.endswith('_path') and isinstance(value,str):
            hashfield=field[:-5]+'_sha256'
            if hashfield in gate: verify_binding(value,gate[hashfield],'root extra gate '+field)
    pp=folder/'prediction_completion.json';pc=loadj(pp)
    check(pc['status']=='ALL36_'+phase.upper()+'_FORECASTS_COMPLETE_UNSCORED' and len(pc['forecasts'])==36,'all36 '+phase+' forecasts complete before truth')
    if isconfirm:
        check(pc['confirmation_truth_scored_by_predictor'] is False and pc['all36_conditions_retained'] is True,'confirmation no prior scoring/full family')
        verify_binding(ROOT/'scripts/130_r6_access_experiment_v1.py',pc['learner_script_sha256'],'confirmation inherited learner')
        verify_binding(folder/'confirmation_protocol.json',pc['confirmation_protocol_sha256'],'confirmation completion protocol')
    else: check(pc['all_future_target_scoring_independent'] is True and pc['confirmation_2014_used'] is False,'calibration no confirmation use')
    for field,path in (('script_sha256',producer),('root_gate_sha256',gatepath),
        ('source_completion_sha256',REAL/'source_fit_completion.json'),('data_manifest_sha256',DATA/'dataset_manifest_v1.json')):
        verify_binding(path,pc[field],'real forecast completion '+field)
    expected={real_tag(a,h,s) for a in REAL_ADAPTERS for h in HISTORIES for s in SEEDS}
    forecasts={Path(b['path']).stem:b for b in pc['forecasts']}
    check(set(forecasts)==expected,'all36 real forecast identities')
    for tag,b in forecasts.items():
        p=Path(b['path']);verify_binding(p,b['sha256'],'real forecast '+tag)
        rp=p.with_suffix('.json');verify_binding(rp,b['report_sha256'],'real forecast report '+tag)
        r=loadj(rp);f=reports[tag]
        check(r['status']=='COMPLETE_'+phase.upper()+'_FORECAST_UNSCORED','real forecast status '+tag)
        if isconfirm: check(r['confirmation_truth_scored_by_predictor'] is False,'confirmation truth not scored '+tag)
        else: check(r['confirmation_values_used'] is False and r['future_calibration_truth_scored_by_predictor'] is False,'calibration boundary '+tag)
        check(r['forecast_sha256']==b['sha256'] and r['source_fit_report_sha256']==digest(REAL/'source_fits'/f'{tag}.json'),'real prediction fit binding '+tag)
        check(r['checkpoint_sha256']==f['checkpoint_sha256'] and r['script_sha256']==pc['script_sha256'] and r['root_gate_sha256']==digest(gatepath),'real forecast lineage '+tag)
        check(r['parameter_count']==240036 and r['prediction_active_parameter_count']==196356,'real forecast parameter declaration '+tag)
    references=pc['quality_references']
    check(pc['quality_reference_count']==3 and len(references)==3 and
          {b['name'] for b in references}=={'persistence','daily_naive','weekly_naive'},'all three quality references completed before truth')
    for b in references:
        p=Path(b['path']);verify_binding(p,b['sha256'],'quality reference '+b['name'])
        rp=p.with_suffix('.json');verify_binding(rp,b['report_sha256'],'quality reference report '+b['name'])
        r=loadj(rp)
        check(r['status']=='COMPLETE_'+phase.upper()+'_QUALITY_REFERENCE_UNSCORED' and r['fit_count']==0,'quality reference status '+b['name'])
        check(r['forecast_sha256']==b['sha256'] and r['script_sha256']==pc['script_sha256'] and r['root_gate_sha256']==digest(gatepath),'quality reference provenance '+b['name'])
    return manifest,protocol,reports,forecasts,references

def bootstrap_macro(sums,counts):
    """sums[M,cell,customer,week]; counts[customer,week]."""
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    nc,nw=counts.shape;reps=np.empty((BOOTSTRAP_REPS,sums.shape[0],sums.shape[1]),np.float64)
    for b in range(BOOTSTRAP_REPS):
        customers=rng.integers(nc,size=nc);weeks=rng.integers(nw,size=nw)
        denom=counts[np.ix_(customers,weeks)].sum(1)
        if np.any(denom==0): raise ArithmeticError('bootstrap sampled zero-availability customer')
        numerator=sums[:,:,customers][:,:,:,weeks].sum(3)
        reps[b]=(numerator/denom[None,None,:]).mean(2)
    return reps

def uci(phase='calibration'):
    manifest,protocol,reports,forecasts,references=real_prediction_gate(phase)  # All hashes before truth.
    isconfirm=phase=='confirmation';folder=CONFIRM if isconfirm else REAL
    prefix='confirmation' if isconfirm else 'uci'
    nc,nd=(32,365) if isconfirm else (16,184)
    sourceproof=loadj(OUT/'source_audit.json')
    verify_binding(CODE,sourceproof['scorer_sha256'],'independent source audit scorer')
    verify_binding(REAL/'source_fit_completion.json',sourceproof['source_completion_sha256'],'independent source proof binding')
    labels=np.load(DATA/'provider_labels.npy',allow_pickle=False);ids=np.load(DATA/'client_ids.npy',allow_pickle=False)
    customers=manifest['customer_groups'][phase];cols=[ids.tolist().index(c) for c in customers]
    check(cols==list(range(48,80) if isconfirm else range(32,48)),'actual '+phase+' exact fixed columns')
    start=int(np.searchsorted(labels,np.datetime64('2014-01-01' if isconfirm else '2013-07-01')))
    stop=int(np.searchsorted(labels,np.datetime64('2015-01-01' if isconfirm else '2014-01-01')))
    origins=np.arange(start,stop,96,dtype=np.int64);dates=labels[origins].astype('datetime64[D]')
    check(len(origins)==nd and np.all(origins+96<=stop),'actual '+phase+' target chronology')
    panel=np.load(DATA/'panel_kw.npy',mmap_mode='r');valid=np.load(DATA/'valid.npy',mmap_mode='r')
    truth=np.stack([np.stack([panel[o:o+96,c] for o in origins]) for c in cols]).astype(np.float64)
    observation=np.stack([np.stack([valid[o:o+96,c] for o in origins]) for c in cols]).astype(bool)
    live=observation.reshape(nc,nd,48,2).all(3)
    check(np.isfinite(truth.reshape(nc,nd,48,2)[live]).all(),'finite observed complete-pair truth')
    scales=[]
    for c in cols:
        ob=np.asarray(valid[start-672:start,c],bool);v=np.asarray(panel[start-672:start,c],np.float64)
        scales.append(max(float(v[ob].sum(dtype=np.float64)/ob.sum()) if ob.any() else 0.,1.))
    scales=np.asarray(scales,np.float64)
    counts_day=live.sum(2);counts_customer=counts_day.sum(1)
    check(np.all(counts_customer>0),'all fixed '+phase+' customers scoreable')
    weekdates=dates-((dates.astype(np.int64)+3)%7).astype('timedelta64[D]')
    uniqueweeks,weekindex=np.unique(weekdates,return_inverse=True)
    counts_week=np.stack([counts_day[:,weekindex==w].sum(1) for w in range(len(uniqueweeks))],1)
    cells=[(a,h) for a in REAL_ADAPTERS for h in HISTORIES]
    sums_week=np.zeros((len(METRICS),12,nc,len(uniqueweeks)),np.float64)
    scores=[];households=[];date_rows=[];coverage=[]
    for j,customer in enumerate(customers):
        coverage.append({'customer_id':customer,'column':cols[j],'available_complete_pairs':int(counts_customer[j]),
            'total_forecast_pairs':nd*48,'observed_endpoint_count':int(observation[j].sum()),
            'fixed_prior_support_scale_kw':float(scales[j])})
    for ci,(adapter,history) in enumerate(cells):
        for seed in SEEDS:
            tag=real_tag(adapter,history,seed);p=forecasts[tag]['path']
            with np.load(p,allow_pickle=False) as z:
                check(set(z.files)=={'customer_ids','forecast_dates','origin_indices','predictions_kw','support_scales_kw'},'real actual prediction schema '+tag)
                check(z['customer_ids'].tolist()==customers and np.array_equal(z['forecast_dates'],dates) and np.array_equal(z['origin_indices'],origins),'real serialized customer/date/origin alignment '+tag)
                check(np.array_equal(z['support_scales_kw'],scales),'independent fixed prior support scales '+tag)
                pred=z['predictions_kw'].astype(np.float64)
            check(pred.shape==(nc,nd,96) and np.isfinite(pred).all(),'real actual finite prediction '+tag)
            # np.where removes non-observed labels before arithmetic: NaN*0
            # is never used as a masking operation.
            err=np.where(np.repeat(live,2,2),pred-truth,0.).reshape(nc,nd,48,2)
            mean=err.mean(3);detail=(err[:,:,:,1]-err[:,:,:,0])*.5
            pairabs=np.abs(err).mean(3);pairsquared=np.square(err).mean(3)
            arrays=(pairabs,pairabs/scales[:,None,None],pairsquared,mean*mean,detail*detail)
            identity=float(np.max(np.abs(pairsquared-mean*mean-detail*detail)))
            check(identity<2e-7*max(1,float(np.max(pairsquared))),'real exact mean/detail MSE decomposition '+tag)
            daily=np.stack([a.sum(2) for a in arrays])
            sums=np.stack([daily[:,:,weekindex==w].sum(2) for w in range(len(uniqueweeks))],2)
            # Above list elements [metric,customer]; stacking axis2 yields M,C,W.
            sums=sums.reshape(len(METRICS),nc,len(uniqueweeks))
            sums_week[:,ci]+=sums/3
            totals=daily.sum(2)/counts_customer[None,:]
            macro=totals.mean(1)
            r=reports[tag]
            scores.append({'adapter':adapter,'history':history,'seed':seed,'scoreable_customers':nc,
                'complete_pairs':int(counts_customer.sum()),'calendar_dates':nd,
                **{m:float(macro[k]) for k,m in enumerate(METRICS)},
                'endpoint_rmse_kw':float(math.sqrt(macro[2])),
                'selected_update':r['selected_update'],'fit_wall_seconds':r['fit_wall_seconds'],
                'fit_process_cpu_seconds':r['fit_process_cpu_seconds'],'checkpoint_bytes':r['checkpoint_bytes'],
                'registered_parameters':240036,'prediction_active_parameters':196356})
            for j,customer in enumerate(customers):
                households.append({'adapter':adapter,'history':history,'seed':seed,'customer_id':customer,
                    'complete_pairs':int(counts_customer[j]),**{m:float(totals[k,j]) for k,m in enumerate(METRICS)}})
            for di,date in enumerate(dates):
                n=counts_day[:,di];keep=n>0
                date_rows.append({'adapter':adapter,'history':history,'seed':seed,'forecast_date':str(date),
                    'scoreable_customers':int(keep.sum()),'complete_pairs':int(n.sum()),
                    **{m:float(np.mean(daily[k,keep,di]/n[keep])) for k,m in enumerate(METRICS)}})
    check(np.allclose(sums_week[2],sums_week[3]+sums_week[4],rtol=2e-12,atol=1e-6),'aggregated MSE decomposition')
    macro=(sums_week.sum(3)/counts_customer[None,None,:]).mean(2)
    summary=[]
    for ci,(a,h) in enumerate(cells):
        g=[r for r in scores if r['adapter']==a and r['history']==h]
        summary.append({'adapter':a,'history':h,'fits':3,
            **{m+'_three_seed_mean':float(macro[k,ci]) for k,m in enumerate(METRICS)},
            'endpoint_rmse_kw_sqrt_mean_mse':float(math.sqrt(macro[2,ci])),
            'endpoint_nmae_seed_min':min(r['endpoint_nmae'] for r in g),
            'endpoint_nmae_seed_max':max(r['endpoint_nmae'] for r in g),
            'endpoint_mae_kw_seed_min':min(r['endpoint_mae_kw'] for r in g),
            'endpoint_mae_kw_seed_max':max(r['endpoint_mae_kw'] for r in g)})
    # Every contrast is fixed before target scoring; neither winners nor sign
    # determine what is written. Positive = first condition has larger error.
    weights={}
    def vec(terms):
        w=np.zeros(12)
        for a,h,value in terms: w[cells.index((a,h))]+=value
        return w
    for a in REAL_ADAPTERS: weights[a+'_fine_minus_coarse']=vec([(a,'fine',1),(a,'coarse',-1)])
    weights['primary_max_minus_avg_history_interaction']=vec([('max_repeat','fine',1),('max_repeat','coarse',-1),('avg_repeat','fine',-1),('avg_repeat','coarse',1)])
    for h in HISTORIES:
        for first,second in (('max_repeat','haar_absolute'),('haar_absolute','haar_signed'),('haar_signed','direct'),('max_blur_repeat','max_repeat')):
            weights[f'{first}_minus_{second}_{h}']=vec([(first,h,1),(second,h,-1)])
    reps=bootstrap_macro(sums_week,counts_week)
    contrasts=[];seedcontrasts=[]
    for name,w in weights.items():
        boot=reps@w;point=macro@w
        for k,m in enumerate(METRICS):
            lo,hi=np.quantile(boot[:,k],[.025,.975])
            contrasts.append({'contrast':name,'metric':m,'three_seed_mean_difference':float(point[k]),
                'conditional_customer_calendar_week_ci95_low':float(lo),
                'conditional_customer_calendar_week_ci95_high':float(hi),
                'bootstrap_resamples':BOOTSTRAP_REPS,'bootstrap_seed':BOOTSTRAP_SEED,
                'customers':nc,'calendar_week_blocks':len(uniqueweeks),
                'scope':'conditional on three fitted seeds; '+phase+' sample; percentile intervals not multiplicity-adjusted'})
            for seed in SEEDS:
                values=np.asarray([next(r[m] for r in scores if (r['adapter'],r['history'],r['seed'])==(a,h,seed)) for a,h in cells])
                seedcontrasts.append({'contrast':name,'metric':m,'seed':seed,'difference':float(values@w)})
    # No-fit quality references are independently reconstructed from fine
    # causal histories; their errors use the exact same scales/pair mask.
    raw=np.stack([panel[o-672:o,c] for c in cols for o in origins]).astype(np.float32)
    ob=np.stack([valid[o-672:o,c] for c in cols for o in origins]).astype(np.uint8)
    fine,_=own_history(raw,ob)
    reference_metrics=[]
    for b in references:
        name=b['name']
        own=(np.repeat(fine[:,-1:],96,1) if name=='persistence' else fine[:,-96:] if name=='daily_naive' else fine[:,:96]).reshape(nc,nd,96)
        with np.load(b['path'],allow_pickle=False) as z:
            check(z['customer_ids'].tolist()==customers and np.array_equal(z['forecast_dates'],dates) and
                  np.array_equal(z['origin_indices'],origins) and np.array_equal(z['support_scales_kw'],scales),'quality reference keys/scales '+name)
            check(np.array_equal(own,z['predictions_kw']),'independent no-fit history forecast reconstruction '+name)
        er=np.where(np.repeat(live,2,2),own.astype(np.float64)-truth,0.).reshape(nc,nd,48,2)
        mean=er.mean(3);detail=(er[:,:,:,1]-er[:,:,:,0])*.5
        arrays=(np.abs(er).mean(3),np.abs(er).mean(3)/scales[:,None,None],np.square(er).mean(3),mean*mean,detail*detail)
        metrics=np.asarray([(a.sum((1,2))/counts_customer).mean() for a in arrays])
        reference_metrics.append({'reference':name,'fit_count':0,'customers':nc,'complete_pairs':int(counts_customer.sum()),
            **{m:float(metrics[k]) for k,m in enumerate(METRICS)},'endpoint_rmse_kw':float(math.sqrt(metrics[2]))})
    files={prefix+'_metrics_by_seed.csv':scores,prefix+'_metrics_summary.csv':summary,
        prefix+'_metrics_by_customer.csv':households,prefix+'_metrics_by_date.csv':date_rows,
        prefix+'_coverage_and_scales.csv':coverage,prefix+'_contrasts_conditional_bootstrap.csv':contrasts,
        prefix+'_contrasts_by_seed.csv':seedcontrasts,prefix+'_quality_references.csv':reference_metrics}
    for name,rows in files.items(): savecsv(OUT/name,rows)
    statname=prefix+'_paired_week_sufficient_statistics.npz'
    if (OUT/statname).exists(): raise FileExistsError(OUT/statname)
    np.savez_compressed(OUT/statname,metric_names=np.asarray(METRICS),
        cell_adapter=np.asarray([a for a,h in cells]),cell_history=np.asarray([h for a,h in cells]),
        customer_ids=np.asarray(customers),calendar_week_start=uniqueweeks,
        observed_pair_counts=counts_week,three_seed_mean_error_sums=sums_week)
    outputsha={name:digest(OUT/name) for name in files}
    outputsha[statname]=digest(OUT/statname)
    savej(OUT/(prefix+'_audit.json'),{'status':'PASS_COMPLETE_36_FIT_'+phase.upper()+'_FAMILY',
        'scorer_sha256':digest(CODE),'prediction_completion_sha256':digest(folder/'prediction_completion.json'),
        'source_completion_sha256':digest(REAL/'source_fit_completion.json'),
        'root_gate_sha256':digest(folder/('root_confirmation_gate.json' if isconfirm else 'root_calibration_gate.json')),'data_manifest_sha256':digest(DATA/'dataset_manifest_v1.json'),
        'registered_parameters':240036,'prediction_active_parameters':196356,
        'customers':nc,'forecast_dates':nd,'complete_pairs':int(counts_customer.sum()),
        'calendar_week_blocks':len(uniqueweeks),'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_resamples':BOOTSTRAP_REPS,
        'metric_normalization':'independently reconstructed each customer prior672-slot observed arithmetic mean; float64; floor1kW; no test-derived scale',
        'error_mask':'both endpoints observed; same target pairs across every adapter/history/seed',
        'aggregation':'each customer total pair error/available pairs, then equal-customer macro; three fixed seed errors averaged before joint paired bootstrap',
        'confirmation_2014_numeric_values_read':isconfirm,'no_producer_fitter_or_metric_helper_imported':True,
        'checks':list(CHECKS),'output_sha256':outputsha})
    primary=[r for r in contrasts if r['contrast']=='primary_max_minus_avg_history_interaction' and r['metric']=='endpoint_nmae']
    print(json.dumps({'status':'PASS_UCI_'+phase.upper(),'fits':36,'complete_pairs':int(counts_customer.sum()),'primary':primary,'references':reference_metrics}))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('engineering','oracle','source','synthetic','uci','confirmation'))
    args=p.parse_args();started=time.perf_counter()
    {'engineering':engineering,'oracle':oracle,'source':source_audit,'synthetic':synthetic,'uci':uci,
     'confirmation':lambda:uci('confirmation')}[args.stage]()
    print(json.dumps({'stage':args.stage,'independent_wall_seconds':time.perf_counter()-started}))

if __name__=='__main__': main()
