#!/usr/bin/env python3
"""Measurement-derived, abstaining shadow adviser. This module never schedules work."""
from pathlib import Path
import argparse,hashlib,json,math,statistics,sys

VERSION=1
TARGETS=('cpu_gpu_boundary_ms','gpu_stream_ms','gpu_resident_ms')
RIDGE=.001
MIN_TRAIN=16
MIN_CALIBRATION=5
IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def signature_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def check_record(r):
    if r.get('ok') is not True or r.get('result',{}).get('pass') is not True:raise ValueError('failed measurement')
    if signature_hash(r['signature'])!=r.get('signature_sha256'):raise ValueError('signature content mismatch')
    validate_case(r['result'])
    c,w=r['result'],r['workload']
    if c['tokens']!=w['tokens'] or c['experts']!=w['experts'] or c['histogram']!=w['histogram']:raise ValueError('measured workload mismatch')
    if [c['K'],c['M']]!=r['case']['shape'][:2] or c['quant']!=r['case']['type'].lower() or c['direction']!=r['case']['part']:raise ValueError('measured operator mismatch')
    for name in TARGETS:
        values=r['result'].get(name)
        if not isinstance(values,list) or len(values)<3 or any(type(v) not in [int,float] or not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid raw timing samples')

def expected_records(plan,records,splits):
    expected={(case,w['id']) for case in plan['cases'] for w in plan['workloads'] if w['split'] in splits}
    found=[(r['case']['name'],r['workload']['id']) for r in records]
    if len(found)!=len(set(found)) or set(found)!=expected:raise ValueError('incomplete/duplicate/unplanned evidence set')
    planned={w['id']:w for w in plan['workloads']}
    for r in records:
        w=planned[r['workload']['id']]
        if r['workload']['sha256']!=w['sha256'] or r['workload']['split']!=w['split']:raise ValueError('workload plan mismatch')

def verify_model(model):
    if model.get('schema')!=VERSION or model.get('implementation_sha256')!=IMPLEMENTATION_SHA256:raise ValueError('model implementation mismatch')
    copy=dict(model);identifier=copy.pop('model_id',None)
    if digest(copy)!=identifier:raise ValueError('model checksum mismatch')
    if signature_hash(model['timing_signature'])!=model['timing_signature_sha256']:raise ValueError('model timing signature mismatch')

def validate_case(case):
    for key in ['tokens','experts','topk','K','M','expert_bytes']:
        if type(case.get(key)) is not int or case[key]<=0:raise ValueError('invalid '+key)
    E,T,k=case['experts'],case['tokens'],case['topk'];counts=case.get('histogram')
    if E>8192 or T>65536 or k>E or k>64:raise ValueError('shape bounds')
    if not isinstance(counts,list) or len(counts)!=E or any(type(n) is not int or n<0 or n>T for n in counts):raise ValueError('invalid histogram')
    if sum(counts)!=T*k:raise ValueError('histogram sum mismatch')
    active=sum(n>0 for n in counts);runs=0;size=0
    for e,n in enumerate(counts):
        if not n:continue
        size+=case['expert_bytes']
        if e==0 or not counts[e-1]:runs+=1
        if (e==E-1 or not counts[e+1]) and e<E-1:size+=min(case['expert_bytes'],512)
    if case.get('active_experts',active)!=active or case.get('transfer_runs',runs)!=runs or case.get('submitted_weight_bytes',size)!=size:raise ValueError('selective transfer metadata mismatch')
    if case.get('quant') not in ['iq2_s','iq3_xxs','iq3_s','mxfp4'] or case.get('direction') not in ['up','down']:raise ValueError('unsupported operator')
    return dict(case,active_experts=active,transfer_runs=runs,submitted_weight_bytes=size)

def key(case):
    return '|'.join(map(str,[case['quant'],case['direction'],case['K'],case['M'],case['topk'],case['expert_bytes'],'T>=64']))

def features(case,target):
    c=validate_case(case);h=[n for n in c['histogram'] if n];active=len(h);total=sum(h)
    if target=='cpu_gpu_boundary_ms':
        if c['quant']=='mxfp4':
            names=['constant','active','groups4','tails2','tails1','assignments']
            values=[1,active,sum(n//4 for n in h),sum((n%4)//2 for n in h),sum(n%2 for n in h),total]
        else:
            names=['constant','active','small_rows','tiled_experts','tiles16','assignments']
            values=[1,active,sum(n for n in h if n<8),sum(n>=8 for n in h),sum((n+15)//16 for n in h if n>=8),total]
    else:
        # Disjoint work terms permit BLAS to have a different per-row slope after its setup charge.
        names=['constant','active','small_rows','large_rows','large_experts','route_scan_work']
        values=[1,active,sum(n for n in h if n<64),sum(n for n in h if n>=64),sum(n>=64 for n in h),c['experts']*total]
        if target=='gpu_stream_ms':names+=['transfer_runs'];values+=[c['transfer_runs']]
    return names,[float(x) for x in values]

def matrix_rank(rows,tol=1e-8):
    a=[list(r) for r in rows];rank=0
    for col in range(len(a[0])):
        pivot=max(range(rank,len(a)),key=lambda i:abs(a[i][col]),default=None)
        if pivot is None or abs(a[pivot][col])<tol:continue
        a[rank],a[pivot]=a[pivot],a[rank];v=a[rank][col];a[rank]=[x/v for x in a[rank]]
        for i in range(rank+1,len(a)):
            q=a[i][col];a[i]=[x-q*y for x,y in zip(a[i],a[rank])]
        rank+=1
        if rank==len(a):break
    return rank

def fit_nnls(X,y,ridge=RIDGE):
    if len(X)!=len(y) or len(X)<2 or not X or not X[0]:raise ValueError('bad regression dimensions')
    p=len(X[0]);n=len(X)
    if any(len(r)!=p or any(not math.isfinite(v) or v<0 for v in r) for r in X) or any(not math.isfinite(v) or v<=0 for v in y):raise ValueError('invalid measurement')
    scales=[math.sqrt(sum(r[j]*r[j] for r in X)/n) or 1 for j in range(p)]
    A=[[r[j]/scales[j] for j in range(p)] for r in X]
    norms=[sum(r[j]*r[j] for r in A)/n for j in range(p)]
    penalties=[0 if j==0 else ridge for j in range(p)]
    beta=[0.0]*p;pred=[0.0]*n;yrms=math.sqrt(sum(t*t for t in y)/n)
    converged=False;violation=float('inf')
    for iteration in range(20000):
        for j in range(p):
            denom=norms[j]+penalties[j]
            if denom==0:continue
            numer=sum(A[i][j]*(y[i]-pred[i]+A[i][j]*beta[j]) for i in range(n))/n
            value=max(0,numer/denom);delta=value-beta[j];beta[j]=value
            for i in range(n):pred[i]+=delta*A[i][j]
        if iteration%10==0:
            gradients=[sum(A[i][j]*(pred[i]-y[i]) for i in range(n))/n+penalties[j]*beta[j] for j in range(p)]
            violation=max(abs(g) if b>1e-12 else max(0,-g) for g,b in zip(gradients,beta))
            if violation<=1e-7*max(1,yrms):converged=True;break
    if not converged:raise ValueError('NNLS did not converge')
    return {'coefficients':[b/s for b,s in zip(beta,scales)],'scales':scales,'ridge':ridge,'iterations':iteration+1,
            'kkt_violation_scaled':violation,'normalized_design_rank':matrix_rank(A),'columns':p,
            'coefficient_identifiability':'ridge prediction; correlated columns retained, coefficients not causal'}

def value(regression,x):return max(0,sum(a*b for a,b in zip(regression['coefficients'],x)))

def feature_support(X,scales):
    def distance(a,b):return math.sqrt(sum(((x-y)/s)**2 for x,y,s in zip(a[1:],b[1:],scales[1:]))/max(1,len(scales)-1))
    nearest=[min(distance(row,other) for j,other in enumerate(X) if i!=j) for i,row in enumerate(X)]
    return {'min':[min(r[j] for r in X) for j in range(len(scales))],
            'max':[max(r[j] for r in X) for j in range(len(scales))],'training_vectors':X,
            'max_distance':1.5*max(nearest),'training_leave_one_out_nearest_max':max(nearest)}

def check_feature_support(reg,x):
    limits=reg['feature_support']
    if len(x)!=len(reg['features']):return 'feature_schema_mismatch'
    if any(v<lo-1e-9 or v>hi+1e-9 for v,lo,hi in zip(x,limits['min'],limits['max'])):return 'feature_range_extrapolation'
    nearest=min(math.sqrt(sum(((a-b)/s)**2 for a,b,s in zip(x[1:],r[1:],reg['scales'][1:]))/max(1,len(x)-1)) for r in limits['training_vectors'])
    if nearest>limits['max_distance']+1e-9:return 'sparse_feature_support'
    return None

def support(train,case):
    if case['experts'] not in train['declared_experts']:return 'unmeasured_declared_expert_count'
    if not train['tokens_min']<=case['tokens']<=train['tokens_max']:return 'token_count_outside_calibration'
    if not train['active_min']<=case['active_experts']<=train['active_max']:return 'active_count_outside_calibration'
    if max(case['histogram'])>train['max_expert_rows']:return 'expert_load_outside_calibration'
    return None

def train_model(records,plan):
    if not records:raise ValueError('no data')
    signatures={r['signature_sha256'] for r in records}
    if len(signatures)!=1:raise ValueError('incompatible provenance')
    expected_records(plan,records,{'train','calibration'})
    groups={};case_definitions={}
    for r in records:
        check_record(r)
        name=r['case']['name']
        if name in case_definitions and case_definitions[name]!=r['case']:raise ValueError('operator case definition changed')
        case_definitions[name]=r['case']
        if r['workload']['split']=='test':raise ValueError('heldout test cannot fit model')
        c=validate_case(r['result']);groups.setdefault(key(c),[]).append(r)
    model={'schema':VERSION,'mode':'shadow_only','timing_signature_sha256':next(iter(signatures)),
           'implementation_sha256':IMPLEMENTATION_SHA256,'study_plan':plan,'study_plan_sha256':digest(plan),'case_definitions':case_definitions,
           'timing_signature':records[0]['signature'],'strata':{},'method':'nonnegative ridge + empirical multiplicative error envelope',
           'training_record_hashes':sorted(digest(r) for r in records),'split_forecasts':'unsupported: no concurrent interference calibration'}
    for k,rows in groups.items():
        train=[r for r in rows if r['workload']['split']=='train'];cal=[r for r in rows if r['workload']['split']=='calibration']
        if len(train)<MIN_TRAIN or len(cal)<MIN_CALIBRATION:raise ValueError('insufficient data in '+k)
        cases=[validate_case(r['result']) for r in train]
        limits={'declared_experts':sorted({c['experts'] for c in cases}),'tokens_min':min(c['tokens'] for c in cases),
                'tokens_max':max(c['tokens'] for c in cases),'active_min':min(c['active_experts'] for c in cases),
                'active_max':max(c['active_experts'] for c in cases),'max_expert_rows':max(max(c['histogram']) for c in cases)}
        entry={'support':limits,'targets':{},'train_count':len(train),'calibration_count':len(cal)}
        for target in TARGETS:
            names,X0=features(cases[0],target);X=[features(c,target)[1] for c in cases]
            y=[statistics.median(r['result'][target]) for r in train];reg=fit_nnls(X,y)
            reg.update(features=names,feature_support=feature_support(X,reg['scales']))
            below=above=1.0;noise=0.0;details=[];supported_calibration=0
            for r in cal:
                c=validate_case(r['result']);why=support(limits,c)
                why=why or check_feature_support(reg,features(c,target)[1])
                predicted=value(reg,features(c,target)[1]);actual=statistics.median(r['result'][target])
                if min(actual,predicted)<=0:raise ValueError('nonpositive prediction')
                if why is None:
                    supported_calibration+=1
                    below=max(below,predicted/actual);above=max(above,actual/predicted)
                    noise=max(noise,statistics.median(abs(t-actual) for t in r['result'][target]))
                details.append({'workload':r['workload']['id'],'predicted_ms':predicted,'actual_ms':actual,'support_reason':why})
            entry['targets'][target]={**reg,'supported_calibration_count':supported_calibration,'error_envelope':{'lower_divisor':below,'upper_multiplier':above,
                'timing_MAD_floor_ms':noise,'basis':'maximum errors on separate calibration workloads; not a confidence interval'},'calibration':details}
        model['strata'][k]=entry
    model['model_id']=digest(model)
    return model

def advise(model,event):
    verify_model(model)
    current=event.get('current_assignment')
    out={'schema':1,'mode':'shadow_only','current_assignment':current,'assignment_after':current,'would_change_assignment':False,
         'hypothetical_choice':None,'model_id':model.get('model_id'),'predictions':{},'abstain_reasons':[],
         'split_forecast':None,'split_status':model['split_forecasts']}
    def abstain(reason):out['abstain_reasons'].append(reason);return out
    if current not in ['cpu','gpu']:return abstain('unknown_current_assignment')
    if event.get('timing_signature_sha256')!=model['timing_signature_sha256']:return abstain('provenance_or_boundary_mismatch')
    if event.get('feature_stage')!='host_routes_ready' or event.get('decision_stage')!='post_assignment_shadow':return abstain('features_not_available_at_declared_decision')
    if event.get('unique_routes_validated') is not True:return abstain('route_uniqueness_unverified')
    if event.get('native_moe_cache') is not False:return abstain('unmodeled_expert_cache')
    if event.get('host_weights_ready') is not True:return abstain('CPU_host_weights_not_verified_available')
    for backend in ['cpu','gpu']:
        q=event.get(backend+'_queue_ms')
        if type(q) not in [int,float] or not math.isfinite(q) or q<0:return abstain('queue_state_unobserved')
        if q!=0:return abstain('nonempty_queue_unvalidated')
    try:c=validate_case(event['operator'])
    except (ValueError,KeyError,TypeError):return abstain('invalid_operator_features')
    if c['tokens']<64:return abstain('unmodeled_small_batch_or_decode')
    entry=model['strata'].get(key(c))
    if entry is None:return abstain('unmeasured_operator_stratum')
    why=support(entry['support'],c)
    if why:return abstain(why)
    if type(event.get('gpu_weight_resident')) is not bool:return abstain('unknown_weight_residency')
    gpu_target='gpu_resident_ms' if event['gpu_weight_resident'] else 'gpu_stream_ms'
    for backend,target in [('cpu','cpu_gpu_boundary_ms'),('gpu',gpu_target)]:
        reg=entry['targets'][target];names,x=features(c,target)
        if names!=reg['features']:return abstain('feature_schema_mismatch')
        why=check_feature_support(reg,x)
        if why:return abstain(backend+':'+why)
        if reg['supported_calibration_count']<MIN_CALIBRATION:return abstain(backend+':insufficient_supported_error_calibration')
    for backend,target in [('cpu','cpu_gpu_boundary_ms'),('gpu',gpu_target)]:
        reg=entry['targets'][target];p=value(reg,features(c,target)[1]);band=reg['error_envelope'];noise=band['timing_MAD_floor_ms']
        out['predictions'][backend]={'service_ms':p,'lower_ms':max(0,p/band['lower_divisor']-noise),'upper_ms':p*band['upper_multiplier']+noise,
                                     'scope':model['timing_signature']['scope'],'band_kind':'empirical error envelope'}
    cpu,gpu=out['predictions']['cpu'],out['predictions']['gpu']
    if cpu['upper_ms']<gpu['lower_ms']:out['hypothetical_choice']='cpu'
    elif gpu['upper_ms']<cpu['lower_ms']:out['hypothetical_choice']='gpu'
    else:return abstain('empirical_error_envelopes_overlap')
    return out

def synthetic_event(record):
    c=record['result'];use_gpu=c['tokens']>=(1024 if c['quant']=='mxfp4' else 512)
    return {'schema':1,'origin':'synthetic_measured_fixture','timing_signature_sha256':record['signature_sha256'],
      'current_assignment':'gpu' if use_gpu else 'cpu','current_assignment_basis':'existing fixed-threshold rule, not a production routing observation',
      'feature_stage':'host_routes_ready','decision_stage':'post_assignment_shadow','unique_routes_validated':True,
      'native_moe_cache':False,'host_weights_ready':True,'gpu_weight_resident':False,'cpu_queue_ms':0,'gpu_queue_ms':0,'operator':c}

def evaluate(model,records):
    verify_model(model);expected_records(model['study_plan'],records,{'test'});rows=[];target_errors={}
    for r in records:
        if r['workload']['split']!='test' or not r['ok']:raise ValueError('expected successful heldout record')
        check_record(r)
        if r['case']!=model['case_definitions'].get(r['case']['name']):raise ValueError('heldout operator definition changed')
        if r['signature_sha256']!=model['timing_signature_sha256']:raise ValueError('heldout provenance mismatch')
        c=validate_case(r['result']);entry=model['strata'].get(key(c))
        if entry is None:raise ValueError('missing fitted stratum')
        for target in TARGETS:
            reg=entry['targets'][target];x=features(c,target)[1];why=support(entry['support'],c) or check_feature_support(reg,x)
            prediction=value(reg,x);actual=statistics.median(r['result'][target]);band=reg['error_envelope']
            lo=max(0,prediction/band['lower_divisor']-band['timing_MAD_floor_ms']);hi=prediction*band['upper_multiplier']+band['timing_MAD_floor_ms']
            target_errors.setdefault(key(c)+'|'+target,[]).append({'workload':r['workload']['id'],'support_reason':why,
                'absolute_ms':abs(prediction-actual),'relative':abs(prediction-actual)/actual,'actual_ms':actual,'predicted_ms':prediction,
                'inside_empirical_envelope':lo<=actual<=hi,'supported_error_calibration_count':reg['supported_calibration_count']})
        event=synthetic_event(r);a=advise(model,event)
        actual={'cpu':statistics.median(r['result']['cpu_gpu_boundary_ms']),'gpu':statistics.median(r['result']['gpu_stream_ms'])}
        chosen=a['hypothetical_choice'] or event['current_assignment'];oracle=min(actual.values())
        errors={b:{'absolute_ms':abs(p['service_ms']-actual[b]),'relative':abs(p['service_ms']-actual[b])/actual[b],
                   'underprediction_ms':max(0,actual[b]-p['service_ms']),'actual_ms':actual[b],'predicted_ms':p['service_ms'],
                   'inside_empirical_envelope':p['lower_ms']<=actual[b]<=p['upper_ms']} for b,p in a['predictions'].items()}
        rows.append({'case':r['case']['name'],'workload':r['workload']['id'],'advice':a,'errors':errors,'measured_cpu_ms':actual['cpu'],
          'measured_gpu_ms':actual['gpu'],'chosen_for_evaluation':chosen,'measured_choice_regret_ms':actual[chosen]-oracle,
          'fixed_policy_ms':actual[event['current_assignment']],'shadow_choice_measured_ms':actual[chosen]})
    all_errors=[x for r in rows for x in r['errors'].values()];rel=sorted(x['relative'] for x in all_errors)
    return {'model_id':model['model_id'],'heldout_cases':len(rows),'advised_cases':sum(r['advice']['hypothetical_choice'] is not None for r in rows),
      'prediction_coverage':sum(bool(r['errors']) for r in rows)/len(rows),
      'median_relative_error':statistics.median(rel) if rel else None,'p90_relative_error':rel[max(0,math.ceil(.9*len(rel))-1)] if rel else None,
      'mean_absolute_error_ms':statistics.mean(x['absolute_ms'] for x in all_errors) if all_errors else None,
      'worst_underprediction_ms':max((x['underprediction_ms'] for x in all_errors),default=None),
      'mean_choice_regret_ms':statistics.mean(r['measured_choice_regret_ms'] for r in rows),
      'fixed_policy_sum_ms':sum(r['fixed_policy_ms'] for r in rows),'shadow_hypothetical_sum_ms':sum(r['shadow_choice_measured_ms'] for r in rows),
      'scope':'Heldout standalone workloads; summed time is not a model/request speedup. Abstention retains the fixed baseline.',
      'per_stratum_target':target_errors,'rows':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['fit','evaluate','advise']);ap.add_argument('--input',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args();root=Path(__file__).resolve().parent;mp=root/'MODEL.json'
    if args.command=='fit':
        if (root/'EVALUATION_LOCK.json').exists() or mp.exists():raise SystemExit('Model is frozen; use a new study for refitting')
        records=[json.loads(p.read_text()) for p in sorted((root/'data').glob('*/RESULT.json'))]
        model=train_model(records,json.loads((root/'STUDY.json').read_text()));mp.write_text(json.dumps(model,indent=2)+'\n');print('Frozen model',model['model_id']);return
    model=json.loads(mp.read_text())
    verify_model(model)
    if args.command=='evaluate':
        records=[json.loads(p.read_text()) for p in sorted((root/'data').glob('*--test-*/RESULT.json'))]
        lock=json.loads((root/'EVALUATION_LOCK.json').read_text());h=hashlib.sha256(mp.read_bytes()).hexdigest()
        if lock['model_sha256']!=h or any(r.get('frozen_model_sha256')!=h for r in records):raise SystemExit('Evaluation model hash mismatch')
        result=evaluate(model,records)
    else:
        if not args.input:raise SystemExit('--input required')
        result=advise(model,json.loads(args.input.read_text()))
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text)
if __name__=='__main__':main()
