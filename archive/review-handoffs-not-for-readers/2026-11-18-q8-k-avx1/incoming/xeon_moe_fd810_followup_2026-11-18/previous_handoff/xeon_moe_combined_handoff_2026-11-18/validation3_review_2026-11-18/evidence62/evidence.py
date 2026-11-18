"""Read a validation3 archive. Usage: python evidence.py PATH_TO_ARCHIVE > numeric_evidence.txt."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
def read(p):
    return json.loads(p.read_bytes())
def digest(data):
    return hashlib.sha256(data).hexdigest()
def filehash(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def put(key, value):
    print(key, json.dumps(value, ensure_ascii=False))
def check(cond, detail):
    if not cond:
        raise ValueError(detail)
def scalar_corr(x, k):
    return round(float(np.corrcoef(x[k:], x[:-k])[0, 1]), 4)
def result_lines(p):
    return [i for i,s in enumerate(p.read_text().splitlines(), 1) if any(s.startswith('      "'+key+'":') for key in ('start_ns','output_ready_ns','before_T1','after','custom_builds_after_each_T1'))]

manifest = read(root/'FILES_SHA256.json')
for name, meta in manifest.items():
    p = root/name
    check(p.stat().st_size == meta['bytes'] and filehash(p) == meta['sha256'], p)
mapping = read(root/'RAW_MAPPING.json')
check(len(mapping) == len(manifest)-1 and {x['shared'] for x in mapping} == manifest.keys()-{'RAW_MAPPING.json'}, 'mapping inventory')
for e in mapping:
    check(e['byte_identical'] and manifest[e['shared']]['sha256']==e['sha256'], e['shared'])
put('publication', {'manifest_entries':len(manifest),'bytes':sum(x['bytes'] for x in manifest.values()),'mapping_entries':len(mapping)})
check(read(root/'python-repeat/DECISION.json')['initial_timing_analysis_sha256']==filehash(root/'TIMING_ANALYSIS.json'),'repeat link')
builds=[]
provider_sets=[]
quality = defaultdict(dict)
timings=defaultdict(lambda:defaultdict(list))
hits=defaultdict(list)
answers=defaultdict(dict)
deltas=defaultdict(list)
run_hashes=0
normal_configs=set()
for label, base in [('initial',root),('repeat',root/'python-repeat')]:
    caller=base/'caller'
    state=read(base/'STATE.json')
    records=state['runs']
    check(state['completed'] and len(records)==(10 if label=='initial' else 4), 'run count')
    plan=read(caller/'FROZEN_PLAN.json')
    cases={x['case_id']:x['execution_input'] for x in plan['cases']}
    design={x['name']:x for x in read(caller/'DESIGN.json')['stages']}
    build=read(caller/'BUILD.json')
    check(build['returncode']==0, 'build status')
    for item, sha in build['source_sha256'].items():
        check(filehash(caller/item)==sha, item)
    builds.append(build)
    bank=(caller/'capacity.plan').read_text().splitlines()
    selected={int(row.split()[0]) for row in bank[1:] if int(row.split()[4])}
    check(len(selected)==23 and sum(int(row.split()[4]) for row in bank[1:])==35,'plan')
    for name, entry in records.items():
        d=caller/'runs'/Path(entry['path']).name
        rpt=read(d/'RUN.json'); stage=rpt['stage']
        spec=read(d/'RUN_SPEC.json'); guard=read(d/'guard-health.json'); launch=read(d/'LAUNCH_MANIFEST.json')
        normal_configs.add(launch['baseline_config_sha256'])
        check(rpt['passed'] and guard['status']=='exited' and guard['child_returncode']==0==entry['returncode'] and entry['status']=='passed',d)
        check(entry['manifest_sha256']==filehash(d/'RESULTS_SHA256.json'),'manifest')
        for rel,sha in read(d/'RESULTS_SHA256.json').items():
            if rel=='server-config.json':
                actual=digest((json.dumps({k:launch[k] for k in ('command','environment')}, indent=2)+'\n').encode())
            else:
                actual=filehash(d/('guard-health.json' if rel=='health.json' else rel))
            check(actual==sha, (d,rel));run_hashes+=1
        check(design[name]==spec==launch['spec']==rpt['spec'] and (spec['stage'],spec['mode'])==(stage,rpt['arm']), name)
        for key,p in [('runner_sha256',caller/'run_validation.py'),('build_sha256',d/'BUILD.json'),('run_spec_sha256',d/'RUN_SPEC.json'),('plan_sha256',caller/'FROZEN_PLAN.json'),('bank_plan_sha256',caller/'capacity.plan')]:
            check(filehash(p)==launch[key],key)
        check(read(d/'BUILD.json')==build,'BUILD')
        pins=read(d/'PROVIDER_EXPECTED.json')
        for key in ['providers','active_backend_providers']:
            check(pins[key]==rpt[key],key)
        active_backends = [row['name'] for row in rpt['active_backend_providers']]
        check(active_backends == (['CUDA0', 'CPU'] if rpt['arm']=='F' else ['CUDA0', 'XEON_HOT_FFN', 'CPU']), 'arm-dependent scheduler')
        provider_sets.append(rpt['providers'])
        check(rpt['placement']==[{'layer':i,'host':i<37} for i in range(40)],'placement')
        check(rpt['cparams']['n_ubatch']==1024 and rpt['cparams']['threads']==12 and rpt['cparams']['threads_batch']==24 and rpt['cparams']['n_outputs_max']==1,'config')
        for c in rpt['cases']:
            ids=cases[c['case_id']]
            for r,k in [('prefix_ids','prefix_ids'),('teacher_forced_input64','teacher_forced_input64'),('next_targets65','teacher_forced_next_targets65')]:
                check(c[r]==ids[k], k)
            for k,src in [('prefix_i32_sha256','prefix_ids'),('joined_i32_sha256','joined_expected_ids')]:
                check(c[k]==digest(np.array(ids[src],dtype='<i4').tobytes()), k)
        prep=rpt['preparation']
        ja=cases['text_plan_constraints_ja']
        check(prep['count']==4 and prep['timed'] is False and prep['prefix_ids']==ja['prefix_ids'] and prep['T1_inputs']==ja['teacher_forced_input64'][:4], 'prep')
        def hist(before,after,n):
            if rpt['arm']=='F':
                check(before is None and after is None,'F counters');return None
            for key,v in [('native_bytes',475453440),('bank_allocation_bytes',475457024)]:
                check(before[key]==after[key]==v,'bank bytes')
            h=np.subtract(after['executed_H'],before['executed_H'])
            dl=np.subtract(after['calls'],before['calls'])
            check(h.shape==(40,7) and (h>=0).all() and np.array_equal(h.sum(1),dl) and dl.tolist()==[n if lay in selected else 0 for lay in range(40)],'H layer totals')
            return h
        hist(prep['before'],prep['after'],4)
        for b in rpt['blocks']:
            c=b['case_id']; mode=b['mode']; n=b['prefix_count']; ci=cases[c]
            check(b['prefix_ids']==ci['prefix_ids'] and n==len(b['prefix_ids']) and b['input_tokens']==b['next_targets'][:64] and b['next_targets']==ci['teacher_forced_next_targets65'] and len(b['next_targets'])==65 and b['positions']==list(range(n,n+64)), 'block inputs')
            hist(b['before_prefix'],b['before_T1'],0)
            h=hist(b['before_T1'],b['after'],64)
            starts,ends=map(np.array,(b['start_ns'],b['output_ready_ns']))
            check(len(starts)==len(ends)==64 and np.all(ends>starts) and np.all(starts[1:]>=ends[:-1]) and b['status']==[0]*64,'timestamps')
            if stage=='quality':
                p=d/b['file'];a=np.fromfile(p,dtype='<f4').reshape(65,129280);check(np.isfinite(a).all(),p)
                quality[c][mode]=a
                if mode!='F':
                    x=b['custom_builds_after_each_T1']
                    ds=[(pos,v) for i,(pos,v) in enumerate(zip(b['positions'],x)) if i==0 or v!=x[i-1]]
                    put('quality_build_increments', [c,mode,b['before_T1']['custom_nodes_built'],ds])
            elif stage=='timing':
                check(not b['quality'] and b['custom_builds_after_each_T1'] is None,'timing boundaries')
                ms=(ends-starts)/1e6
                timings[label][c].append((spec['process_index'],mode,ms))
                if h is not None:hits[(label,c)].append(h)
                where=int(ms.argmax())
                row={'series':label,'case':c,'block':spec['process_index'],'mode':mode,'first_ms':float(ms[0]),'all_mean_ms':float(ms.mean()),'median_ms':float(np.median(ms)),'p95_ms':float(np.quantile(ms,.95)),'max_ms':float(ms[where]),'max_position':n+where,'pos256_ms':float(ms[256-n]) if n<=256<n+64 else None,'odd_minus_even_index_mean_ms':float(ms[1::2].mean()-ms[::2].mean()),'lag1':scalar_corr(ms,1),'lag2':scalar_corr(ms,2)}
                put('timing_block',row);deltas[label].append(row)
        if stage=='timing' or stage=='quality':
            put('raw_keys_lines',[str((d/'RUN.json').relative_to(root)),result_lines(d/'RUN.json')])
        for a in rpt['answers']:
            c=a['case_id'];mode=a['mode'];ids=a['generated_ids'];n=len(ids)
            hist(a['before_prefix'],a['before_T1'],0);hist(a['before_T1'],a['after'],n-1)
            check(n==a['generated_count'] and n-1==a['decoded_T1_calls'] and not a['reference_supplied'],'answer count')
            journal=[json.loads(line) for line in (d/a['events_file']).read_text().splitlines()]
            samples=[e for e in journal if e['kind']=='sample']
            decodes=[e for e in journal if e['kind']=='decode']
            check([s['token_id'] for s in samples]==ids and [s['ordinal'] for s in samples]==list(range(n)) and [e['input_token_id'] for e in decodes]==ids[:-1] and all(e['status']==0 for e in decodes),'answer ids')
            for piece,file in [('piece_all_hex','all_text_file'),('piece_visible_hex','visible_text_file')]:
                check(b''.join(bytes.fromhex(e[piece]) for e in samples)==(d/a[file]).read_bytes(),file)
            check((a['stop_reason']=='eog' and samples[-1]['is_eog']) or (a['stop_reason']=='length' and n==1024 and not samples[-1]['is_eog']), 'termination')
            answers[c][mode]=ids
            put('answer',[c,mode,n,a['stop_reason']])
check(all(x==provider_sets[0] for x in provider_sets), 'consistent providers across runs')
check(builds[0]['driver_sha256']==builds[1]['driver_sha256'] and builds[0]['engine_libraries']==builds[1]['engine_libraries'] and builds[1]['compiled_parent_BUILD_sha256']==filehash(root/'caller/BUILD.json'),'cross-build reuse')
restore=read(root/'RESTORED_RUNTIME.json')
check(restore['normal_API_restored'] and not restore['diagnostic_libraries_loaded'] and restore['API_health']=='ok' and normal_configs=={restore['config_sha256']}, 'restore receipt')
put('run_integrity',{'runs':len(provider_sets),'manifest_hashes_checked':run_hashes,'configs_reconstructed_in_memory':len(provider_sets),'provider_symbols':len(provider_sets[0]),'backend_pins_F':2,'backend_pins_SP':3,'driver_sha256':builds[0]['driver_sha256'],'restore_receipt_pid':restore['pid']})
for c,m in quality.items():
    f,s,p=[m[k] for k in ('F','S','P')]
    target=cases[c]['teacher_forced_next_targets65']
    eq=(f.view('<u4')==p.view('<u4')).all(axis=1)
    check((p.view('<u4')==s.view('<u4')).all(), 'S/P bit match')
    def nll(row,index):
        x=row.astype('float64');peak=x.max()
        return peak-x[index]+np.log(np.exp(x-peak).sum())
    diff=np.array([nll(z,t)-nll(x,t) for x,z,t in zip(f,p,target)])
    ids_f=f.argmax(axis=1);ids_p=p.argmax(axis=1)
    differences=[(int(i),int(ids_f[i]),int(ids_p[i]),target[i]) for i in np.flatnonzero(ids_f!=ids_p)]
    put('quality',{'case':c,'S_P_equal_rows':65,'F_P_equal_rows':np.flatnonzero(eq).tolist(),'top1_F_P_target':differences,'max_abs':float(np.abs(p.astype(float)-f).max()),'mean_target_NLL_P_minus_F':diff.mean(),'logits_scalars_FSP':3*f.size})
for label,sets in timings.items():
    total=0
    for c,blocks in sets.items():
        blocks.sort(key=lambda b:b[0]);check([b[1] for b in blocks]==list('FPPF'), 'ABBA')
        v=[b[2] for b in blocks];total+=sum(len(x) for x in v)
        f=np.r_[v[0],v[3]];p=np.r_[v[1],v[2]]
        put('speed',{'series':label,'case':c,'calls':len(f)+len(p),'F_T1_s':1000/f.mean(),'P_T1_s':1000/p.mean(),'P_gain_pct':100*(f.sum()/p.sum()-1),'saved_ms':f.mean()-p.mean(),'F_last_minus_first_ms':v[3].mean()-v[0].mean(),'pair_ratios':[v[0].mean()/v[1].mean(),v[3].mean()/v[2].mean()]})
        h1,h2=hits[label,c];check(np.array_equal(h1,h2), 'repeat H differs')
        h=h1.sum(axis=0);hit=int(h@np.arange(7))
        put('hits',{'series':label,'case':c,'H0_to_H6':h.tolist(),'hot_picks':hit,'hot_pct_37_layers':100*hit/(37*64*6),'H0_pct_path_calls':100*h[0]/h.sum()})
    put('timed_calls',[label,total])
for label,rows in deltas.items():
    for k in ['odd_minus_even_index_mean_ms','lag1','lag2','first_ms','max_ms']:
        nums=[r[k] for r in rows];put('block_ranges',[label,k,min(nums),max(nums)])
for c,m in answers.items():
    first=next(i for i,(f,p) in enumerate(zip(m['F'],m['P'])) if f!=p)
    put('first_free_answer_divergence',[c,first])
