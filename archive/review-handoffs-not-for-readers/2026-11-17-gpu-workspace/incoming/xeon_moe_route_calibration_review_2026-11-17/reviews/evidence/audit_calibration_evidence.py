#!/usr/bin/env python3
"""Independent read-only audit of the pinned route-calibration archive.

Only Python standard-library parsing is used. No supplied script, executable,
shared library, model, GPU command, API, or network operation is run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

PIN = 'fa6671f7b5281bad28947fe5324e2495e792a86f'
REL = Path('archive/review-handoffs-not-for-readers/2026-11-17-route-calibration')
STAMPS = ['step_begin_ns','decode_begin_ns','decode_return_ns','output_ready_ns',
          'validation_end_ns','finalize_end_ns','step_end_ns']

def need(value, message):
    if not value:
        raise AssertionError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def load(path):
    return json.loads(path.read_bytes())

def summary(xs):
    ordered = sorted(xs)
    index = (len(ordered)-1)*0.95
    lo, hi = math.floor(index), math.ceil(index)
    return dict(n=len(xs),mean=statistics.fmean(xs),median=statistics.median(xs),
                p95=ordered[lo]*(1-(index-lo))+ordered[hi]*(index-lo),
                minimum=ordered[0],maximum=ordered[-1])

def close_numbers(actual, expected, path=''):
    if isinstance(actual,dict):
        need(actual.keys()==expected.keys(),'JSON keys '+path)
        for key in actual:
            close_numbers(actual[key],expected[key],path+'/'+key)
    elif isinstance(actual,list):
        need(len(actual)==len(expected),'JSON length '+path)
        for index,(a,b) in enumerate(zip(actual,expected)):
            close_numbers(a,b,path+'/'+str(index))
    elif isinstance(actual,float):
        need(math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-8),'numeric mismatch '+path)
    else:
        need(actual==expected,'value mismatch '+path)

def check_ledger(trace,n,observing):
    expected = dict(records=37*n if observing else 0,attempts=n,calls=n,dropped_events=0,
                    complete_decode_attempts=n if observing else 0,flags=0,enabled=False,observing=observing)
    need(trace['stats']==expected,'complete stats')
    need(len(trace['records'])==expected['records'],'actual records')
    need(len(trace['attempts'])==len(trace['calls'])==n,'actual ledger count')
    generations=set()
    for i,(a,c) in enumerate(zip(trace['attempts'],trace['calls'])):
        label=dict(request=0,sequence=0,decode_call=i,phase=2,logical_input_token=i)
        generation=a['generation'];generations.add(generation)
        expected_a=dict(label=label,id=i+1,generation=generation,first_record=37*i if observing else 0,
                        records=37 if observing else 0,layers=(1<<37)-1 if observing else 0,
                        position=64+i,context=1,ubatch=0,n_tokens=1,flags=0,graph_status=0,
                        observing=observing,scope_t1=True,graph_returned=True,caller_completed=True)
        need(a==expected_a,'attempt all fields')
        need(c==dict(label=label,first_attempt=i,attempts=1,flags=0,public_or_driver_status=0,
                     synchronized=True,completed=True),'call all fields')
        if not observing:
            continue
        records=trace['records'][37*i:37*(i+1)]
        need([r['layer'] for r in records]==list(range(37)),'all layer order')
        for layer,r in enumerate(records):
            ids=r['ids']
            need(len(ids)==len(set(ids))==6 and all(type(x) is int and 0<=x<384 for x in ids),'IDs')
            expected_r=dict(ordinal=37*i+layer,attempt_id=i+1,request=0,sequence=0,decode_call=i,
                            ubatch=0,logical_input_token=i,position=64+i,generation=generation,
                            context=1,layer=layer,phase=2,projection=1,n_tokens=1,ids=ids,flags=0)
            need(r==expected_r,'record all fields')
    return sorted(generations)

def check_block(block,meta,vocab):
    n=meta['calls'];mode=meta['mode'];numeric=meta['numerical']
    for key in ('name','mode','case_id','calls'):
        need(block[key]==meta[key],'block metadata '+key)
    need(meta['passed'] is True,'block passed')
    need(block['prefix_tokens']==64 and block['position_first']==64 and block['position_last']==63+n,'positions')
    collect=mode in ('D1','D2');observe=mode=='D2'
    need(block['collector_session']==collect and block['observing']==observe,'mode collection')
    validation='all-vocab finite and saved F32' if numeric else 'synchronized pointer nonnull only; no full finite scan or logits copy'
    need(block['output_validation']==validation,'explicit validation scope')
    need(len(block['steps'])==n,'step count')
    spans={k:[] for k in ['engine_us','label_prepare_us','decode_return_us','output_sync_us',
                         'validation_us','finalize_us','post_us','step_us','next_step_interval_us']}
    end_previous=None
    for i,s in enumerate(block['steps']):
        need({k:v for k,v in s.items() if k not in STAMPS}==dict(index=i,position=64+i,decode_status=0,
             synchronized=True,collector_finalize_called=collect,ok=True),'step metadata all fields')
        stamps=[s[key] for key in STAMPS]
        need(all(type(x) is int for x in stamps) and stamps==sorted(stamps),'integer monotonic timestamps')
        if end_previous is not None:
            need(stamps[0]>=end_previous,'cross-step monotonic order')
            spans['next_step_interval_us'].append((stamps[0]-block['steps'][i-1]['step_begin_ns'])/1000)
        end_previous=stamps[-1]
        for key,a,b in [('engine_us',1,3),('label_prepare_us',0,1),('decode_return_us',1,2),
                        ('output_sync_us',2,3),('validation_us',3,4),('finalize_us',4,5),
                        ('post_us',5,6),('step_us',0,6)]:
            spans[key].append((stamps[b]-stamps[a])/1000)
    need(block['prefill_begin_ns']<=block['prefill_ready_ns']<=block['session_setup_begin_ns']
         <=block['session_setup_end_ns']<=block['steps'][0]['step_begin_ns'],'prefill/setup/block order')
    need(end_previous<=block['block_end_ns'],'export after block boundary')
    item=dict(name=block['name'],mode=mode,case_id=block['case_id'],numerical=numeric,calls=n,
              interval_statistics={k:summary(v) for k,v in spans.items() if v},engine_each_us=spans['engine_us'],
              first_engine_us=spans['engine_us'][0],engine_after_first_us=summary(spans['engine_us'][1:]),
              block_wall_us=(block['block_end_ns']-block['steps'][0]['step_begin_ns'])/1000,
              prefill_us=(block['prefill_ready_ns']-block['prefill_begin_ns'])/1000,
              setup_us=(block['session_setup_end_ns']-block['session_setup_begin_ns'])/1000,
              output_validation=block['output_validation'])
    generations=[]
    if collect:
        generations=check_ledger(block['trace'],n,observe)
        item['records']=len(block['trace']['records'])
        if observe:
            item['layer_unique_experts']=[len({e for r in block['trace']['records'] if r['layer']==l for e in r['ids']}) for l in range(37)]
    else:
        need('trace' not in block,'F/D0 must not have active ledger')
    return item,generations

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();repo=args.repo.resolve();out=args.output.resolve();base=repo/REL
    need(not out.is_relative_to(repo),'output outside repo')
    out.mkdir(parents=True,exist_ok=True)
    public=load(base/'FILES_SHA256.json')
    actual={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    need(actual==set(public)|{'FILES_SHA256.json'},'complete public file inventory')
    source_hashes={}
    for rel,receipt in public.items():
        p=base/rel
        need(not p.is_symlink() and p.resolve().is_relative_to(base.resolve()),'manifest member')
        data=p.read_bytes();need(receipt==dict(bytes=len(data),sha256=digest(data)),'public hash '+rel)
        source_hashes[str(REL/rel)]=receipt
    mapping=load(base/'RAW_MAPPING.json')
    for m in mapping:
        need(m['byte_identical'] is True and digest((base/m['shared']).read_bytes())==m['sha256'],'raw mapping')
    incoming=base/'incoming/xeon_moe_route_capture_review_2026-11-17'
    manifest=load(incoming/'FILES_SHA256.json');need(len(manifest)==25,'incoming count')
    for rel,x in manifest.items():
        data=(incoming/rel).read_bytes();need(x==dict(bytes=len(data),sha256=digest(data)),'incoming '+rel)
    intake=load(base/'INTAKE.json')
    need(intake['sha256']=='e0f5c399d9b9c439941df7df7701cbe39ec729058e5e11d9af8a713927dde3af','incoming ZIP receipt')
    original_zip=repo.parent/'deliverables/route_capture_review_2026-11-17/xeon_moe_route_capture_review_2026-11-17.zip'
    original_zip_local_match=None
    if original_zip.exists():
        original_zip_local_match=digest(original_zip.read_bytes())==intake['sha256'];need(original_zip_local_match,'prior delivered ZIP')
    build=load(base/'BUILD.json');need(build['returncode']==0,'final build return')
    for name,expected in build['source_sha256'].items():
        need(digest((base/name).read_bytes())==expected,'build source pin '+name)
    need(not(base/'BUILD.stdout.log').read_bytes() and not(base/'BUILD.stderr.log').read_bytes(),'empty final build logs')
    old_capture=repo/REL.parent/'2026-11-17-route-capture'
    old_link=load(old_capture/'LINK_BUILD.json');old_restore=load(old_capture/'RESTORED_RUNTIME.json')
    for name,info in build['engine_libraries']['frozen'].items():
        need(old_restore['mapped_runtime_hashes'][info['path']]==info['sha256'],'frozen libraries unchanged receipt')
    for name,info in old_link['runtime_libraries'].items():
        need(build['engine_libraries']['diagnostic'][name]==info,'diagnostic libraries unchanged receipt')
    need(build['engine_libraries']['diagnostic']['libllama.so.0.8.0']['sha256']==old_link['diagnostic_llama_sha256'],'diagnostic llama unchanged receipt')
    final=load(base/'FINAL_ANALYSIS.json')
    need(final['analysis_source_sha256']==digest((base/'analyze.py').read_bytes()),'analyzer source identity')
    design=load(base/'DESIGN.json');cases=load(base/'CASES.json')['cases']
    incoming_cases=load(incoming/'plans/CALIBRATION_TEXT_EXAMPLES.json')['cases']
    need([(x['case_id'],x['text_utf8'],x['sha256_utf8']) for x in cases[1:]]==
         [(x['case_id'],x['text_utf8'],x['sha256_utf8']) for x in incoming_cases],'six original calibration texts')
    for c in cases:
        need(digest(c['text_utf8'].encode())==c['sha256_utf8'],'text digest')
    run_summaries=[];all_blocks=[];blobs=[];generations=set();guards=[];raw_by_arm={};cross_cases=None;run_hashes=0
    for directory in sorted((base/'runs').iterdir()):
        r=load(directory/'RUN.json');arm=r['arm'];raw_by_arm[arm]=r
        h=load(directory/'guard-health.json');launch=load(directory/'LAUNCH_MANIFEST.json')
        need(r['schema']==1 and r['passed'] is True and r['session_closed_before_owner_destruction'] is True,'completed run')
        need(h['status']=='exited' and h['child_returncode']==0 and not(directory/'STOP_REQUESTED').exists(),'successful guard exit')
        need(h['gpu_monitor']['timeout_count']==0 and h['gpu_monitor']['recovered_queries']==0,'queries')
        need(launch['arm']==arm and launch['reserve_GiB']==64 and launch['budget_seconds']==7200,'launch scope')
        need(launch['API_automatically_stopped_or_restarted'] is False,'launcher service policy')
        need(launch['build_sha256']==digest((base/'BUILD.json').read_bytes()) and (directory/'BUILD.json').read_bytes()==(base/'BUILD.json').read_bytes(),'same final caller build')
        need(launch['runner_sha256']==digest((base/'run_case.py').read_bytes()),'runner source pin')
        for name,expected in load(directory/'RESULTS_SHA256.json').items():
            if name=='server-config.json':
                data=(json.dumps(dict(command=launch['command'],environment=launch['environment']),indent=2)+'\n').encode()
                need(not(directory/name).exists(),'excluded config')
            else:
                data=(directory/('guard-health.json' if name=='health.json' else name)).read_bytes()
            need(digest(data)==expected,'per-run raw manifest '+name);run_hashes+=1
        pin=load(directory/'PROVIDER_EXPECTED.json')
        need(pin['model_loaded'] is False and pin['arm']==arm,'preflight provider scope')
        need(r['providers']==pin['providers'] and r['active_backend_providers']==pin['active_backend_providers'],'actual provider identity all fields')
        kind='diagnostic' if arm=='D' else 'frozen';libs=build['engine_libraries'][kind]
        need(len(r['providers'])==(16 if arm=='D' else 4),'provider count')
        for p in r['providers']:
            name='libllama.so.0.8.0' if p['symbol'].startswith('llama_') else 'libggml-base.so.0.28.0' if p['symbol'].startswith('ggml_backend_sched_') else 'libggml-cpu.so.0.28.0'
            need(p['path']==libs[name]['path'] and p['bytes']>0 and p['inode']>0,'provider path receipt')
        need(r['active_backend_providers']==[
            dict(name='CUDA0',graph_compute_provider=libs['libggml-cuda.so.0.28.0']['path']),
            dict(name='CPU',graph_compute_provider=libs['libggml-cpu.so.0.28.0']['path'])],'active backend providers')
        expected_cp=dict(n_ctx=131072,n_batch=1024,n_ubatch=1024,n_seq_max=1,n_outputs_max=1,
                         n_outputs_max_per_seq=1,n_threads=12,n_threads_batch=24,backend_sampling=False)
        need(r['cparams']==expected_cp,'actual cparams')
        need(r['placement']==[dict(layer=i,expert_weights_host=i<37) for i in range(40)],'expert weight placement')
        need(r['vision_weights_loaded'] is True and r['vocab_size']==129280,'vision/vocabulary')
        need(r['tokenizer']==dict(input_form='raw text; no chat template applied',add_special=True,parse_special=True),'tokenizer scope')
        if cross_cases is None:cross_cases=r['cases']
        need(r['cases']==cross_cases,'all case token IDs equal across processes')
        need(len(r['cases'])==len(cases)==7,'case count')
        case_out=[]
        for original,record in zip(cases,r['cases']):
            need((original['case_id'],original['family_id'],original['sha256_utf8'])==(record['case_id'],record['family_id'],record['text_sha256']),'text/token correspondence')
            toks=record['full_tokens'];need(len(toks)==record['full_token_count'] and len(toks)>=128,'token length')
            need(all(type(t)is int and 0<=t<129280 for t in toks),'token range')
            need(record['used_prefix_tokens']==record['maximum_used_continuation']==64,'token windows')
            case_out.append(dict(case_id=record['case_id'],text_sha256=record['text_sha256'],full_token_count=len(toks),token_ids_native_i32_sha256=digest(struct.pack('<'+'i'*len(toks),*toks))))
        old_end=None;computed=[]
        for meta in r['blocks']:
            block=load(directory/meta['result_file'])
            need(old_end is None or block['prefill_begin_ns']>old_end,'block monotonic order')
            old_end=block['block_end_ns']
            item,gens=check_block(block,meta,r['vocab_size']);generations.update(gens)
            if meta['numerical']:
                data=(directory/block['logits_file']).read_bytes()
                need(len(data)==4*4*129280 and block['logits_shape']==[4,129280],'full logits size/shape')
                need(block['logits_type']=='native x86 little-endian F32','logits format')
                vals=struct.iter_unpack('<f',data);need(all(math.isfinite(v[0]) for v in vals),'all saved logits finite')
                blobs.append(data)
                item.update(logits_sha256=digest(data),logits_scalars=4*129280,frozen_reference_byte_equal=block.get('frozen_reference_byte_equal'))
                if arm!='F-before':need(block['frozen_reference_byte_equal'] is True,'saved reference equality')
            computed.append(item);all_blocks.append(dict(run=directory.name,arm=arm,**item))
        saved_run=next(x for x in final['runs'] if x['run']==directory.name)
        close_numbers(computed,saved_run['blocks'],directory.name+'/blocks')
        need(case_out==saved_run['cases'],'token digest computations')
        if arm=='D':
            need([b['mode'] for b in r['blocks'][:3]]==['D0','D1','D2'],'numeric mode order')
            need([b['mode'] for b in r['blocks'][3:9]]==design['timing_order']['D'],'timing order')
            need([b['case_id'] for b in r['blocks'][9:]]==[c['case_id'] for c in cases[1:]],'all screen fixed order')
        else:
            need([b['name'] for b in r['blocks']]==['correct_F','timing_F_0','timing_F_1'],'F blocks')
        samples=h['recent_samples'];need(len(samples)==30,'guard tail sample count')
        need(all(h['started_unix']<=s['unix']<=h['finished_unix'] for s in samples),'sample time windows')
        need(all(s['gpu']['limit_W']==150 and s['gpu']['ecc']=='Enabled' for s in samples),'saved power/ECC')
        g=dict(run=directory.name,gpu_max_C=max(s['gpu']['gpu_C'] for s in samples),hbm_max_C=max(s['gpu']['hbm_C'] for s in samples),
               cpu_max_C=max(v for s in samples for v in s['cpu_C'].values()),fan_max_percent=max(v for s in samples for v in s['fans']['percent']),
               RAM_available_min_GiB=min(s['available_kib']/1024**2 for s in samples),VRAM_used_max_MiB=max(s['gpu']['vram_used_MiB'] for s in samples),
               sample_span_seconds=samples[-1]['unix']-samples[0]['unix'],process_elapsed_seconds=h['finished_unix']-h['started_unix'])
        need(g['RAM_available_min_GiB']>=64 and g['fan_max_percent']<=70,'sampled bounds')
        guards.append(g)
        log_bytes=(directory/'server.stderr.log').read_bytes();log=log_bytes.decode('utf-8',errors='replace')
        need('n_outputs_max         = 1' in log and 'CUDA0 compute buffer size =  4669.26 MiB' in log,'log context/workspace')
        failures=[dict(line=i+1,text=line) for i,line in enumerate(log.splitlines()) if re.search(r'\b(error|fatal|failed|stop_requested)\b',line,re.I)]
        need(not failures,'logged failure markers')
        stdout=(directory/'server.stdout.log').read_text().splitlines()
        need(stdout==['completed '+b['name']+' calls='+str(b['calls']) for b in r['blocks']]+['study passed '+arm],'all block stdout')
        run_summaries.append(dict(run=directory.name,arm=arm,blocks=len(computed),T1_calls=sum(b['calls'] for b in computed),
                                 cparams=r['cparams'],memory_after_warmup=r['memory_after_warmup'],
                                 provider_function_count=len(r['providers']),providers_match_all_saved_fields=True,
                                 stderr=dict(bytes=len(log_bytes),lines=len(log.splitlines()),utf8_replacements=log.count('\ufffd'),failure_markers=failures),cases=case_out))
    need([r['arm'] for r in run_summaries]==design['process_order'],'process run order')
    need(len(blobs)==5 and all(b==blobs[0] for b in blobs),'all five F32 files exactly equal')
    need(digest(blobs[0])=='95430744558e7414b336f6548a6c03e064e12e1adc03cb8256fc45dcfc3187b0','expected F32 digest')
    need(len(all_blocks)==21 and sum(b['calls'] for b in all_blocks)==1044,'total block/call count')
    need(sum(b.get('records',0) for b in all_blocks)==19092,'total active route records')
    screen=[b for b in all_blocks if b['name'].startswith('screen_')]
    need(len(screen)==6 and sum(b['records'] for b in screen)==14208,'screen records')
    timings=[b for b in all_blocks if b['name'].startswith('timing_')]
    by_mode={mode:[b['interval_statistics']['engine_us']['mean'] for b in timings if b['mode']==mode] for mode in ['F','D0','D1','D2']}
    close_numbers(by_mode,final['timing_block_means_us'],'timing block means')
    means={mode:statistics.fmean(values) for mode,values in by_mode.items()}
    ratios=dict(D0_over_F_percent=(means['D0']/means['F']-1)*100,D1_over_D0_percent=(means['D1']/means['D0']-1)*100,
                D2_over_D1_percent=(means['D2']/means['D1']-1)*100,F_max_min_block_percent=(max(by_mode['F'])/min(by_mode['F'])-1)*100)
    d_root=next(p for p in (base/'runs').iterdir() if '-D-' in p.name)
    d_correct=load(d_root/'correct_D2.json')['trace']
    d_timing0=load(d_root/'timing_2_D2.json')['trace'];d_timing1=load(d_root/'timing_3_D2.json')['trace']
    repeated_equal=d_timing0==d_timing1
    correct_prefix=all(d_correct[k]==d_timing0[k][:len(d_correct[k])] for k in ('records','attempts','calls'))
    old_raw=load(old_capture/'runs/20261117T001035Z-68c12e67/RUN.json')['trace']
    old_route_prefix=all(d_timing0[k]==old_raw[k][:len(d_timing0[k])] for k in ('records','attempts','calls'))
    restored=load(base/'RESTORED_RUNTIME.json');baseline=load(base/'BASELINE_API_RECEIPT.json')
    need(restored['API_health']=='ok' and restored['normal_API_restored'] and not restored['diagnostic_libraries_loaded'],'normal restored receipt')
    need(restored['normal_runtime_backing_files_verified'],'backing receipt')
    need(all(v=='active' for v in restored['services'].values()) and not restored['monitor_errors'] and not restored['active_alerts'],'restored services/monitor')
    need(restored['config_sha256']==baseline['config_sha256']==old_restore['config_sha256'],'baseline config unchanged receipt')
    need(restored['mapped_runtime_hashes']==old_restore['mapped_runtime_hashes'],'normal runtime unchanged receipt')
    result=dict(schema=1,pinned_commit=PIN,scope='Independent saved-file/source audit only; supplied programs and model/GPU/API not executed',
        all_checks_passed=True,public_manifest_entries=len(public),total_public_files=len(actual),raw_mapping_entries=len(mapping),
        run_manifest_entries_checked_or_reconstructed=run_hashes,incoming_manifest_entries=len(manifest),prior_delivered_zip_local_sha_match=original_zip_local_match,
        final_source_hashes_match=True,engine_library_receipts_match_prior_capture=True,remote_libraries_or_process_independently_checked=False,
        totals=dict(runs=3,blocks=21,T1_calls=1044,numerical_calls=20,timing_calls=640,screen_calls=384,active_route_records=19092,screen_route_records=14208),
        numeric=dict(files=5,scalars_each=517120,bytes_each=2068480,all_bytes_equal=True,all_saved_f32_finite=True,sha256=digest(blobs[0]),
                     scope='Four fixed wiring continuation inputs only; timing and language-screen logits are not saved or full-finite checked'),
        graph_generations=sorted(generations),runs=run_summaries,timing_block_means_us=by_mode,descriptive_mean_us_by_mode=means,descriptive_ratios=ratios,
        all_original_analysis_intervals_and_statistics_recomputed=True,p95_definition='linear interpolation at (n-1)*0.95; nearest-rank is a different estimator',
        blocks=all_blocks,screen_summary=[dict(case_id=b['case_id'],engine_mean_ms=b['interval_statistics']['engine_us']['mean']/1000,
            unique_experts_layer_median=statistics.median(b['layer_unique_experts']),unique_experts_layer_min=min(b['layer_unique_experts']),unique_experts_layer_max=max(b['layer_unique_experts'])) for b in screen],
        repeated_wiring_routes=dict(two_D2_timing_ledgers_all_equal=repeated_equal,correct_D2_prefix_equal=correct_prefix,old_cap1024_capture_prefix64_ledgers_equal=old_route_prefix),
        guards=guards,normal_API_restored_receipt=dict(pid=restored['pid'],config_and_library_hashes_unchanged=True,diagnostic_libraries_loaded=False,
            monitor_age_seconds=restored['unix']-restored['monitor_timestamp'],independent_live_observation=False),
        limitations=['F/D comparison is conditional on a short repeated-English fixed trajectory, sequential processes and few blocks; no equivalence confidence interval.',
            'Natural-language timing screens have one block each and fixed order; not a causal language experiment.',
            'Logits byte equality and full finite check are only for the five four-call files.',
            'Generation is 1; no new real-model fault/rebuild/rollback injection.',
            'CUDA free and NVML free are distinct APIs/phases; no positive bank allocation/admission or CPU/GPU overlap test.',
            'Guard reports retain the last 30 periodic samples, not continuous extrema.'],script_sha256=digest(Path(__file__).read_bytes()))
    (out/'CALIBRATION_EVIDENCE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'SOURCE_SHA256.json').write_text(json.dumps(dict(schema=1,pinned_commit=PIN,files=source_hashes),indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_checks_passed','public_manifest_entries','incoming_manifest_entries','totals','numeric','graph_generations','descriptive_mean_us_by_mode','descriptive_ratios','screen_summary','repeated_wiring_routes')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
