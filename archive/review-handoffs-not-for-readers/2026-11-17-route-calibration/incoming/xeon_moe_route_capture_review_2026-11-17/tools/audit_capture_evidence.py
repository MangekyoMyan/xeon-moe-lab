#!/usr/bin/env python3
"""Independent saved-data audit. Never execute supplied code, load ELF, or access hardware.

The caller provides a pinned repository and a new output directory outside it.
Only Python's standard library is used; all numeric outputs are descriptive.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

COMMIT = 'caf35bcdd27c0240572e05a60633f07fca272844'
CAPTURE = 'archive/review-handoffs-not-for-readers/2026-11-17-route-capture'
INTEGRATION = 'archive/review-handoffs-not-for-readers/2026-11-17-route-integration'
SYMBOLS = ['llama_decode','llama_get_logits_ith','ggml_backend_cpu_init',
    'ggml_backend_sched_graph_compute_async','xeon_route37_open_v1','xeon_route37_matches_v1',
    'xeon_route37_label_v1','xeon_route37_begin_eval_v1','xeon_route37_end_eval_v1',
    'xeon_route37_complete_call_v1','xeon_route37_close_v1','xeon_route37_record_v1',
    'xeon_route37_stats_v1','xeon_route37_read_record_v1','xeon_route37_read_attempt_v1',
    'xeon_route37_read_call_v1']

def need(test, message):
    if not test: raise AssertionError(message)

def sha(data): return hashlib.sha256(data).hexdigest()
def load(path): return json.loads(path.read_text())

def frequencies(rows):
    answer=[]
    for layer in range(37):
        layer_rows=[r for r in rows if r['layer']==layer]
        sets=[set(r['ids']) for r in layer_rows]
        counter=Counter(e for r in layer_rows for e in r['ids'])
        overlap=Counter(len(a&b) for a,b in zip(sets,sets[1:]))
        answer.append({'layer':layer,'unique_experts':len(counter),
            'unique_unordered_top6_sets':len({tuple(sorted(x)) for x in sets}),
            'adjacent_overlap_0_to_6':[overlap[i] for i in range(7)],
            'frequencies':[[e,n] for e,n in sorted(counter.items(),key=lambda x:(-x[1],x[0]))]})
    return answer

def audit_one(root, saved, pinned_providers, actual_provider_checks, expected_library_hashes):
    run=load(root/'RUN.json');smoke=load(root/'SMOKE4.json')
    launch=load(root/'LAUNCH_MANIFEST.json');guard=load(root/'guard-health.json')
    n=run['planned_capture_calls'];trace=run['trace'];stats=trace['stats']
    need(n in (4,68) and n==launch['planned_calls'],'planned calls')
    need(launch['continue64']==(n==68),'mode')
    need(run['schema']==1 and run['passed'] is True and run['model_routes'] is True,'final result')
    need(run['session_id']==root.name and run['session_closed_before_owner_destruction'] is True,'session lifecycle')
    need(run['prefix_tokens']==64 and len(run['tokens'])==64+n,'token prefix')
    need(all(type(x) is int and x>=0 for x in run['tokens']),'token IDs')
    need(run['vision_weights_loaded'] is True,'vision allocation recorded')
    need(run['placement']==[dict(layer=i,expert_weights_host=(i<37)) for i in range(40)],'weight placement')
    need(run['providers']==pinned_providers,'all provider fields match prepared fixture')
    need([x['symbol'] for x in run['providers']]==SYMBOLS,'all 16 symbols')
    need(len(run['active_backend_providers'])==2,'two active backend providers')
    need([x['name'] for x in run['active_backend_providers']]==['CUDA0','CPU'],'backend names')
    need([x['device_type'] for x in run['active_backend_providers']]==[1,0],'backend types')
    need(run['active_backend_providers'][1]['graph_compute_provider']==run['providers'][2]['path'],'same CPU provider')
    need(len(actual_provider_checks)==18,'16 function and 2 backend checks')
    for provider,check in zip(run['providers'],actual_provider_checks[:16]):
        need(check['symbol']==provider['symbol'] and check['path']==provider['path'],'symbol/path receipt')
        need(check['recorded_inode_size_matches'] is True,'inode/size receipt')
        need(check['sha256']==expected_library_hashes[check['path']],'pinned function backing hash')
    for provider,check in zip(run['active_backend_providers'],actual_provider_checks[16:]):
        need(check['backend']==provider['name'] and check['path']==provider['graph_compute_provider'],'active backend receipt')
        need(check['sha256']==expected_library_hashes[check['path']],'pinned backend hash')
    expected_stats=dict(records=37*n,attempts=n,calls=n,dropped_events=0,
        complete_decode_attempts=n,flags=0,enabled=False,observing=True)
    need(stats==expected_stats,'complete stats fields')
    need(len(trace['records'])==37*n and len(trace['attempts'])==len(trace['calls'])==len(run['call_results'])==n,'actual ledger sizes')
    for i,(attempt,call,result) in enumerate(zip(trace['attempts'],trace['calls'],run['call_results'])):
        label=dict(request=0,sequence=0,decode_call=i,phase=2,logical_input_token=i)
        expected_attempt=dict(label=label,id=i+1,generation=1,first_record=37*i,records=37,
            layers=(1<<37)-1,position=64+i,context=1,ubatch=0,n_tokens=1,flags=0,graph_status=0,
            observing=True,scope_t1=True,graph_returned=True,caller_completed=True)
        expected_call=dict(label=label,first_attempt=i,attempts=1,flags=0,public_or_driver_status=0,
            synchronized=True,completed=True)
        expected_result=dict(logical_input_token=i,position=64+i,label_accepted=True,
            decode_returned=True,decode_status=0,output_called=True,output_valid=True,
            synchronized=True,drain_called=False,finalize_called=True,finalize_ok=True,
            exception_stage=0,ok=True)
        need(attempt==expected_attempt,f'attempt all fields {i}')
        need(call==expected_call,f'call all fields {i}')
        need(result==expected_result,f'call result all fields {i}')
        records=trace['records'][37*i:37*(i+1)]
        need([r['layer'] for r in records]==list(range(37)),f'layer order/coverage {i}')
        for layer,record in enumerate(records):
            ids=record['ids']
            need(len(ids)==len(set(ids))==6 and all(type(e) is int and 0<=e<384 for e in ids),'joint6 ID domain/uniqueness')
            expected_record=dict(ordinal=37*i+layer,attempt_id=i+1,request=0,sequence=0,
                decode_call=i,ubatch=0,logical_input_token=i,position=64+i,generation=1,
                context=1,layer=layer,phase=2,projection=1,n_tokens=1,ids=ids,flags=0)
            need(record==expected_record,f'record all fields {i}/{layer}')
    need(smoke['passed'] is False and smoke['trace']['stats']['enabled'] is True,'smoke is pre-close snapshot')
    for key,length in [('records',148),('attempts',4),('calls',4)]:
        need(smoke['trace'][key]==trace[key][:length],f'smoke {key} equality')
    need(smoke['call_results']==run['call_results'][:4],'smoke call results')
    need(smoke['trace']['stats']==dict(records=148,attempts=4,calls=4,dropped_events=0,
        complete_decode_attempts=4,flags=0,enabled=True,observing=True),'smoke stats')
    stdout=(root/'server.stdout.log').read_text().splitlines()
    need(len(stdout)==2 and stdout[0]=='route37: 4-call coverage passed (148 records)','stdout smoke')
    output=load_string(stdout[1])
    need(output['passed'] is True and output['calls']==n and output['records']==37*n,'stdout final')
    need(output['output'].endswith('/'+root.name+'/RUN.json'),'stdout output path')
    raw=(root/'server.stderr.log').read_bytes();stderr=raw.decode('utf-8',errors='replace');lines=stderr.splitlines()
    failures=[{'line':i+1,'text':line} for i,line in enumerate(lines) if re.search(r'\b(error|fatal|failed|stop_requested)\b',line,re.I)]
    need(not failures,'no logged failure markers')
    timing_lines=[{'line':i+1,'text':line} for i,line in enumerate(lines) if re.search(r'tokens per second|tokens/s|eval time|load time|ms per token|tok/s',line,re.I)]
    buffer_lines=[{'line':i+1,'text':line} for i,line in enumerate(lines)
        if re.search(r'KV buffer size|compute buffer size|n_ctx\s*=|n_seq_max\s*=|n_vocab\s*=',line)]
    need(guard['status']=='exited' and guard['child_returncode']==0,'child successful exit')
    need(not(root/'STOP_REQUESTED').exists(),'no exported stop marker')
    samples=guard['recent_samples'];need(len(samples)==30,'retained samples count')
    need(all(guard['started_unix']<=s['unix']<=guard['finished_unix'] for s in samples),'sample time window')
    need(guard['gpu_monitor']['timeout_count']==0,'GPU query timeouts')
    need(guard['limits']['fan_percent']==70 and launch['RAM_reserve_GiB']==64,'guard policy')
    need(all(s['gpu']['limit_W']==150 and s['gpu']['ecc']=='Enabled' for s in samples),'sample power/ECC')
    sampled=dict(retained_samples=len(samples),gpu_peak_C=max(s['gpu']['gpu_C'] for s in samples),
        hbm_peak_C=max(s['gpu']['hbm_C'] for s in samples),
        vram_peak_MiB=max(s['gpu']['vram_used_MiB'] for s in samples),
        power_peak_W=max(s['gpu']['power_W'] for s in samples),
        cpu_peak_C=max(v for s in samples for v in s['cpu_C'].values()),
        available_min_GiB=min(s['available_kib']/1024**2 for s in samples),
        fan_peak_percent=max(v for s in samples for v in s['fans']['percent']),
        gpu_query_timeouts=guard['gpu_monitor']['timeout_count'])
    need(sampled['available_min_GiB']>=64,'saved RAM reserve samples')
    per_layer=frequencies(trace['records'])
    need(per_layer==saved['per_layer'],'independently recomputed per-layer statistics')
    need(sampled==saved['sampled_guard'],'saved sample aggregates')
    need(saved['graph_generations']==[1] and saved['calls']==n and saved['records']==37*n,'saved run summary')
    need(saved['expert_selections']==n*37*6,'selection count')
    need(saved['process_elapsed_seconds']==guard['finished_unix']-guard['started_unix'],'whole-process interval')
    unexported_raw=[]
    for name,digest in saved['raw_sha256'].items():
        name='guard-health.json' if name=='health.json' else name
        if name=='server-config.json':
            need(not(root/name).exists(),'private server config intentionally not exported')
            unexported_raw.append(dict(name=name,recorded_sha256=digest,independently_recomputed=False))
            continue
        need(sha((root/name).read_bytes())==digest,'saved raw digest '+name)
    return run,launch,dict(run=root.name,calls=n,records=n*37,expert_selection_slots=n*37*6,
        graph_generations=[1],position_range=[64,63+n],all_ledger_fields_checked=True,
        all_call_results_successful=True,finite_logit_check_scope='source and boolean receipt; no logits bytes or hashes saved',
        smoke_is_preclose_not_failure=True,session_closed=True,
        provider_functions=16,active_backends=2,providers_equal_prepared_record=True,
        intentionally_unexported_raw_hash_receipts=unexported_raw,
        guard_status=guard['status'],child_returncode=guard['child_returncode'],
        whole_process_elapsed_s_not_decode_latency=guard['finished_unix']-guard['started_unix'],
        sampled_guard=sampled,sample_first_unix=samples[0]['unix'],sample_last_unix=samples[-1]['unix'],
        sample_coverage_s=samples[-1]['unix']-samples[0]['unix'],
        stderr=dict(bytes=len(raw),lines=len(lines),utf8_replacement_characters=stderr.count('\ufffd'),
            failure_marker_lines=failures,throughput_or_eval_timing_lines=timing_lines,buffer_lines=buffer_lines),
        per_layer=per_layer)

def load_string(s): return json.loads(s)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=args.repo.resolve();out=args.output.resolve()
    need(not out.is_relative_to(repo),'output outside repository')
    out.mkdir(parents=True,exist_ok=True)
    n=repo/CAPTURE;i=repo/INTEGRATION
    manifest=load(n/'FILES_SHA256.json')
    actual={p.relative_to(n).as_posix() for p in n.rglob('*') if p.is_file()}
    need(actual==set(manifest)|{'FILES_SHA256.json'},'complete exported set')
    sources={}
    for rel in sorted(actual):
        p=n/rel;need(not p.is_symlink() and p.resolve().is_relative_to(n.resolve()),'safe manifest member')
        data=p.read_bytes();entry=dict(bytes=len(data),sha256=sha(data));sources[CAPTURE+'/'+rel]=entry
        if rel!='FILES_SHA256.json':need(entry==manifest[rel],'member bytes/hash '+rel)
    for mapping in load(n/'RAW_MAPPING.json'):
        rel=mapping['shared'];need(rel.startswith('runs/'),'raw mapping member')
        need(mapping['byte_identical'] is True and sha((n/rel).read_bytes())==mapping['sha256'],'raw mapping hash')
        want='guard-health.json' if Path(mapping['original']).name=='health.json' else Path(mapping['original']).name
        need(Path(rel).name==want,'raw mapping rename')
    same=['driver.cpp','route_client.h','call_control.h','CLIENT_BUILD.json','LINK_BUILD.json','run_capture.py']
    for f in same: need((n/f).read_bytes()==(i/f).read_bytes(),'integration source identical '+f)
    links=load(n/'LINK_BUILD.json');clients=load(n/'CLIENT_BUILD.json')
    for name,digest in clients['source_sha256'].items():
        p=n/name if (n/name).exists() else i/name
        need(sha(p.read_bytes())==digest,'client source hash '+name)
        sources[str(p.relative_to(repo))]=dict(bytes=p.stat().st_size,sha256=digest)
    for name,digest in links['sources'].items():
        p=i/'route37'/name;need(sha(p.read_bytes())==digest,'linked source hash '+name)
        sources[str(p.relative_to(repo))]=dict(bytes=p.stat().st_size,sha256=digest)
    for name,expected in links['model_sources'].items():
        p=i/'provenance/model-source'/name;data=p.read_bytes()
        need(dict(bytes=len(data),sha256=sha(data))==expected,'actual model source manifest '+name)
        sources[str(p.relative_to(repo))]=expected
    for fname,key in [('unity_1_cxx.cxx','sha256'),('unity_1_cxx.cxx.o.d','dependency_sha256')]:
        p=i/'provenance'/fname;need(sha(p.read_bytes())==links['model_Unity'][key],'Unity evidence hash')
        sources[str(p.relative_to(repo))]=dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
    need(links['model_Unity']['object_sha256']==links['reused_objects'][links['model_Unity']['object']]['sha256'],'Unity object receipt correspondence')
    need(links['baseline_relink_byte_identical'] is True and links['collector_defined_only_in_CPU'] is True,'baseline link receipt')
    prepared=load(i/'PROVIDERS_FINAL.json')['providers']
    provider_check=load(n/'ACTUAL_PROVIDER_VERIFICATION.json')
    need(provider_check['passed'] is True and len(provider_check['runs'])==2,'provider verification receipt')
    expected_library_hashes={v['path']:v['sha256'] for v in links['runtime_libraries'].values()}
    expected_library_hashes[prepared[0]['path']]=links['diagnostic_llama_sha256']
    saved=load(n/'ACTUAL_CAPTURE_VERIFICATION.json')
    need(saved['verification_source_sha256']==sha((n/'verify_capture.py').read_bytes()),'original verifier source hash')
    need(len(saved['runs'])==2,'two saved verification runs')
    runs=[];launches=[];summaries=[]
    for root in sorted((n/'runs').iterdir()):
        saved_run=next(x for x in saved['runs'] if x['run']==root.name)
        checks=next(x['checks'] for x in provider_check['runs'] if x['run']==root.name)
        run,launch,summary=audit_one(root,saved_run,prepared,checks,expected_library_hashes)
        need(launch['driver_sha256']==clients['binaries']['driver'],'driver digest receipt')
        need(launch['diagnostic_llama_sha256']==links['diagnostic_llama_sha256'],'diagnostic link digest')
        runs.append(run);launches.append(launch);summaries.append(summary)
    a,b=runs
    need(a['planned_capture_calls']==4 and b['planned_capture_calls']==68,'run ordering')
    need(a['tokens']==b['tokens'][:68],'shared prefix and first 4 input tokens')
    for k,length in [('records',148),('attempts',4),('calls',4)]:need(a['trace'][k]==b['trace'][k][:length],'cross-run all fields '+k)
    need(a['call_results']==b['call_results'][:4],'cross-run output-valid and completion results')
    need(a['providers']==b['providers'] and a['active_backend_providers']==b['active_backend_providers'] and a['placement']==b['placement'],'cross-run providers/placement')
    launch_a={k:v for k,v in launches[0].items() if k not in ('continue64','planned_calls')}
    launch_b={k:v for k,v in launches[1].items() if k not in ('continue64','planned_calls')}
    need(launch_a==launch_b,'launch differences confined to planned call count')

    tokens=b['tokens'][64:];records=b['trace']['records']
    periods=[p for p in range(1,len(tokens)//2+1) if all(tokens[j]==tokens[j-p] for j in range(p,len(tokens)))]
    need(periods==[22] and len(set(tokens))==20,'fixed input period')
    by={(r['logical_input_token'],r['layer']):r['ids'] for r in records}
    lags=[]
    for lag in [1,2,11,22,44,66]:
        pairs=[(by[(t,l)],by[(t+lag,l)]) for t in range(68-lag) for l in range(37)]
        hist=Counter(len(set(x)&set(y)) for x,y in pairs)
        lags.append(dict(lag_input_tokens=lag,layer_pairs=len(pairs),
            common_experts_0_to_6=[hist[k] for k in range(7)],
            mean_common_experts=sum(k*v for k,v in hist.items())/len(pairs),
            ordered_joint6_equal=sum(x==y for x,y in pairs),unordered_joint6_equal=hist[6]))
    need(lags[0]['common_experts_0_to_6']==[361,568,660,553,267,63,7],'published adjacent histogram')
    allfreq=Counter((r['layer'],e) for r in records for e in r['ids'])
    unique=[x['unique_experts'] for x in summaries[1]['per_layer']]
    need(min(unique)==31 and max(unique)==101 and statistics.median(unique)==60,'unique expert extrema/median')
    stats=dict(scope='Descriptive statistics of one repeated-English teacher-forced sequence; not bank evaluation or performance prediction',
        continuation_input_tokens=68,unique_continuation_token_IDs=20,minimum_exact_continuation_period_tokens=22,
        unique_token_frequencies=[[t,c] for t,c in sorted(Counter(tokens).items(),key=lambda x:(-x[1],x[0]))],
        observed_positions=[64,131],prefix_tokens=64,independent_semantic_workload_families=1,
        raw_run_count=2,unique_input_prefix_trajectory_count=1,duplicated_first_four_calls_not_added_to_sample_count=True,
        per_layer_unique_experts=unique,min_unique_experts=min(unique),median_unique_experts=statistics.median(unique),max_unique_experts=max(unique),
        lag_overlap=lags,top_layer_experts=[dict(layer=l,expert=e,selections=c,call_fraction=c/68)
            for (l,e),c in sorted(allfreq.items(),key=lambda x:(-x[1],x[0]))[:30]],
        selected_in_every_capture_call=[dict(layer=l,expert=e) for (l,e),c in sorted(allfreq.items()) if c==68],
        validation_limits=['Token shuffling cannot create independent workload holdout from a 22-token periodic text.',
            'A contiguous first/second split remains the same repeated passage and growing KV trajectory.',
            'No Japanese or other-language workload is present; language invariance/difference is untested.',
            'Joint6 overlap is not cache hit; there is no deployed candidate GPU bank or ready/overlap measurement.',
            'No sampled generated continuation or task correctness target is stored.'])
    restored=load(n/'RESTORED_RUNTIME.json')
    need(restored['API_health']=='ok' and restored['normal_API_restored'] is True,'normal API restore receipt')
    need(restored['diagnostic_libraries_loaded'] is False and restored['normal_runtime_backing_files_verified'] is True,'normal restored libraries receipt')
    need(restored['config_sha256']==launches[0]['baseline_config_sha256'],'restored config hash')
    need(all(x=='active' for x in restored['services'].values()),'restored services')
    need(not restored['monitor_errors'] and not restored['active_alerts'],'restore no alerts receipt')
    expected_restore_hashes={'libggml-cpu.so.0.28.0':'33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3',
        'libllama.so.0.8.0':links['baseline_relink_sha256'],
        'libggml-cuda.so.0.28.0':links['runtime_libraries']['libggml-cuda.so.0.28.0']['sha256'],
        'libggml-base.so.0.28.0':links['runtime_libraries']['libggml-base.so.0.28.0']['sha256']}
    need({Path(p).name:h for p,h in restored['mapped_runtime_hashes'].items()}==expected_restore_hashes,'restored baseline library hashes')
    report=dict(schema=1,pinned_commit=COMMIT,scope='Independent saved-evidence/data audit, no supplied program/model/GPU execution',
        all_independent_checks_passed=True,manifest_entries=len(manifest),total_exported_files=len(actual),
        raw_mapping_entries=12,source_copies_identical_to_integration=same,
        source_hashes_checked=True,model_source_Unity_correspondence='exported source/include/dependency hashes and object receipts agree; not full rebuild',
        binary_hashes_independently_recomputed=False,actual_remote_provider_inodes_independently_rechecked=False,
        runs=summaries,cross_run_first4=dict(input_tokens_equal=True,all_ordered_record_fields_equal=True,
            attempts_equal=True,calls_equal=True,call_result_fields_equal=True,provider_fields_equal=True),
        route_statistics=stats,restored_runtime=dict(pid=restored['pid'],config_hash_matches_launch=True,
            normal_library_hash_receipts_match=True,diagnostic_libraries_loaded=False,
            services_active=True,monitor_age_seconds=restored['unix']-restored['monitor_timestamp'],
            independent_live_process_check=False),
        important_limits=['The captured phase2 label is explicit teacher-forced continuation, not actual sampled output.',
            'Generation is 1 throughout; actual model rebuild/abort/rollback/failure paths are not covered.',
            'Finiteness is asserted by driver and saved as boolean; logits bytes/hashes and cross-backend numeric errors are not present.',
            'Per-call latency is absent; full-process durations include load, warmup, prefill, growing JSON snapshots, validation, I/O and cleanup.',
            'Saved GPU VRAM value is used-memory samples, not a fresh free/reserved or cudaMemGetInfo admission result.',
            'Last30 periodic health samples cannot prove continuous peaks or continuous idle/hardware state.',
            'Restored runtime is an execution-side receipt, not a new remote process inspection by this review.'],
        script_sha256=sha(Path(__file__).read_bytes()))
    (out/'CAPTURE_EVIDENCE_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'ROUTE_PERIODICITY_AND_FREQUENCIES.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    (out/'CAPTURE_SOURCE_SHA256.json').write_text(json.dumps(dict(schema=1,pinned_commit=COMMIT,files=sources),indent=2)+'\n')
    print(json.dumps(dict(passed=True,files=len(actual),runs=[dict(calls=x['calls'],records=x['records']) for x in summaries],
        period=22,unique_continuation_tokens=20,lag1_mean=lags[0]['mean_common_experts'],lag22_mean=lags[3]['mean_common_experts'],
        output=str(out)),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
