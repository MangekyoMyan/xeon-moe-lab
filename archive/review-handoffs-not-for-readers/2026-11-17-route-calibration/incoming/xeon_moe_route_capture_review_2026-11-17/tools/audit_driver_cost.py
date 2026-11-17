#!/usr/bin/env python3
"""Read-only source/receipt analysis, not an engine or timing benchmark."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve()
assert not out.is_relative_to(repo)
out.mkdir(parents=True,exist_ok=True)
n='archive/review-handoffs-not-for-readers/2026-11-17-route-capture/'
i='archive/review-handoffs-not-for-readers/2026-11-17-route-integration/'
h='archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab/'
u=h+'deepseek41-bringup-20261101/source/'
paths=[n+x for x in ['driver.cpp','route_client.h','call_control.h','run_capture.py',
    'ACTUAL_CONSULT_PRO_JA.txt','ACTUAL_RESULTS_JA.txt','README.txt','CLIENT_BUILD.json',
    'LINK_BUILD.json','RESTORED_RUNTIME.json']]
paths += [i+x for x in ['route37/route37.cpp','route37/route37.h','route37/route37-bridge.cpp',
    'route37/llama-context.cpp','route37/ggml-cpu.c','route37/caller.patch','route37/cpu.patch',
    'provenance/model-source/src/models/deepseek41.cpp']]
paths += [u+x for x in ['common/common.h','common/common.cpp','common/speculative.cpp',
    'tools/server/server-context.cpp','ggml/src/ggml-alloc.c']]
paths += [h+'gpu-stream-20261103/ggml-backend.cpp',h+'deepseek41-bringup-20261101/supervise.py']
driver=(repo/(n+'driver.cpp')).read_text();client=(repo/(n+'route_client.h')).read_text()
assert 'report["trace"]=snapshot(*api);save_json(out/"RUN.json",report);' in driver
assert 'need(coverage(report["trace"],i+1)' in driver
assert 'f<<data.dump(2)' in client
assert 'for(size_t i=0;i<s.records;++i)' in client
assert 'params.sampling.backend_sampling=false' in driver
assert 'for(int i=0;i<llama_vocab_n_tokens(vocab);++i)' in driver
reports=[]
encode=lambda obj:(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
for run in ['20261117T000246Z-d4e0df3a','20261117T001035Z-68c12e67']:
    rel=n+'runs/'+run+'/'
    for fn in ['RUN.json','SMOKE4.json','LAUNCH_MANIFEST.json','server.stderr.log']:
        paths.append(rel+fn)
    raw=(repo/(rel+'RUN.json')).read_bytes();j=json.loads(raw)
    assert encode(j)==raw,'format reconstruction not exact'
    nt=j['planned_capture_calls'];assert nt in (4,68)
    assert len(j['call_results'])==nt and len(j['trace']['records'])==37*nt
    # The source only changes these fields inside the successful capture loop.
    q=copy.deepcopy(j);q['passed']=False;q.pop('session_closed_before_owner_destruction')
    final_trace=copy.deepcopy(q['trace']);final_results=copy.deepcopy(q['call_results'])
    sizes=[];smoke_match=False
    for k in range(1,nt+1):
        q['call_results']=final_results[:k]
        for field,count in [('records',37*k),('attempts',k),('calls',k)]:
            q['trace'][field]=final_trace[field][:count]
        q['trace']['stats'].update(records=37*k,attempts=k,calls=k,
                                   complete_decode_attempts=k,enabled=True)
        data=encode(q);sizes.append(len(data))
        if k==4:
            smoke_match=data==(repo/(rel+'SMOKE4.json')).read_bytes()
            assert smoke_match,'smoke4 reconstruction not exact'
    stderr=(repo/(rel+'server.stderr.log')).read_text(errors='replace')
    for needle in ['n_vocab               = 129280','n_outputs_max         = 1024',
        'n_batch               = 1024','n_ubatch              = 1024',
        'CUDA0 compute buffer size =  4669.26 MiB']:
        assert needle in stderr,needle
    reports.append({'run':run,'calls':nt,'raw_final_RUN_bytes':len(raw),
        'final_dump_format_reconstructed_exactly':True,'SMOKE4_reconstructed_exactly':smoke_match,
        'per_call_RUN_serialized_bytes_source_reconstruction':sum(sizes),
        'per_call_RUN_serialized_bytes_in_order':sizes,
        'per_call_snapshot_record_visits':37*nt*(nt+1)//2,
        'per_call_coverage_record_visits':37*nt*(nt+1)//2,
        'final_snapshot_additional_record_visits':37*nt,
        'per_call_snapshot_attempt_visits':nt*(nt+1)//2,
        'per_call_snapshot_call_visits':nt*(nt+1)//2,
        'vocabulary_size':129280,'captured_call_finite_checks':129280*nt,
        'logits_f32_bytes_per_call':129280*4,
        'diag_n_outputs_max':1024,'diag_n_batch':1024,'diag_n_ubatch':1024,
        'diag_CUDA_compute_MiB_printed':4669.26,
        'limits':['These are source-derived loop/serialization counts, not time measurements.',
                  'Serialized/write-request bytes are not physical disk I/O bytes.',
                  'Per-call serialized total excludes startup, SMOKE4 and final RUN writes.',
                  'No normal API output reservation value has been independently observed here.']})
restored=json.loads((repo/(n+'RESTORED_RUNTIME.json')).read_text())
assert restored['normal_API_restored'] and not restored['diagnostic_libraries_loaded']
result={'schema':1,'pinned_commit':'caf35bcdd27c0240572e05a60633f07fca272844',
    'engine_or_supplied_script_executed':False,'timing_benchmark_performed':False,
    'runs':reports,'normal_API_restored_receipt':True,'diagnostic_libraries_loaded_in_restored_receipt':False,
    'normal_n_outputs_max_1_condition':'Only if non-embedding, non-audio, no enabled speculative method, parallel=1; verify actual restored server context/receipt before equating.',
    'collector_source_loop_model':{'L':37,'T1_separate_gate_up_down_expected_hook_calls':111,
        'registration_prior_gate_iterations':37*36//2,
        'active_gate_pointer_visits':37*38//2,
        'active_up_down_unmatched_pointer_visits':2*37*37,
        'active_up_down_unmatched_alias_visits':2*37*37,
        'record_payload_bytes_per_successful_T1_attempt':37*128,
        'assumptions':'Normal separate 3-projection MMID path for CPU layers0..36, no additional LoRA/extra-MMID path. Counts are source model, not runtime counters.'}}
(out/'DRIVER_COST_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
manifest={}
for rel in sorted(set(paths)):
    data=(repo/rel).read_bytes()
    manifest[rel]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                   'lines':len(data.splitlines())}
(out/'SOURCE_MANIFEST.json').write_text(json.dumps({'schema':1,'commit':result['pinned_commit'],'files':manifest},indent=2)+'\n')
print(json.dumps({'source_files':len(manifest),'calls':[x['calls'] for x in reports],
    'per_call_serialized_bytes':[x['per_call_RUN_serialized_bytes_source_reconstruction'] for x in reports],
    'output':str(out)},indent=2))
