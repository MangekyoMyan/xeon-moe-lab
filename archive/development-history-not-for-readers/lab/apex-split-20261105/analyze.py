from pathlib import Path
import hashlib,json,math,statistics
R=Path(__file__).resolve().parent;plan=json.loads((R/'VALIDATION_PLAN.json').read_text());layers={x['layer']:x for x in json.loads((R/'LAYERS.json').read_text())}
amend=json.loads((R/'CONTROL_AMENDMENT.json').read_text())
assert amend['base_plan_sha256']==hashlib.sha256((R/'VALIDATION_PLAN.json').read_bytes()).hexdigest()
jobs=[j for j in plan['jobs'] if j!=amend['excluded_precompute_job']]+[amend['replacement_job']]
full=json.loads((R/'FULL_BATCH_PLAN.json').read_text());assert full['build']==plan['build'];jobs+=full['jobs']
rows=[]
def median(values):return statistics.median(values)
for job in jobs:
    folder=R/'data'/f"validation-L{job['layer']}-{job['workload']}-C{job['chunk']}-t{job['threads']}"
    receipt=json.loads((folder/'RESULT.json').read_text());d=receipt['result'];a=receipt['arguments']
    assert receipt['ok'] and d['pass'] and receipt['build']==plan['build'] and receipt['workload']['sha256']==job['route_sha256']
    for k in ['layer','workload','chunk','threads','repeats','gpu_experts']:assert a[k]==job[k]
    assert d['experts']==384 and d['topk']==6 and len(d['histogram'])==384 and sum(d['histogram'])==d['tokens']*6
    A=d['active_experts'];measurements=d['measurements'];selected=[x for x in measurements if x['gpu_experts']==job['gpu_experts']];assert len(selected)==1;candidate=selected[0]
    original_gpu=[x for x in measurements if x['gpu_experts']==A];original_cpu=[x for x in measurements if x['gpu_experts']==0];assert len(original_cpu)==1 and len(original_gpu)>=2
    for x in measurements:
        assert len(x['wall_ms'])==job['repeats'] and all(math.isfinite(v) and v>0 for v in x['wall_ms']) and x['nmse_vs_native_CPU']<2e-3
        assert x['serial_parallel_nmse']<1e-10
    counts=sorted(d['histogram'],reverse=True);assert candidate['cpu_pairs']==d['tokens']*6-sum(counts[:job['gpu_experts']])
    expertbytes=sum(w['expert_bytes'] for w in layers[job['layer']]['weights']);assert candidate['weight_H2D_bytes']==job['gpu_experts']*expertbytes
    cg=median([v for x in original_gpu for v in x['wall_ms']]);cc=median(original_cpu[0]['wall_ms']);cm=median(candidate['wall_ms']);serial=median(candidate['serial_wall_ms'])
    gpu_plans=[median(x['wall_ms']) for x in original_gpu];index=measurements.index(candidate);assert measurements[index-1]['gpu_experts']==A and measurements[index+1]['gpu_experts']==A
    bracket=median(measurements[index-1]['wall_ms']+measurements[index+1]['wall_ms']);best=min(cc,cg)
    row={**job,'tokens':d['tokens'],'weights_types':d['weights_types'],'CPU_ms':cc,'GPU_ms':cg,'GPU_bracketing_ms':bracket,
         'GPU_plan_medians_ms':gpu_plans,'split_ms':cm,'split_min_ms':min(candidate['wall_ms']),'split_max_ms':max(candidate['wall_ms']),
         'split_MAD_ms':median([abs(x-cm) for x in candidate['wall_ms']]),'split_speedup_vs_best_whole':best/cm,
         'split_speedup_vs_fastest_GPU_plan_or_CPU':min(cc,min(gpu_plans))/cm,'split_vs_bracket_or_CPU':min(cc,bracket)/cm,
         'latency_reduction_vs_best_whole':1-cm/best,'serial_split_ms':serial,'serial_to_parallel_ratio':serial/cm,
         'prepare_ms':candidate['prepare_ms'],'prepare_plus_split_ms':candidate['prepare_ms']+cm,
         'conservative_ratio_including_fresh_plan':best/(candidate['prepare_ms']+cm),
         'CPU_compute_concurrent_ms':median(candidate['cpu_compute_ms']),'CPU_compute_serial_ms':median(candidate['serial_cpu_compute_ms']),
         'CPU_compute_slowdown_in_overlap':median(candidate['cpu_compute_ms'])/median(candidate['serial_cpu_compute_ms'])-1,
         'CPU_branch_done_ms':median(candidate['cpu_done_ms']),'GPU_branch_done_ms':median(candidate['gpu_done_ms']),
         'GPU_serial_path_ms':median([g-c for g,c in zip(candidate['serial_gpu_done_ms'],candidate['serial_cpu_done_ms'])]),
         'CPU_input_ms':median(candidate['cpu_input_ms']),'CPU_return_ms':median(candidate['cpu_return_ms']),
         'CPU_pairs':candidate['cpu_pairs'],'weight_H2D_bytes_baseline':A*expertbytes,'weight_H2D_bytes_split':candidate['weight_H2D_bytes'],
         'CPU_activation_roundtrip_payload_bytes':candidate['cpu_pairs']*5120*4*2,
         'numeric_nmse_vs_native_CPU':candidate['nmse_vs_native_CPU'],'serial_parallel_nmse':candidate['serial_parallel_nmse'],
         'record_sha256':hashlib.sha256((folder/'RESULT.json').read_bytes()).hexdigest()}
    native_path=R/'native'/f"L{job['layer']}-{job['workload']}"/'RESULT.json';native=json.loads(native_path.read_text())
    assert native['ok'] and native['result']['pass'] and native['workload']['sha256']==job['route_sha256'] and native['config_sha256']==receipt['config_sha256']
    assert native['build']['included_split_BUILD']==plan['build'] and native['result']['merge_nmse']<1e-10
    assert len(native['result']['wall_ms'])==7 and all(math.isfinite(v) and v>0 for v in native['result']['wall_ms'])
    nc=median(native['result']['wall_ms']);strongest=min(nc,cc,cg)
    row.update(native_CPU_boundary_ms=nc,native_CPU_compute_ms=median(native['result']['cpu_compute_ms']),
        best_observed_endpoint_ms=strongest,split_speedup_vs_best_observed_endpoint=strongest/cm,
        gain_after_fresh_plan_vs_best_observed=strongest/(cm+candidate['prepare_ms']),
        packed_CPU_over_native_CPU=cc/nc,native_control_record_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest())
    rows.append(row)
summary={'scope':'12 fixed-split operator conditions,5 synthetic routing workloads,2 actual layers,compact GPU bank sizes8/16/24/32. Not12 independent user requests.',
         'selection':'fractions frozen from pilot before new timings; no adaptive controller/full API integration tested',
         'baseline':'best of packed CPU, native-shape CPU, packed compact-bank GPU complete expert FFN; not current full API graph',
         'validation_plan_sha256':hashlib.sha256((R/'VALIDATION_PLAN.json').read_bytes()).hexdigest(),'full_batch_plan_sha256':hashlib.sha256((R/'FULL_BATCH_PLAN.json').read_bytes()).hexdigest(),'resource_only_control_amendment':amend,'cases':rows,
         'max_numeric_nmse':max(x['numeric_nmse_vs_native_CPU'] for x in rows),'max_serial_parallel_nmse':max(x['serial_parallel_nmse'] for x in rows),
         'production_modified':False}
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
for x in rows:print(x['layer'],x['workload'],'C'+str(x['chunk']),*[round(x[k],3) for k in ['CPU_ms','GPU_ms','split_ms','split_speedup_vs_best_whole','serial_to_parallel_ratio','conservative_ratio_including_fresh_plan']])
