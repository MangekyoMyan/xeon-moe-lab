from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;Q=R/'q8-avx1';LAB=R.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=json.loads((Q/'FFN_BUILD.json').read_text());cmd=old['command']
    with (Q/'FFN_BUILD.stdout.log').open('w') as o,(Q/'FFN_BUILD.stderr.log').open('w') as e:subprocess.run(cmd,stdout=o,stderr=e,check=True,timeout=120)
    assert (Q/'FFN_BUILD.stderr.log').stat().st_size==0
    old.update(source_sha256=sha(Q/'real_ffn.cpp'),binary_sha256=sha(Q/'real_ffn'));(Q/'FFN_BUILD.json').write_text(json.dumps(old,indent=2)+'\n')
    assert not (Q/'FFN_DESIGN.json').exists()
    capture=LAB/'parallel-pilot-20261117/capture/runs/20261117T052657Z-U1024-3483cb75'
    manifest=json.loads((capture/'RESULTS_SHA256.json').read_text())
    for n,h in manifest.items():assert sha(capture/n)==h,n
    cpuplan={};plan=LAB/'exploration-20261112/plans/frequency19200.plan'
    for line in plan.read_text().splitlines()[1:]:
        x=line.split();ids=list(map(int,x[6:]));assert len(ids)==int(x[5]);cpuplan[x[0]]=ids
    engine=LAB/'deepseek41-bringup-20261101';config=json.loads((engine/'server-config.json').read_text());cmd=config['command'];model=Path(cmd[cmd.index('-m')+1]);shards=sorted(model.parent.glob('*.gguf'));assert len(shards)==7
    specs=[]
    for m in ['control','candidate']:specs.append(dict(name='quality-'+m,mode=m,stage='quality',cells=[1,5,6],threads=[12,24],samples=1,loops=1))
    order=['control','candidate','candidate','control','candidate','control','control','candidate']
    for i,m in enumerate(order):
        reverse=i%4>=2;specs.append(dict(name=f'timing-{i}-{m}',mode=m,stage='timing',cells=[6,5,1] if reverse else [1,5,6],threads=[24,12] if reverse else [12,24],samples=5,loops=8))
    common=dict(capture_json=str(capture/'CAPTURE.json'),capture_manifest_sha256=sha(capture/'RESULTS_SHA256.json'),cpu_plan=cpuplan,cpu_plan_sha256=sha(plan),shards=[str(p) for p in shards],shard_stats={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in shards},base=str(LAB/'gpu-stream-20261103/runtime/libggml-base.so.0.28.0'))
    design=dict(common=common,stages=specs,baseline_config_sha256=sha(engine/'server-config.json'),CPU_build_sha256=sha(Q/'BUILD.json'),FFN_build_sha256=sha(Q/'FFN_BUILD.json'),guard_sha256=sha(engine/'supervise.py'),timed_graph_calls=1920,
        scope='3 saved actual x/top6 FFN fixtures: cell1 IQ2_S/IQ3_XXS, cell5 IQ3_XXS/IQ3_S, cell6 MXFP4 negative control.12physical/24logical affinity; T1 geometry at24threads is NOT a prefill workload.',
        process_design='quality control/candidate, then ABBA BAAB eight fresh processes; fourprocesses per mode; pair samecaseorder. Threewarmup then5samples×8wholeFFNcalls. Keep all samples. No newmodelinference or token/s claim.',
        reference='Everywarmup/sample output must exactlymatch saved baseline down-output. Sameoriginal384-expert GGUF tensors mappedread-only; selectedcompact mask from live plan and oldruntime capture; cache has only required selectedexperts.',
        adoption='No automatic API change; gate full-model test on meaningful stablewholeoperator benefit. No gridsearch or addedCPUcandidate.',
        NUMA='without_thp.py and numactl --physcpubind=0-23 --interleave=0,1; individualpoolstrict0..threads-1; actual originalfilepage placement depends on existing pagecache. Fresh compact uses inherited interleave, rowlocal0 as currentconfig.')
    (Q/'FFN_DESIGN.json').write_text(json.dumps(design,indent=2)+'\n')
    (Q/'STATE.json').write_text(json.dumps(dict(phase='Prepared; numericpassed; API not stopped',runs={},restore_required=False),indent=2)+'\n');print('Frozen10process study prepared;1920timed FFNcalls')
if __name__=='__main__':main()
