#!/usr/bin/env python3
"""Compile real Model tests; independently reconcile CLI phase/detail output."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile

SRC=Path(__file__).resolve().parent
OUT=SRC.parent/'results/hybrid-sim-deep-20261023'
AUDIT=OUT/'audit'


def close(a,b,label,rtol=1e-5):
    if not math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=1e-7):
        raise AssertionError((label,a,b))


def main():
    AUDIT.mkdir(exist_ok=True)
    calibration=json.loads((OUT/'cpu-model.json').read_text())
    with tempfile.TemporaryDirectory(prefix='integration-tests-') as tmp:
        executable=Path(tmp)/'integration-tests'
        command=['c++','-std=c++17','-O2','-Wall','-Wextra','-Wpedantic',str(SRC/'integration-tests.cpp'),'-o',str(executable)]
        build=subprocess.run(command,check=True,text=True,capture_output=True)
        result=subprocess.run([str(executable),str(OUT/'cpu-full22.tsv'),str(OUT/'layers-native.tsv'),
                               *map(str,calibration['full22_ms'][1:])],check=True,text=True,capture_output=True)
    configs=json.loads((OUT/'smoke-configs.json').read_text())
    rows=list(csv.DictReader((OUT/'smoke-results.tsv').open(),delimiter='\t'))
    reconciled=[]
    for c in configs:
        detail=list(csv.DictReader((OUT/'details'/f"{c['id']}.csv").open()))
        phases=[tuple(map(int,l.split())) for l in Path(c['phases']).read_text().splitlines()]
        assert len(detail)==len(phases)==Path(c['trace']).stat().st_size//528
        for d,(decode,turn) in zip(detail,phases):
            assert (int(d['decode']),int(d['turn']))==(decode,turn)
            assert int(d['gpu'])+int(d['cpu'])==240
            assert 0<=int(d['gpu'])<=int(d['ready'])<=240
        scored=[d for d in detail if int(d['decode'])]
        turns=sorted({int(d['turn']) for d in scored})
        for turn in [-1,*turns]:
            selected=scored if turn==-1 else [d for d in scored if int(d['turn'])==turn]
            summary=next(r for r in rows if r['id']==c['id'] and int(r['turn'])==turn)
            n=len(selected);assert int(summary['steps'])==n
            mean=sum(float(d['ms']) for d in selected)/n
            close(summary['ms_token'],mean,'phase latency mean')
            close(summary['tok_s'],1000/mean,'phase reciprocal throughput')
            for field,key,scale in [('ready_fraction','ready',240),('gpu_fraction','gpu',240),
                                    ('cpu_experts_token','cpu',1),('copy_MB_token','weight_MB',1),
                                    ('copies_token','copies',1),('input_wait_ms','input_wait_ms',1),
                                    ('return_wait_ms','return_wait_ms',1)]:
                close(summary[field],sum(float(d[key]) for d in selected)/n/scale,field)
            ordered=sorted(float(d['ms']) for d in selected)
            close(summary['p50_ms'],ordered[int(.5*(n-1))],'p50')
            close(summary['p95_ms'],ordered[int(.95*(n-1))],'p95')
            assert float(summary['host_peak_GBs'])<=c['host_bw']*c['mem_scale']*(1+1e-10)
            assert float(summary['gpu_peak_GBs'])<=c['gpu_peak_bw']*(1+1e-10)
        reconciled.append({'id':c['id'],'all_trace_steps':len(detail),'scored_decode_steps':len(scored),
                           'turns':len(turns),'summary_rows_verified':len(turns)+1})
    screened=[]
    status=OUT/'screen-status.json'
    if status.exists() and json.loads(status.read_text())['exit_code']==0:
        screen_configs=json.loads((OUT/'screen-configs.json').read_text())
        screen_rows=list(csv.DictReader((OUT/'screen-results.tsv').open(),delimiter='\t'))
        for c in screen_configs:
            phases=[tuple(map(int,l.split())) for l in Path(c['phases']).read_text().splitlines()]
            assert len(phases)==Path(c['trace']).stat().st_size//528
            expected={}
            for decode,turn in phases:
                assert decode in (0,1)
                if decode:expected[turn]=expected.get(turn,0)+1
            rows_for_case={int(r['turn']):r for r in screen_rows if r['id']==c['id']}
            assert set(rows_for_case)=={-1,*expected}
            assert int(rows_for_case[-1]['steps'])==sum(expected.values())
            for turn,n in expected.items():assert int(rows_for_case[turn]['steps'])==n
            for r in rows_for_case.values():
                close(float(r['gpu_fraction'])+float(r['cpu_experts_token'])/240,1,'CPU/GPU assignments partition routes')
                assert 0<=float(r['gpu_fraction'])<=float(r['ready_fraction'])+1e-10<=1+1e-10
                assert float(r['cpu_busy_ms'])<=float(r['ms_token'])*(1+1e-10)
                assert float(r['gpu_busy_ms'])<=float(r['ms_token'])*(1+1e-10)
                assert float(r['host_peak_GBs'])<=c['host_bw']*c['mem_scale']*(1+1e-10)
                assert float(r['gpu_peak_GBs'])<=c['gpu_peak_bw']*(1+1e-10)
                path_fields=['common_path_ms','attention_path_ms','moe_input_ms','parallel_ms',
                             'cpu_tail_ms','gpu_tail_ms','merge_path_ms','other_path_ms']
                if all(k in r for k in path_fields):
                    assert all(float(r[k])>=0 for k in path_fields)
                    close(sum(float(r[k]) for k in path_fields),r['ms_token'],'critical-path output partition',rtol=1e-9)
            for field in ['ms_token','ready_fraction','gpu_fraction','cpu_experts_token','copy_MB_token',
                          'copies_token','input_wait_ms','return_wait_ms','cpu_busy_ms','gpu_busy_ms']:
                weighted=sum(float(rows_for_case[t][field])*n for t,n in expected.items())/sum(expected.values())
                close(rows_for_case[-1][field],weighted,'per-turn weighted aggregate: '+field,rtol=1e-9)
            screened.append({'id':c['id'],'decode_steps':sum(expected.values()),'turns':len(expected)})
    payload={'passed':True,'build_command':command,'build_stdout':build.stdout,'build_stderr':build.stderr,
             'model_test_stdout':result.stdout,'model_test_stderr':result.stderr,'phase_reconciliation':reconciled,
             'screen_phase_and_resource_checks':screened,
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [SRC/'deep-sim.cpp',SRC/'resources.hpp',SRC/'cache.hpp',SRC/'run_deep.py',
                               SRC/'integration-tests.cpp',SRC/'integration-tests.py']},
             'scope':'No GPU performance claims; verifies Model/resource integration and existing smoke aggregation.'}
    (AUDIT/'INTEGRATION_TEST_RESULTS.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(result.stdout.strip())
    print(json.dumps(reconciled,ensure_ascii=False))
    print('Screen configurations checked:',len(screened))


if __name__=='__main__':main()
