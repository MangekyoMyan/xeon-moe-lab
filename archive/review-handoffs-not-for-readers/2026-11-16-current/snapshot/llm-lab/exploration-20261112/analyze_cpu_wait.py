"""Summarize measured CPU work, placement, traffic, and speed without power estimates."""
from pathlib import Path
import collections,json
R=Path(__file__).resolve().parent
def main():
    screen=json.loads((R/'cpu-wait/SCREEN_RUNS.json').read_text());rows=[];groups=collections.defaultdict(list)
    for item in screen['runs']:
        path=Path(item['result']);data=json.loads(path.read_text());stat=json.loads((path.parent/'CPU_STAT.json').read_text())
        assert data['error'] is None and stat['valid']
        counters={x['event']:x['value'] for x in stat['events']};cpu=data['decode_process_cpu'];steps=cpu['decoded_steps']
        assert data['output_token_count']==data['final']['timings']['predicted_n']==96
        before={x['tid']:x for x in data['threads_begin']['threads']};active=[]
        for end in data['threads_end']['threads']:
            if end['tid'] not in before:continue
            delta=end['cpu_seconds']-before[end['tid']]['cpu_seconds']
            if delta>max(.05,cpu['observed_wall_seconds']*.02):active.append({'tid':end['tid'],'cpu_seconds':delta,'affinity':end['affinity']})
        threads=int(item['profile'].split('-')[1][1:]);expected=[0,1,2,6,7,8] if threads==6 else list(range(threads))
        observed=sorted({x['affinity'][0] for x in active if len(x['affinity'])==1})
        assert observed==expected,(item['profile'],observed,expected)
        env=data['actual_omp_environment'];policy=item['profile'].split('-')[2]
        assert env.get('OMP_WAIT_POLICY')==('PASSIVE' if policy=='passive' else None)
        assert env.get('GOMP_SPINCOUNT')==('10000' if policy=='spin10000' else None)
        traffic=[s['imc'] for s in data['samples'] if s['phase']=='decode' and s['imc']['status']=='ok']
        assert all(x['running_fraction_min']>=.99 and not x['multiplexed'] for x in traffic)
        sampled_seconds=sum(x['interval_seconds'] for x in traffic)
        row={**item,'decode_tok_s':data['final']['timings']['predicted_per_second'],'prompt_ms':data['final']['timings']['prompt_ms'],'observed_ttft_s':data['ttft_s'],'cpu_seconds_per_step':cpu['cpu_seconds']/steps,'logical_CPU_time_parallelism':cpu['cpu_seconds']/cpu['observed_wall_seconds'],
             'instructions_per_step':counters['instructions:u']/steps,'cycles_per_step':counters['cycles:u']/steps,'IPC_user':counters['instructions:u']/counters['cycles:u'],'user_cycles_to_reference_ratio':counters['cycles:u']/counters['ref-cycles:u'],
             'context_switches_per_step':counters['context-switches']/steps,'cpu_migrations':counters['cpu-migrations'],'active_worker_cpus':observed,'active_threads':active,
             'IMC_read_GB_s':sum(x['read_bytes_per_second']*x['interval_seconds'] for x in traffic)/sampled_seconds/1e9 if sampled_seconds else None,'IMC_sampled_seconds':sampled_seconds,
             'output_sha256':data['output_sha256'],'output_token_sha256':data['output_token_sha256']}
        rows.append(row);groups[item['profile']].append(row)
    same=all(len({(r['output_sha256'],r['output_token_sha256']) for r in rows if r['workload']==workload})==1 for workload in ['count','code'])
    summary=[]
    for name,cases in groups.items():
        summary.append({'profile':name,'count':len(cases),'aggregate_tok_s':len(cases)/sum(1/c['decode_tok_s'] for c in cases),'mean_prompt_ms':sum(c['prompt_ms'] for c in cases)/len(cases),'mean_cpu_seconds_per_step':sum(c['cpu_seconds_per_step'] for c in cases)/len(cases),'mean_context_switches_per_step':sum(c['context_switches_per_step'] for c in cases)/len(cases),'mean_IMC_read_GB_s':sum(c['IMC_read_GB_s'] for c in cases)/len(cases)})
    result={'screen_complete':screen['complete'],'all_outputs_identical':same,'rows':rows,'summary':summary,'scope':'Instrumented first-token-to-stop client window; CPU time is logical-processor time, not physical cores or electrical power; wait policy also affects prefill although its thread count stays24; only default12 repeats; screening candidates need plain confirmation if selected'}
    (R/'cpu-wait/RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'all_outputs_identical':same,'summary':summary},indent=2))
if __name__=='__main__':main()
