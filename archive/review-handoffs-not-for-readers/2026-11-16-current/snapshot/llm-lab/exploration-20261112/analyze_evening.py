"""Summaries of completed evening experiments; never extrapolate to tok/s."""
from pathlib import Path
import json,statistics
R=Path(__file__).resolve().parent
def main():
    tlb=[]
    for p in sorted((R/'runs').glob('*evening-tlb*/CPU_STAT.json')):
        stat=json.loads(p.read_text());result=json.loads((p.parent/'RESULT.json').read_text())
        events={x['event']:x for x in stat['events']};assert all(v['running_percent']>=99 for v in events.values())
        assert result['error'] is None
        steps=result['final']['timings']['predicted_n']-1;w=events['r8208:u']['value'];d=events['r8408:u']['value'];c=events['cycles:u']['value']
        tlb.append({'source':str(p),'decode_steps':steps,'walks':w,'PMH_busy_cycles':d,'user_cycles':c,'walks_per_step':w/steps,'PMH_busy_cycles_per_step':d/steps,'PMH_busy_per_user_cycle':d/c})
    if tlb:(R/'cpu-wait/TLB_OBSERVATION.json').write_text(json.dumps({'rows':tlb,'combined_PMH_busy_per_user_cycle':sum(x['PMH_busy_cycles'] for x in tlb)/sum(x['user_cycles'] for x in tlb),'interpretation':'PMH active user cycles, not stall time, average walk latency, or THP speedup bound. Two synthetic workloads, processwide counters; no attribution to compact mapping.'},indent=2)+'\n')
    hot=[]
    for p in sorted((R/'execution').glob('hot-ffn-L*-s*/stdout.log')):
        d=json.loads(p.read_text());assert d['pass']
        groups={h:[x for x in d['measurements'] if x['hot']==h] for h in [0,1,3,6]}
        baseline=statistics.mean(statistics.median(x['concurrent_ms']) for x in groups[0])
        for h,rows in groups.items():
            cm=statistics.mean(statistics.median(x['concurrent_ms']) for x in rows);sm=statistics.mean(statistics.median(x['serial_ms']) for x in rows)
            hot.append({'layer':d['layer'],'seed':d['seed'],'H':h,'all_CPU_ms_bracket_mean':baseline,'concurrent_ms_mean_of_block_medians':cm,'serial_ms_mean_of_block_medians':sm,'time_reduction_vs_bracketed_CPU_percent':(1-cm/baseline)*100,'time_reduction_vs_serial_percent':(1-cm/sm)*100,'max_NMSE_vs_FP64_reference':max(x['mixed_error_vs_FP64_reference']['nmse'] for x in rows),'source':str(p)})
    if hot:(R/'hot-ffn/FIXTURE_RESULTS.json').write_text(json.dumps({'scope':'Real6expertweights andsyntheticinput,hostinput→host6slotoutput,weightsalreadyresident,perlayerconditionalH. Timedrepetitionsarecorrelated;noIIDconfidenceclaims,nofullmodelgainprediction. CPUusesselectedcompactcache.', 'rows':hot},indent=2)+'\n')
    path=R/'hybrid-pool/LIVE_ABBA.json'
    if path.exists():
        d=json.loads(path.read_text());rows=[]
        for context in [1024,4096]:
            arms=[]
            for arm in ['control','candidate']:
                paths=[]
                for run in [1,2]:
                    matches=list((R/'runs').glob(f'*-hybrid-{arm}-r{run}-prefill-count-plain-c{context}-n16/RESULT.json'));assert len(matches)==1;paths.append(matches[0])
                runs=[json.loads(p.read_text()) for p in paths];arms.append(runs)
            assert all(x['error'] is None for arm in arms for x in arm)
            rows.append({'context':context,'control_prompt_ms':[x['final']['timings']['prompt_ms'] for x in arms[0]],'candidate_prompt_ms':[x['final']['timings']['prompt_ms'] for x in arms[1]],'input_time_reduction_percent':(1-sum(x['final']['timings']['prompt_ms'] for x in arms[1])/sum(x['final']['timings']['prompt_ms'] for x in arms[0]))*100,'same_output':len({x['output_sha256'] for arm in arms for x in arm})==1})
        (R/'hybrid-pool/DECISION.json').write_text(json.dumps({'decode_gain_percent':d['aggregate_gain_percent'],'prefill':rows,'adopted':False,'reason':'No benefit: decode effectively tied; both long input comparisons slower. Keep checkpoint OpenMP.'},indent=2)+'\n')
    print(json.dumps({'TLB_samples':len(tlb),'hot_fixture_rows':len(hot),'hybrid_summary':path.exists()}))
if __name__=='__main__':main()
