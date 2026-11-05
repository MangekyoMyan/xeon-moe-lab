#!/usr/bin/env python3
"""Read bounded metadata captures. Never invent GPU durations or missing CPU work."""
from pathlib import Path
import argparse,collections,json,statistics

def report(path):
    with path.open('rb') as f:raw=f.read(8*1024**2+1)
    if len(raw)>8*1024**2:raise ValueError('Capture exceeds logger byte cap')
    complete=raw.endswith(b'\n');lines=raw.splitlines()
    if lines and not complete:lines.pop()
    events=[];statuses=[];invalid=[];keys=set();groups={};by_quant={}
    for number,line in enumerate(lines,1):
        row=json.loads(line)
        if row.get('kind')=='capture_status':statuses.append(row);continue
        if row.get('kind')!='route_observation':raise ValueError('Unknown record kind')
        if row.get('schema')!=1 or row.get('assignment_changed') is not False:raise ValueError('Unexpected observer schema or assignment')
        counts=row['histogram'];E=row['experts'];T=row['tokens'];k=row['topk'];size=row['expert_bytes']
        if len(counts)!=E or any(type(n) is not int or n<0 or n>T for n in counts) or sum(counts)!=T*k:raise ValueError('Invalid histogram')
        runs=sum(n>0 and (e==0 or counts[e-1]==0) for e,n in enumerate(counts))
        expected=sum(n>0 for n in counts)*size+sum(n>0 and e<E-1 and counts[e+1]==0 for e,n in enumerate(counts))*min(size,512)
        if expected!=row['submitted_H2D_bytes'] or runs!=row['submitted_copy_calls']:invalid.append({'line':number,'reason':'selective_transfer_metadata_mismatch'})
        key=(row['pid'],row['session'],row['operation'])
        if key in keys:raise ValueError('Duplicate operation record')
        keys.add(key)
        group=groups.setdefault((row['pid'],row['session'],row['route_epoch']),[]);group.append(row)
        by_quant.setdefault(row['quant'],[]).append(row);events.append(row)
    final=statuses[-1] if statuses else None
    intact_footer=bool(complete and lines and final and final.get('final') and json.loads(lines[-1])==final)
    sessions={(r['pid'],r['session']) for r in events+statuses}
    if len(sessions)>1:raise ValueError('Mixed capture sessions')
    ledger_ok=False
    if intact_footer:
        ledger_ok=(final['written']==len(events) and final['attempted']==final['accepted']+final['enqueue_dropped'] and final['accepted']==final['written']+final['write_dropped'])
    known_fetch=[];unknown_fetch=0
    for group in groups.values():
        refreshed=[r for r in group if r['route_refreshed']]
        if len(refreshed)==1:known_fetch.append(refreshed[0]['shared_fetch_wait_ms'])
        else:unknown_fetch+=1
    aggregates={}
    for quant,rows in by_quant.items():
        hist=[n for r in rows for n in r['histogram'] if n]
        aggregates[quant]={'operators':len(rows),'tokens':sorted({r['tokens'] for r in rows}),
            'declared_experts':sorted({r['experts'] for r in rows}),'active_expert_load_median':statistics.median(hist),
            'active_expert_load_max':max(hist),'expert_instances_with_ge4_rows':sum(n>=4 for n in hist),
            'submitted_H2D_bytes':sum(r['submitted_H2D_bytes'] for r in rows),
            'copy_callback_inclusive_host_ms':sum(r['copy_callback_host_ms'] for r in rows)}
    return {'schema':1,'source':str(path),'complete_lines':len(lines),'trailing_partial_line_ignored':not complete,
        'final_footer_present':intact_footer,'final_ledger_consistent':ledger_ok,
        'observations':len(events),'capture_status':final,'invalid_observations':invalid,
        'routing_groups':len(groups),'groups_with_unobserved_or_ambiguous_shared_fetch':unknown_fetch,
        'observed_shared_fetch_host_ms':sum(known_fetch),'by_quant':aggregates,
        'prediction_ready':False,'assignment_changes':0,
        'limits':['Callback is after assignment; CPU and GPU-resident operations are unobserved.',
          'Copy callback wall time includes route-fetch wait and observer overhead. Do not add shared fetch again.',
          'Host API durations are not DMA or kernel durations. No transfer bandwidth inferred.',
          'Missing refresh in a group means unknown fetch cost, not zero.',
          'Raw callback does not verify queue state, host registration, or full calibration provenance.',
          'The pilot model supports measured banks <=96 experts. Actual384-bank predictions are withheld.',
          'No final footer means a live or interrupted capture; zero events is not zero expert work.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path,nargs='?');p.add_argument('--latest',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if bool(a.capture)==a.latest:p.error('Specify one capture path or --latest')
    if a.latest:
        paths=list(Path('/var/lib/xeon-expert-shadow/captures').glob('routes-*.jsonl'))
        if not paths:raise SystemExit('No production capture yet; the observer is optional and off by default.')
        a.capture=max(paths,key=lambda f:f.stat().st_mtime_ns)
    text=json.dumps(report(a.capture),indent=2,ensure_ascii=False)+'\n'
    if a.output:a.output.write_text(text)
    else:print(text,end='')
if __name__=='__main__':main()
