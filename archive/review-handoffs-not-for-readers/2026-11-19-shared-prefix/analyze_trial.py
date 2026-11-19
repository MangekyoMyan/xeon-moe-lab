"""Offline prospective whole-token ABBA calculation. Does not launch a model."""
from pathlib import Path
import argparse,hashlib,json,statistics
def read(p):return json.loads(p.read_text())
def analyze(root):
    t=root/'trial';design=read(t/'DESIGN.json');metric=read(t/'PERFORMANCE_METRIC.json')
    stages=[s for s in design['stages'] if s['stage']=='timing'];assert [s['arm'] for s in stages]==list('ABBA')
    data=[]
    for spec in stages:
        run=t/'runs'/spec['name'];j=read(run/'RUN.json');assert j['passed'] and j['spec']==spec
        assert not read(run/'LAUNCH.json')['environment'].get('LD_PRELOAD')
        cases={}
        for b in j['blocks']:
            durations=[(y-x)/1e6 for x,y in zip(b['start_ns'],b['output_ready_ns'])]
            assert len(durations)==64 and min(durations)>0 and b['status']==[0]*64
            final=(run/b['file']).read_bytes();assert final==(root/'main/runs/baseline'/b['file']).read_bytes()[-129280*4:]
            cases[b['case_id']]=dict(mean_ms=statistics.mean(durations),median_ms=statistics.median(durations),first_ms=durations[0],min_ms=min(durations),max_ms=max(durations),all_ms=durations)
        data.append(dict(name=spec['name'],arm=spec['arm'],case_order=spec['cases'],cases=cases,run_json_sha256=hashlib.sha256((run/'RUN.json').read_bytes()).hexdigest()))
    case_results={}
    for name in data[0]['cases']:
        v=[x['cases'][name]['mean_ms'] for x in data];pairs=[v[1]/v[0],v[2]/v[3]];a=(v[0]+v[3])/2;b=(v[1]+v[2])/2
        case_results[name]=dict(process_means_ms=v,paired_ratios_B_over_A=pairs,A_mean_ms=a,B_mean_ms=b,ratio_B_over_A=b/a,time_reduction_percent=100*(1-b/a),rate_increase_percent=100*(a/b-1),saved_ms=a-b,baseline_last_minus_first_ms=v[3]-v[0],passes=all(x<1 for x in pairs) and b/a<=0.995)
    passed=all(x['passes'] for x in case_results.values())
    return dict(metric=metric,independent_processes=4,timed_calls=512,cases=case_results,processes=data,screen_passed=passed,decision='advance to ordinary greedy verification; no automatic adoption' if passed else 'hold; no further retries in this study',interpretation='Two fixed training cases. Inclusive64call means, same-period uninstrumented original/candidate. Do not interpret kernel busy time as promised wall saving.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();j=analyze(a.root)
    with a.output.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in j.items() if k!='processes'},indent=2))
if __name__=='__main__':main()
