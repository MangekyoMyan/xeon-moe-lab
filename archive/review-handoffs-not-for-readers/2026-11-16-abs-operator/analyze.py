"""Strict receipt and timing analysis; retain all observations, including outliers."""
from pathlib import Path
from statistics import median,mean
import hashlib,json,math
R=Path(__file__).resolve().parent
def read_run(name,kind):
    folder=R/'execution'/name
    status=json.loads((folder/'STATUS.json').read_text());assert status['exit_code']==0 and status['error'] is None,name
    lines=[json.loads(l) for l in (folder/'stdout.log').read_text().splitlines() if l]
    assert len(lines)==3 and lines[-1]=={'passed':True},name
    assert not (folder/'stderr.log').read_bytes(),name
    assert {x['cached_of_six'] for x in lines[:-1]}=={0,6}
    for x in lines[:-1]:
        assert x['kind']==kind and x['threads']==12 and sorted(x['pins'])==list(range(12)) and x['bit_mismatches']==0
        assert len(x['routes'])==len(set(x['routes']))==48
        assert x['cache_bytes']==(318504960 if x['cached_of_six'] else 0)
    return lines[:-1]
def description(values):
    return {'median':median(values),'mean':mean(values),'min':min(values),'max':max(values),'wins':sum(v>1 for v in values),'n':len(values),'all':values}
def main():
    result={'same_library':[],'cross_library':[],'limitations':['Repeated timed blocks are not independent requests/process replicates.','No full-model tokens/s, actual routed activation distribution, or confidence intervals.','Short guard sample intervals may repeat older daemon sensor snapshots.','Timing outliers are retained.']}
    for tag in ['iq2','iq3s']:
        for repeat in [1,2]:
            name=f'real-{tag}-r{repeat}'
            for x in read_run(name,'operator'):
                samples=x['pairs'];assert len(samples)==128
                assert {(s['block'],s['candidate'],s['order']) for s in samples}=={(b,c,o) for b in range(16) for c in [1,2] for o in range(4)}
                for s in samples:
                    assert math.isfinite(s['ms']) and s['ms']>0 and s['route_group']==s['block']%8
                    reverse=(s['block']%8+s['block']//8)%2
                    order=[s['candidate'],0,0,s['candidate']] if reverse else [0,s['candidate'],s['candidate'],0]
                    assert s['mode']==order[s['order']]
                for c in [1,2]:
                    ratios=[]
                    for b in range(16):
                        p=[s for s in samples if s['block']==b and s['candidate']==c]
                        ratios.append(mean(s['ms'] for s in p if s['mode']==0)/mean(s['ms'] for s in p if s['mode']==c))
                    d=description(ratios)
                    d.update(run=name,type=x['type'],hits=x['cached_of_six'],candidate=c,time_change_percent=100*(1/d['median']-1),screen_pass=(d['median']>=1.02 and d['wins']>=12) if x['cached_of_six'] else 1/d['median']<=1.01)
                    result['same_library'].append(d)
    receipt=json.loads((R/'FIXED_RECEIPT.json').read_text());assert receipt['complete'] and len(receipt['runs'])==12
    for tag in ['iq2','iq3s']:
        runs=[r for r in receipt['runs'] if r['tag']==tag]
        assert [r['label'] for r in runs]==['frozen','off','shared','shared','off','frozen']
        for fname in ['inputs.bin','outputs-hit0.bin','outputs-hit6.bin']:
            assert len({r['files'][fname]['sha256'] for r in runs})==1,(tag,fname)
            for r in runs:
                p=R/'fixed-outputs'/r['run']/fname;assert hashlib.sha256(p.read_bytes()).hexdigest()==r['files'][fname]['sha256']
        outputs={r['run']:read_run(r['run'],'fixed') for r in runs}
        for hits in [6,0]:
            groups={label:[] for label in ['frozen','off','shared']}
            process_medians={label:[] for label in groups}
            for r in runs:
                x=next(x for x in outputs[r['run']] if x['cached_of_six']==hits)
                assert x['mode']==(1 if r['label']=='shared' else 0) and x['experimental']==(r['label']!='frozen')
                assert [s['block'] for s in x['samples']]==list(range(32))
                for s in x['samples']:assert math.isfinite(s['ms']) and s['ms']>0 and s['route_group']==s['block']%8
                v=[s['ms'] for s in x['samples']];groups[r['label']].append(v);process_medians[r['label']].append(median(v))
            means={label:[mean(v[i] for v in data) for i in range(32)] for label,data in groups.items()}
            for label in ['off','shared']:
                ratios=[a/b for a,b in zip(means['frozen'],means[label])];d=description(ratios)
                d.update(type=tag,hits=hits,candidate=label,process_medians_ms=process_medians,time_change_percent=100*(1/d['median']-1))
                d['screen_pass']=d['median']>=1.02 if hits else 1/d['median']<=1.01
                result['cross_library'].append(d)
    (R/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    for d in result['cross_library']:print(d['type'],d['hits'],d['candidate'],'ratio',round(d['median'],6),'time change %',round(d['time_change_percent'],3),'process medians',d['process_medians_ms'])
if __name__=='__main__':main()
