"""Actual custom-node H histograms; no original-route reconstruction."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
def main():
    timing=json.loads((R/'TIMING_ANALYSIS.json').read_text());out=[]
    for c in timing['cases']:
        bb=[]
        for b in c['blocks']:
            if b['mode']!='P':continue
            before=b['before_T1'];after=b['after'];layers=[];total=[0]*7
            for l in range(40):
                hist=[v-u for u,v in zip(before['executed_H'][l],after['executed_H'][l])]
                calls=after['calls'][l]-before['calls'][l];assert sum(hist)==calls
                if calls:
                    assert calls==64
                    hit=sum(i*n for i,n in enumerate(hist));total=[x+y for x,y in zip(total,hist)]
                    layers.append(dict(layer=l,calls=calls,H=hist,hot_slots=hit,slot_hit_fraction=hit/(calls*6)))
            assert len(layers)==23 and sum(total)==23*64
            hits=sum(i*n for i,n in enumerate(total))
            bb.append(dict(process=b['process'],layers=layers,total_H=total,hot_slots=hits,
                selected23_slot_fraction=hits/(23*64*6),all37_CPU_layer_slot_fraction=hits/(37*64*6),
                selected23_H0_fraction=total[0]/(23*64)))
        out.append(dict(case_id=c['case_id'],blocks=bb,P_repeats_same_histogram=bb[0]['layers']==bb[1]['layers']))
    result=dict(scope='Executed custom-node H only, original35 unchanged. Includes23 selected host layers;14 unselected host layers bypass and3 fullGPUlayers are not in these denominators. Each node selects6 experts. H is count, not expert identity or measured saved time. No route ID collector in timing.',cases=out)
    p=R/'HIT_ANALYSIS.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(case=c['case_id'],hist_same=c['P_repeats_same_histogram'],blocks=[{k:v for k,v in b.items() if k!='layers'} for b in c['blocks']]) for c in out]))
if __name__=='__main__':main()
