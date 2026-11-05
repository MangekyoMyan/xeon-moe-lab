#!/usr/bin/env python3
"""Aggregate completed deterministic simulations; no inferential confidence claims."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/hybrid-sim-deep-20261023'
MODES=['smoke','screen','sweep','diagnostics','cpu_sensitivity']

def main():
    runs={};configs={};rows=[]
    for mode in MODES:
        status=json.loads((OUT/f'{mode}-status.json').read_text())
        assert status['exit_code']==0
        cfg=json.loads((OUT/f'{mode}-configs.json').read_text())
        parsed=list(csv.DictReader((OUT/f'{mode}-results.tsv').open(),delimiter='\t'))
        overall=[r for r in parsed if r['turn']=='-1']
        assert len(overall)==status['cases']==len(cfg)
        for r in overall:
            converted={k:v if k in ['id','card','trace'] else float(v) for k,v in r.items()}
            assert abs(sum(converted[k] for k in ['common_path_ms','attention_path_ms','moe_input_ms','parallel_ms','cpu_tail_ms','gpu_tail_ms','merge_path_ms','other_path_ms'])-converted['ms_token'])<1e-7
            rows.append({'mode':mode,**converted});runs[r['id']]=converted
        configs.update({c['id']:c for c in cfg})
    keys=list(rows[0])
    with (OUT/'all-overall-results.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys,delimiter='\t');writer.writeheader();writer.writerows(rows)
    synthetic=[r for r in json.loads((OUT/'synthetic/steady-synthetic.json').read_text())['results'] if r['turn']==-1]
    central=[runs['central_'+c] for c in ['P40','V100']]
    ram=[]
    for ref in [40,45,50,55]:
        for card in ['P40','V100']:
            a=runs[f'cpu_ref{ref}_RAM1_compute1_{card}'];b=runs[f'cpu_ref{ref}_RAM2_compute1_{card}']
            ram.append({'ref_GBs':ref,'card':card,'baseline_tok_s':a['tok_s'],'RAM2_tok_s':b['tok_s'],'speedup_percent':100*(b['tok_s']/a['tok_s']-1)})
    signatures=set()
    for c in configs.values():
        d={k:float(v) if isinstance(v,(int,float)) else v for k,v in c.items() if k not in ['id','detail','card']}
        for key in ['trace','phases','shapes','cpu_file']:
            d[key]=hashlib.sha256(Path(d[key]).read_bytes()).hexdigest()
        signatures.add(json.dumps(d,sort_keys=True))
    summary={'scope':'Conditional resource and cache simulation, not measured GPU or 512GB throughput.',
             'configured_real_trace_runs':len(rows),'unique_real_trace_configurations':len(signatures),
             'synthetic_runs':14,'public_decode_tokens':6763,'public_traces':3,
             'central':central,'V100_central_speedup_percent':100*(central[1]['tok_s']/central[0]['tok_s']-1),
             'RAM_sensitivity':ram,'synthetic':synthetic,
             'P40_rate_crossover_bracket_GBs':[250,255],
             'V100_rate_crossover_bracket_GBs':[340,345]}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['central','synthetic']},indent=2))

if __name__=='__main__':main()
