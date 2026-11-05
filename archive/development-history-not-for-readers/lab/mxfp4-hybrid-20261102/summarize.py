#!/usr/bin/env python3
"""Regenerate evidence tables without executing workloads or touching the API."""
from pathlib import Path
import csv
import json

R=Path(__file__).resolve().parent
rows=[];suites={}
for path in sorted((R/'results').glob('*/summary.jsonl')):
    data=[json.loads(line) for line in path.read_text().splitlines()]
    assert all(x['passed'] for x in data)
    suites[path.parent.name]={'cases':len(data),'all_passed':True}
    for x in data:
        rows.append({'suite':path.parent.name,'type':x['type'],'threads':x['threads'],
          'direction':'down' if x['down'] else 'up','k':x['k'],'m':x['m'],'tokens':x['tokens'],
          'routing':x['routing'],'pair_median_ms':x['stock']['median_ms'],
          'hybrid_median_ms':x['candidate']['median_ms'],'speed_ratio':x['speed_ratio'],
          'speed_change_percent':100*(x['speed_ratio']-1),
          'ratio_lower_extreme':x['ratio_lower_extreme'],'ratio_upper_extreme':x['ratio_upper_extreme'],
          'passed':x['passed']})
with (R/'all-graph-results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
extra=json.loads((R/'extra-validation/RESULT.json').read_text())
runtime=json.loads((R/'RUNTIME.json').read_text())
summary={'comparison':'A=previous pair-only CPU candidate; B=new hybrid4/2/1 CPU candidate. A is not the production stock CPU library.',
         'suites':suites,'case_configurations':len(rows),'focus':[x for x in rows if x['suite'].startswith('focus-')],
         'extra_validation':{k:v for k,v in extra.items() if k!='commands'},
         'loader_checks':{k:{f:v for f,v in item.items() if f!='ldd'} for k,item in runtime['loader_checks'].items()},
         'full_model_AB_performed':False,'production_changed':False,
         'remaining_regression':'24 threads, up, 4 inputs: speed ratio 0.97095 vs pair; do not assume uniform improvement.'}
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
table=['threads  shape  inputs  pair_ms  hybrid_ms  speed_ratio  speed_change  extreme_ratio_range']
for x in summary['focus']:
    table.append(f"{x['threads']:7}  {x['direction']:5}  {x['tokens']:6}  {x['pair_median_ms']:7.3f}  {x['hybrid_median_ms']:9.3f}  {x['speed_ratio']:11.3f}  {x['speed_change_percent']:+10.1f}%  [{x['ratio_lower_extreme']:.3f}, {x['ratio_upper_extreme']:.3f}]")
(R/'focus-table.txt').write_text('\n'.join(table)+'\n')
print(json.dumps({'graph_case_configurations':len(rows),'passed':True,'full_model_AB_performed':False}))
