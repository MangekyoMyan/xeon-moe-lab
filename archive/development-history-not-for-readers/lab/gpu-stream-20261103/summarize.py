#!/usr/bin/env python3
"""Extract saved measurements; never execute models or tests."""
from pathlib import Path
import csv
import json

R=Path(__file__).resolve().parent
records=[]
for p in sorted((R/'results').glob('*/result.json')):
    manifest=json.loads((p.parent/'manifest.json').read_text())
    for item in json.loads(p.read_text()):
        r=item['result']
        records.append({'run':p.parent.name,'CPU':manifest['cpu'],'source_mode':r.get('source_mode','vector'),
          'case':item['case']['name'],'tokens':r['tokens'],'experts':r['experts'],'weight_bytes':r['weight_bytes'],
          **{k:r[k] for k in ['cpu_ms','gpu_resident_ms','pageable_upload_compute_ms','registered_upload_compute_ms','registered_all_io_compute_ms','registration_ms','cpu_gpu_nmse','cpu_ref_nmse','gpu_ref_nmse','registered_nmse','pass']},
          'registered_speed_ratio':r['cpu_ms']/r['registered_upload_compute_ms'],
          'all_IO_speed_ratio':r['cpu_ms']/r['registered_all_io_compute_ms']})
with (R/'measurements.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
assert all(r['pass'] for r in records)
validation=json.loads((R/'VALIDATION_LATEST.json').read_text())
validation_result=json.loads((Path(validation['directory'])/'RESULT.json').read_text())
profiles=json.loads((R/'PROFILES.json').read_text())
summary={'tests':len(records),'all_operator_tests_passed':True,
 'readonly_mmap_tests':sum(r['source_mode']=='read_only_mmap' for r in records),
 'full_model_AB_performed':False,'API':'stopped by user authorization',
 'candidate':'gpu3-stream1024','GPU_resident_layers':[37,38,39],
 'thresholds':{'IQ_types':512,'MXFP4':1024},'CPU_kernel':'previous hybrid4/2/1',
 'max_observed_single_operator_CPU_GPU_NMSE':max(r['cpu_gpu_nmse'] for r in records),
 'candidate_registered_host_GiB':profiles['profiles']['gpu3-stream1024']['registered_expert_bytes']/2**30,
 'candidate_resident_expert_GiB':profiles['profiles']['gpu3-stream1024']['static_expert_bytes']/2**30,
 'validation':validation,'validation_result':validation_result,
 'stock_1024':[r for r in records if r['CPU']=='stock' and r['source_mode']=='read_only_mmap' and r['tokens']==1024],
 'hybrid_512_1024':[r for r in records if r['CPU']=='hybrid' and r['tokens']>=512]}
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
for r in summary['stock_1024']:
    print(f"{r['case']}: CPU {r['cpu_ms']:.1f} ms / registered GPU {r['registered_upload_compute_ms']:.1f} ms / all IO {r['registered_all_io_compute_ms']:.1f} ms")
print('operator cases:',len(records),'full model: not yet tested')
