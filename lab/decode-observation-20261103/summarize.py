from pathlib import Path
import json
import statistics

R=Path(__file__).resolve().parent
short=json.loads((R/'suite-state.json').read_text());long=json.loads((R/'long-suite-state.json').read_text())
assert short['complete'] and long['complete']
def summarize_suite(suite):
    rows=suite['completed'];base=[r['final']['timings']['predicted_per_token_ms'] for r in rows if r['mode']=='baseline']
    traced=[r for r in rows if r['mode']=='trace'][0]
    return {'baseline_decode_ms':base,'baseline_median_ms':statistics.median(base),
      'trace_decode_ms':traced['final']['timings']['predicted_per_token_ms'],
      'trace_slowdown_percent':100*(traced['final']['timings']['predicted_per_token_ms']/statistics.median(base)-1),
      'all_output_hashes_match':len({r['output_sha256'] for r in rows})==1,
      'trace_folder':traced['saved'],'runs':rows}
result={'short':summarize_suite(short),'long':summarize_suite(long)}
for key in ['short','long']:
    result[key]['analysis']=json.loads((Path(result[key]['trace_folder'])/'ANALYSIS.json').read_text())
    assert result[key]['all_output_hashes_match']
combined=R/'runs/20261103-175114-trace-sample-n64-ctx3072'
result['combined_host_timing']=json.loads((combined/'ANALYSIS.json').read_text())
result['combined_CPU_sampling']=json.loads((combined/'SAMPLE_ANALYSIS.json').read_text())
result['separate_CPU_sampling']=json.loads((R/'runs/20261103-174144-sample-n96/CPU_SUMMARY.json').read_text())
result['final_checks']=json.loads((R/'final-check.json').read_text())
result['API_restarted']=False;result['production_changed']=False;result['KV_mirroring_implemented']=False
(R/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
print('Short baseline median ms:',result['short']['baseline_median_ms'],'trace slowdown%:',result['short']['trace_slowdown_percent'])
print('Long baseline median ms:',result['long']['baseline_median_ms'],'trace slowdown%:',result['long']['trace_slowdown_percent'])
