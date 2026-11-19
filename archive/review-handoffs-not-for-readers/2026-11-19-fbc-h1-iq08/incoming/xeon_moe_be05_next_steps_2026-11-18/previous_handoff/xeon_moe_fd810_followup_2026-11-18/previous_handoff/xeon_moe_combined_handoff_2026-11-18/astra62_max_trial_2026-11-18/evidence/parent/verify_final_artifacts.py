"""Read-only parent verification of frozen input, child checksums, DSOs and full outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

ap=argparse.ArgumentParser()
ap.add_argument('--trial',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();t=a.trial
source=json.loads((t/'SOURCE_MANIFEST.json').read_text())
for r in source['files']:
    p=t/'source_repo'/r['path']
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
out=t/'outputs';checks=0
for line in (out/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1)
    assert sha(out/name)==digest,name
    checks+=1
manifest=json.loads((out/'RUN_MANIFEST.json').read_text())
dsos=[]
for v in manifest['timing_final_clean_builds']:
    d=t/v['directory_at_VMs_run']
    for typ,name in [('cpu','libggml-cpu.so.0.28.0'),('base','libggml-base.so.0.28.0')]:
        actual=sha(d/name)
        assert actual==v[f'{typ}_dso_sha256']
        dsos.append({'variant':v['name'],'component':typ,'sha256':actual})
variants=[v['name'] for v in manifest['timing_final_clean_builds']]
final=manifest['ordinary_and_rounding_final'];groups=[]
for suite,expected_rows in [('check',153),('edges',33),('round0',48),('round1',48),('round2',48),('round3',48)]:
    files=[]
    reference=(t/'work/final-evidence'/f'baseline-{suite}.bin').read_bytes()
    for v in variants:
        p=t/'work/final-evidence'/f'{v}-{suite}.bin';c=out/'raw/final-checks'/f'{v}-{suite}.csv'
        assert p.read_bytes()==reference
        assert sha(p)==final['sha256_by_dump'][p.name]
        assert sha(c)==final['sha256_by_csv'][c.name]
        rows=list(csv.DictReader(c.open()));assert len(rows)==expected_rows
        files.append({'variant':v,'dump_bytes':p.stat().st_size,'dump_sha256':sha(p),'case_rows':len(rows)})
    groups.append({'suite':suite,'floats_per_variant':len(reference)//4,'all_full_outputs_equal':True,'files':files})
record={
    'scope_ja':'親がファイルを実読取しSHA/全byteを照合。新しいベンチ実行ではない。',
    'original_input_files_unchanged':len(source['files']),
    'original_input_bytes_unchanged':sum(r['bytes'] for r in source['files']),
    'child_output_files_matching_declared_checksums':checks,
    'child_checksums_sha256':sha(out/'SHA256SUMS'),
    'final_DSO_identity_checks':dsos,
    'final_output_suites':groups,
    'normal_and_edges_cases_per_variant':186,
    'normal_and_edges_floats_per_variant':sum(g['floats_per_variant'] for g in groups[:2]),
    'rounding_cases_per_mode_per_variant':48,
    'rounding_floats_per_mode_per_variant':groups[2]['floats_per_variant'],
    'all_passed':True,
}
a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['final_DSO_identity_checks','final_output_suites']},ensure_ascii=False))
