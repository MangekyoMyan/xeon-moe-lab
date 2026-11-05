#!/usr/bin/env python3
import hashlib,json,zipfile
from pathlib import Path
from full_replica_experiment import LAB,ROOT,OUT
archive=LAB.parent/'xeon-full-replica-26b-20261020.zip'
assert not archive.exists()
prefs=LAB/'experiment-preferences.json';old=prefs.read_text();data=json.loads(old)
(OUT/'preferences-before-final-update.json').write_text(old)
data['before_full_replica_work']=data['current_work']
data['current_work']='Completed full-weight NUMA replication tests on Melody1437-26B-A4B Q6_K. Full replication improved decode 11.22% (9.049 to 10.064 t/s), but single-copy 4 KiB interleave reached 10.005 t/s; with 4 KiB pages in both cases full replication added 1.82% (10.186 t/s). Remote retired DRAM-load-uop share fell 47.39% to 7.58%; fine single-copy retained 47.85% while improving DRAM read rate. No DeepSeek extrapolation or default adoption. Normal runtime unchanged; two separate experimental Melody runners available. All full-model logits and Japanese responses matched.'
data['full_replica_results']=str(OUT/'REPORT_JA.txt')
data['full_replica_archive']=str(archive)
prefs.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(OUT/'preferences-after-final-update.json').write_text(prefs.read_text())
files={}
allowed={'.txt','.json','.log','.csv','.patch','.cpp','.h','.c','.stdout','.stderr'}
for p in OUT.rglob('*'):
 if p.is_file() and p.suffix in allowed:files['results/'+str(p.relative_to(OUT))]=p
for name in ['prepare_full_replica.py','full_replica_experiment.py','validate_full_replica.py','full_replica_responses.py','observe_full_replica.py','observe_phase.py','run_full_replica_counters.py','without_thp.py','finish_full_replica.py','query_full_replica_pages.py','summarize_full_replica.py','summarize_bottleneck_observations.py','finalize_full_replica.py','package_full_replica.py','build_deep_candidate.py','bench_deepseek_light.py','deepseek-logits-probe.cpp','numa-replica-probe.cpp','deepseek_tensors.py','index_model_cache.py','run_melody_full_replica_experiment.sh','run_melody_fine_interleave_experiment.sh']:
 files['tools/'+name]=LAB/name
for n in json.loads((OUT/'original-hashes.json').read_text()):
 files['source-before/'+n]=LAB/'llama-private-prefetch'/n
 files['source-after/'+n]=ROOT/'source'/n
for name in ['build-record.json','configure.log','build.log']:files['build/'+name]=ROOT/name
manifest={n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in sorted(files.items())}
readme='''Xeon / Melody 26B-A4B full NUMA replication experiment, 2026-10-20.
Start with results/REPORT_JA.txt and results/REPORT.json.
Results include baseline controls, complete model-output checks, physical page residency, PMU observations, and source changes.
Tools use the original server paths. Commands and environments are saved in individual summaries.
Model weights, compiled binaries, and raw logits are excluded. Logit hashes and equality checks are included.
The normal runtime was not replaced. Scripts are private Linux CPU experiments for node0/node1 and the supplied model.
MANIFEST.json contains the SHA-256 and size of every payload file except this README and the manifest itself.
'''
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.writestr('README.txt',readme);z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 for n,p in sorted(files.items()):z.write(p,n)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for n,m in manifest.items():
  b=z.read(n);assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256']
result={'archive':str(archive),'bytes':archive.stat().st_size,'payload_files':len(files),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'crc_and_payload_hashes_verified':True}
archive.with_suffix('.package-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
