from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sparse-cache'
cmd=json.loads((OUT/'TEST_BUILD.json').read_text())
cmd=[str(OUT/'cost-probe.cpp') if x==str(OUT/'test-sparse.cpp') else str(OUT/'cost-probe') if x==str(OUT/'test-sparse') else x.replace(str(OUT/'runtime'),str(OUT/'release-v2-a')) for x in cmd]
if '-fopenmp' not in cmd:cmd.insert(1,'-fopenmp')
subprocess.run(cmd,check=True,timeout=120)
(OUT/'COST_BUILD.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((OUT/'cost-probe.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((OUT/'cost-probe').read_bytes()).hexdigest()},indent=2)+'\n')
