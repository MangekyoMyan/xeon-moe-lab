from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
c=json.loads((R/'PROBE_BUILD.json').read_text())['argv']
c[c.index(str(R/'probe.cpp'))]=str(R/'fixed_probe.cpp');c[c.index('-o')+1]=str(R/'fixed_probe')
subprocess.run(c,check=True,timeout=180)
(R/'FIXED_BUILD.json').write_text(json.dumps({'argv':c,'source_sha256':hashlib.sha256((R/'fixed_probe.cpp').read_bytes()).hexdigest(),'probe_sha256':hashlib.sha256((R/'fixed_probe').read_bytes()).hexdigest()},indent=2)+'\n')
print('Built fixed-mode cross-library probe')
