from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parents[1]
b=json.loads((LAB/'optimization-20261116/abs-operator/PROBE_BUILD.json').read_text())['argv']
c=[]
for v in b:
    if v==str(LAB/'optimization-20261116/abs-operator/probe.cpp'):v=str(R/'probe.cpp')
    elif v=='-L'+str(LAB/'optimization-20261116/abs-operator/runtime'):v='-L'+str(R/'runtime')
    elif v.startswith('-Wl,-rpath,'):v=v.replace(str(LAB/'optimization-20261116/abs-operator/runtime'),str(R/'runtime'))
    c.append(v)
c.insert(1,'-I'+str(R));c[c.index('-o')+1]=str(R/'probe');c+=['-pthread']
subprocess.run(c,check=True,timeout=180)
(R/'PROBE_BUILD.json').write_text(json.dumps({'argv':c,'source_sha256':hashlib.sha256((R/'probe.cpp').read_bytes()).hexdigest(),'probe_sha256':hashlib.sha256((R/'probe').read_bytes()).hexdigest()},indent=2)+'\n');print('Built actual graph/collector probe')
