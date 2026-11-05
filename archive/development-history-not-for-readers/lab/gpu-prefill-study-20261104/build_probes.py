from pathlib import Path
import subprocess, json, hashlib
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';S=E/'source'
C=L/'v100-prep-20261024/cuda-12.9.2'
P=L/'strata-tc-study-20261031/cupti/cuda_cupti-linux-x86_64-12.9.79-archive'
base=['g++','-O2','-std=c++17','-I'+str(S/'ggml/include'),'-I'+str(C/'include'),'-I'+str(P/'include')]
commands=[
base+['-shared','-fPIC',str(R/'grouped_dispatch.cpp'),'-ldl','-o',str(R/'grouped_dispatch.so')],
base+['-shared','-fPIC',str(R/'activity_trace.cpp'),'-L'+str(P/'lib'),'-Wl,-rpath,'+str(P/'lib'),'-lcupti','-o',str(R/'activity_trace.so')],
base+['-DENABLE_PMU','-I'+str(P/'samples/common'),'-I'+str(P/'samples/range_profiling'),str(R/'operator_probe.cpp'),
      '-L'+str(E/'build/bin'),'-Wl,-rpath,'+str(E/'build/bin'),'-L'+str(C/'lib'),'-Wl,-rpath,'+str(C/'lib'),
      '-L'+str(P/'lib'),'-Wl,-rpath,'+str(P/'lib'),'-lggml-cuda','-lggml-cpu','-lggml-base','-lggml','-lcudart','-lcuda','-lcupti','-ldl','-o',str(R/'operator_probe')]
]
for cmd in commands:
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    print(Path(cmd[-1]).name,p.returncode,p.stderr[-4000:],flush=True)
    if p.returncode:raise SystemExit(p.returncode)
manifest={'commands':commands,'binaries':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [R/'grouped_dispatch.so',R/'activity_trace.so',R/'operator_probe']],
          'source_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.cpp')},'production_changed':False}
(R/'BUILD.json').write_text(json.dumps(manifest,indent=2)+'\n')
