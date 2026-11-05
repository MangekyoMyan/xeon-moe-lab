from pathlib import Path
import json,subprocess,hashlib
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';C=L/'v100-prep-20261024/cuda-12.9.2'
cmd=['g++','-O2','-std=c++17','-I'+str(E/'source/ggml/include'),'-I'+str(C/'include'),str(R/'calibration_probe.cpp'),
     '-L'+str(E/'build/bin'),'-Wl,-rpath,'+str(E/'build/bin'),'-L'+str(C/'lib'),'-Wl,-rpath,'+str(C/'lib'),
     '-lggml-cpu','-lggml-cuda','-lggml-base','-lggml','-lcudart','-o',str(R/'calibration_probe')]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=60);print(p.returncode,p.stderr[-3000:])
if p.returncode:raise SystemExit(p.returncode)
(R/'BUILD.json').write_text(json.dumps({'command':cmd,'binary_sha256':hashlib.sha256((R/'calibration_probe').read_bytes()).hexdigest(),
    'source_sha256':hashlib.sha256((R/'calibration_probe.cpp').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((R/'route_features.h').read_bytes()).hexdigest()},indent=2)+'\n')
