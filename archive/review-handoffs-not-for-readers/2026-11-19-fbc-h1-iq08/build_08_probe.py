from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;D=R/'cpu08';LAB=R.parent;S=LAB/'deepseek41-bringup-20261101/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (D/'BUILD.json').exists() and not (D/'PROBE_BUILD.json').exists()
    flags=['g++','-O3','-std=c++17','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-frounding-math','-fopenmp','-Wall','-Wextra']
    includes=['-I'+str(p) for p in [D,S/'ggml/include',S/'ggml/src',S/'vendor/nlohmann',LAB/'exploration-20261112/row-local']]
    counter=flags+includes+['-shared','-fPIC','-fvisibility=hidden',str(D/'iq08_quality_counts.cpp'),'-ldl','-o',str(D/'quality-counts.so')]
    probe=flags+includes+[str(D/'real_ffn.cpp'),'-L'+str(D/'control/runtime'),'-L'+str(LAB/'gpu-stream-20261103/runtime'),'-Wl,--no-as-needed','-lggml-cpu','-lggml-base','-ldl','-o',str(D/'real_ffn')]
    for name,cmd in [('counter',counter),('probe',probe)]:
        with (D/(name+'.build.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=f,check=True,timeout=180)
    p=dict(commands=[counter,probe],source_sha256={n:sha(D/n) for n in ['real_ffn.cpp','iq08_quality_counts.cpp','iq08_quality_counts.h']},binary_sha256=sha(D/'real_ffn'),counter_sha256=sha(D/'quality-counts.so'))
    (D/'PROBE_BUILD.json').write_text(json.dumps(p,indent=2)+'\n')
    design=json.loads((D/'DESIGN-prebuild.json').read_text());design.update(CPU_build_sha256=sha(D/'BUILD.json'),probe_build_sha256=sha(D/'PROBE_BUILD.json'));(D/'DESIGN.json').write_text(json.dumps(design,indent=2)+'\n')
    (D/'STATE.json').write_text(json.dumps(dict(phase='prepared',runs={},completed=False),indent=2)+'\n')
    print('08 probe/counter built; no numerical/performance execution yet')
if __name__=='__main__':main()
