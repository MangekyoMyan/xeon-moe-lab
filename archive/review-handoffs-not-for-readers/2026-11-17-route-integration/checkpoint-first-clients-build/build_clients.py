from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent;S=LAB/'deepseek41-bringup-20261101/source';B=LAB/'deepseek41-bringup-20261101/build/bin'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
commands=[]
for name in ['integration_probe','driver']:
    c=['g++','-std=c++17','-O2','-g','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-fopenmp','-pthread']
    for inc in [R,R/'route37',S/'include',S/'ggml/include',S/'ggml/src',S/'common',S/'src',S/'tools/mtmd',S/'vendor/nlohmann']:c+=['-I'+str(inc)]
    c += [str(R/(name+'.cpp')),'-L'+str(R/'runtime'),'-L'+str(B),'-Wl,-rpath,'+str(R/'runtime')+':'+str(B),'-Wl,--no-as-needed','-lllama','-lggml-cpu','-lggml-base','-lggml','-ldl']
    if name=='driver':c+=['-lllama-common','-lmtmd']
    c += ['-Wl,--no-undefined','-o',str(R/name)]
    subprocess.run(c,check=True,timeout=180);commands.append(c)
(R/'CLIENT_BUILD.json').write_text(json.dumps({'commands':commands,'binaries':{name:sha(R/name) for name in ['integration_probe','driver']},'source_sha256':{name:sha(R/name) for name in ['integration_probe.cpp','driver.cpp','route_client.h','call_control.h','fixture.h']},'model_executed':False},indent=2)+'\n');print('Built scheduler/caller probe and dedicated driver')
