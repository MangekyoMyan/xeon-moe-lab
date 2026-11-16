"""Build the exact private backend into a small real-weight scheduler test."""
from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn';LAB=R.parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--sanitize',action='store_true');a=p.parse_args()
    suffix='-san' if a.sanitize else '';engine=LAB/'deepseek41-bringup-20261101';cuda=LAB/'v100-prep-20261024/cuda-12.9.2'
    cmd=['g++','-O2','-std=c++20','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-I'+str(OUT),'-I'+str(engine/'source/src'),'-I'+str(engine/'source/include'),'-I'+str(engine/'source/ggml/include'),'-I'+str(engine/'source/ggml/src'),'-I'+str(cuda/'include'),str(OUT/'test-backend.cpp'),'-L'+str(engine/'build/bin'),'-Wl,-rpath,'+str(engine/'build/bin'),'-L'+str(cuda/'lib'),'-Wl,-rpath,'+str(cuda/'lib'),'-lllama','-lggml-cuda','-lggml-cpu','-lggml-base','-lggml','-lcudart','-ldl','-pthread','-o',str(OUT/('test-backend'+suffix))]
    if a.sanitize:cmd+=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g','-O1']
    subprocess.run(cmd,check=True,timeout=240)
    (OUT/('TEST_BUILD'+suffix+'.json')).write_text(json.dumps({'command':cmd,'scope':'Private provider+test instrumented;reused ggml/CUDA/llama libraries unchanged','source_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['test-backend.cpp','hot-backend.cpp','hot-backend.h','fixture-inputs.h']},'binary_sha256':hashlib.sha256((OUT/('test-backend'+suffix)).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Built dedicated-backend scheduler test'+suffix,flush=True)
if __name__=='__main__':main()
