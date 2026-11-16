"""Compile the bounded residency fixture, no service or configuration changes."""
from pathlib import Path
import fcntl,hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent
def main():
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=R/'hot-ffn';out.mkdir(exist_ok=True)
    engine=LAB/'deepseek41-bringup-20261101';cuda=LAB/'v100-prep-20261024/cuda-12.9.2'
    cmd=['g++','-O2','-std=c++20','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-I'+str(engine/'source/ggml/include'),'-I'+str(cuda/'include'),str(R/'hot_ffn_probe.cpp'),'-L'+str(engine/'build/bin'),'-Wl,-rpath,'+str(engine/'build/bin'),'-L'+str(cuda/'lib'),'-Wl,-rpath,'+str(cuda/'lib'),'-lggml-cuda','-lggml-cpu','-lggml-base','-lggml','-lcudart','-ldl','-o',str(out/'probe')]
    subprocess.run(cmd,check=True,timeout=180)
    (out/'BUILD.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((R/'hot_ffn_probe.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((out/'probe').read_bytes()).hexdigest()},indent=2)+'\n')
    print('Built bounded hot-expert FFN fixture.',flush=True)
if __name__=='__main__':main()
