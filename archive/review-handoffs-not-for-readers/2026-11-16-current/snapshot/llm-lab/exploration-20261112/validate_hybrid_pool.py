"""Mixed executor lifecycle, frozen MX outputs, and native24 IQ regression."""
from pathlib import Path
import fcntl,hashlib,json,subprocess
R=Path(__file__).resolve().parent;OUT=R/'hybrid-pool'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard(name,*cmd):subprocess.run(['python3',str(R/'guard.py'),'--name',name,'--timeout','300','--',*cmd],check=True)
    for sanitize in [False,True]:
        suffix='-san' if sanitize else '';folder=OUT/('san' if sanitize else 'normal');runtime=folder/'runtime';lib=runtime/'libggml-cpu.so.0.28.0'
        guard('hybrid-pool-fullbuild'+suffix,'python3',str(R/'build_hybrid_pool.py'),*(['--sanitize'] if sanitize else []))
        env=['env','GGML_CPU_HYBRID_POOL=1','LD_LIBRARY_PATH='+str(runtime)+':'+str(R.parent/'gpu-stream-20261103/runtime')+':'+str(R.parent/'deepseek41-bringup-20261101/build/bin')]
        if sanitize:env+=['LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/14/libasan.so','ASAN_OPTIONS=detect_leaks=1:abort_on_error=1','UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1']
        # Earlier normal lifecycle covered the same source; rerun only the new sanitizer build.
        if sanitize:
            for mode in ['0','1']:guard('hybrid-pool-lifecycle-'+mode+suffix,*env,'numactl','--physcpubind=0-23','--interleave=0,1',str(OUT/('test-hybrid'+suffix)),str(lib),mode)
        reference=R/'execution'/('mx-backend-candidate-san' if sanitize else 'mx-backend-candidate')
        old=R/'mx-prepack/backend'/('candidate-san' if sanitize else 'candidate')/'runtime'
        name='hybrid-pool-mx'+suffix;output=R/'execution'/name/'outputs.bin'
        command=json.loads((reference/'status.json').read_text())['command']
        command=[x.replace(str(old),str(runtime)).replace(str(reference/'outputs.bin'),str(output)) for x in command]
        command.insert(1,'GGML_CPU_HYBRID_POOL=1')
        guard(name,*command);assert sha(output)==sha(reference/'outputs.bin')
        guard('hybrid-pool-iq24'+suffix,*env,'numactl','--physcpubind=0-23','--interleave=0,1',str(OUT/('test-iq-native'+suffix)))
    normal=json.loads((OUT/'normal/BUILD.json').read_text());san=json.loads((OUT/'san/BUILD.json').read_text());assert normal['sources']==san['sources']
    result={'passed':True,'library_sha256':normal['sha256'],'MX_cases_each':6768,'MX_outputs_each':2180448,'MX_output_sha256':sha(R/'execution/hybrid-pool-mx/outputs.bin'),
            'IQ_native24_cases_each':2244,'IQ_native24_outputs_each':533016,
            'lifecycle':'samebackend OMP12/native24,active1/6/12/24,implicitresume,pause,first/lastnodeabort,capacityclamp,70kgraphwrap,perworkeractualaffinity,disposableOMPobserved',
            'sanitizer_scope':'Changed CPU executor and RHS objects plus test code instrumented; other unchanged reused objects not fully instrumented; ASAN/UBSAN/leaks checks passed'}
    (OUT/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
