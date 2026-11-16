"""Bounded isolated graph/lifecycle checks before any native-pool model run."""
from pathlib import Path
import fcntl,hashlib,json,subprocess
R=Path(__file__).resolve().parent;N=R/'native-pool'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard(name,command):subprocess.run(['python3',str(R/'guard.py'),'--name',name,'--timeout','300','--',*command],check=True)
    for sanitize in [False,True]:
        suffix='-san' if sanitize else '';folder=N/('san' if sanitize else 'normal');runtime=folder/'runtime';lib=runtime/'libggml-cpu.so.0.28.0'
        guard('native-pool-build'+suffix,['python3',str(R/'build_native_pool.py'),*(['--sanitize'] if sanitize else [])])
        env=['env','LD_LIBRARY_PATH='+str(runtime)+':'+str(R.parent/'gpu-stream-20261103/runtime')+':'+str(R.parent/'deepseek41-bringup-20261101/build/bin')]
        if sanitize:env+=['LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/14/libasan.so','ASAN_OPTIONS=detect_leaks=1:abort_on_error=1','UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1']
        guard('native-pool-lifecycle'+suffix,[*env,'numactl','--physcpubind=0-23','--interleave=0,1',str(N/('test-pool'+suffix)),str(lib)])
        reference=R/'execution'/('mx-backend-candidate-san' if sanitize else 'mx-backend-candidate')
        old=R/'mx-prepack/backend'/('candidate-san' if sanitize else 'candidate')/'runtime'
        name='native-pool-mx'+suffix;output=R/'execution'/name/'outputs.bin'
        command=json.loads((reference/'status.json').read_text())['command']
        command=[x.replace(str(old),str(runtime)).replace(str(reference/'outputs.bin'),str(output)) for x in command]
        guard(name,command);assert digest(output)==digest(reference/'outputs.bin')
        test=R/'sparse-cache/san/test-sparse' if sanitize else R/'sparse-cache/test-sparse'
        guard('native-pool-iq'+suffix,[*env,str(test)])
    normal=json.loads((N/'normal/BUILD.json').read_text());san=json.loads((N/'san/BUILD.json').read_text())
    assert normal['sources']==san['sources']
    result={'passed':True,'library_sha256':normal['sha256'],'MX_cases_each':6768,'MX_outputs_each':2180448,'MX_output_sha256':digest(R/'execution/native-pool-mx/outputs.bin'),'IQ_cases_each':4488,'IQ_outputs_each':1066032,
            'pool_lifecycle':'6/12/24, repeated1/full size, pause/resume,empty,abort/recover,free;normalandASANUBSAN','asan_ubsan_leak':'passed','scope':'Graph executor only; OpenMP remains for cache/quantizer initialization; row-local not enabled'}
    (N/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
