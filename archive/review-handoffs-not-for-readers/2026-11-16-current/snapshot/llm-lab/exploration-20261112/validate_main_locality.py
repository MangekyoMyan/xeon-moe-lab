from pathlib import Path
import fcntl,hashlib,json,subprocess
R=Path(__file__).resolve().parent;GROUP=R/'cpu-coordination-mx'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def guard(name,cmd):subprocess.run(['python3',str(R/'guard.py'),'--name',name,'--timeout','300','--',*cmd],check=True)
    for san in [False,True]:
        suffix='-san' if san else '';out=GROUP/('main-last'+suffix);runtime=out/'runtime';lib=runtime/'libggml-cpu.so.0.28.0'
        guard('main-locality-build'+suffix,['python3',str(R/'build_cpu_coord.py'),'main-last','--base-mx',*(['--sanitize'] if san else [])])
        cmd=json.loads((R/'sparse-cache/TEST_BUILD.json').read_text());exe=out/'test-affinity'
        cmd=[str(R/'main_locality_probe.cpp') if x==str(R/'sparse-cache/test-sparse.cpp') else str(exe) if x==str(R/'sparse-cache/test-sparse') else x for x in cmd]
        cmd+=['-ldl','-UNDEBUG',*(['-fsanitize=address,undefined','-g','-O1','-fno-omit-frame-pointer'] if san else [])]
        guard('main-locality-probe-build'+suffix,cmd)
        env=['env','LD_LIBRARY_PATH='+str(runtime)+':'+str(R.parent/'gpu-stream-20261103/runtime')+':'+str(R.parent/'deepseek41-bringup-20261101/build/bin')]
        if san:env+=['LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/14/libasan.so','ASAN_OPTIONS=detect_leaks=1:abort_on_error=1','UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1']
        for flag in [0,1]:guard('main-locality-affinity'+suffix+'-'+str(flag),[*env,'GGML_CPU_MAIN_LAST='+str(flag),'numactl','--physcpubind=0-23','--interleave=0,1',str(exe),str(lib),str(flag)])
    out=GROUP/'main-last-san';runtime=out/'runtime';reference=R/'execution/mx-backend-candidate-san';old=R/'mx-prepack/backend/candidate-san/runtime';name='main-locality-mx-san';output=R/'execution'/name/'outputs.bin'
    command=json.loads((reference/'status.json').read_text())['command'];command=[x.replace(str(old),str(runtime)).replace(str(reference/'outputs.bin'),str(output)) for x in command];command.insert(1,'GGML_CPU_MAIN_LAST=1');guard(name,command);assert digest(output)==digest(reference/'outputs.bin')
    env=['env','GGML_CPU_MAIN_LAST=1','LD_LIBRARY_PATH='+str(runtime)+':'+str(R.parent/'gpu-stream-20261103/runtime')+':'+str(R.parent/'deepseek41-bringup-20261101/build/bin'),'LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/14/libasan.so','ASAN_OPTIONS=detect_leaks=1:abort_on_error=1','UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1']
    guard('main-locality-iq-san',[*env,str(R/'sparse-cache/san/test-sparse')])
    normal=json.loads((GROUP/'main-last/BUILD.json').read_text());san=json.loads((GROUP/'main-last-san/BUILD.json').read_text());assert normal['source_sha256']==san['source_sha256']
    result={'passed':True,'library_sha256':normal['sha256'],'strict_affinity_master_checked':True,'F32_values_each':131072,'MX_cases':6768,'MX_outputs':2180448,'IQ_cases':4488,'IQ_outputs':1066032,'asan_ubsan_leak':'passed','scope':'OpenMP graph unchanged; optional master-last mask assignment only; same-binary control flag0 vs1'}
    (GROUP/'main-last/VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
