"""Run coordination sanitizers on an idle API, retaining old immutable artifacts."""
from pathlib import Path
import fcntl,hashlib,json,subprocess
R=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    group=R/'cpu-coordination-mx';normal=group/'combined';build=json.loads((normal/'BUILD.json').read_text())
    assert digest(normal/'ggml-cpu.c')==build['source_sha256']
    assert digest(normal/'runtime/libggml-cpu.so.0.28.0')==build['sha256']
    def guarded(name,command):
        subprocess.run(['python3',str(R/'guard.py'),'--name',name,'--timeout','300','--',*command],check=True)
    guarded('coord-mx-combined-sanitizer-build',['python3',str(R/'build_cpu_coord.py'),'combined','--base-mx','--sanitize'])
    san=group/'combined-san';assert digest(san/'ggml-cpu.c')==digest(normal/'ggml-cpu.c')
    old=R/'mx-prepack/backend/candidate-san/runtime';new=san/'runtime'
    reference=R/'execution/mx-backend-candidate-san'
    command=json.loads((reference/'status.json').read_text())['command']
    name='coord-mx-combined-mx-sanitized';output=R/'execution'/name/'outputs.bin'
    command=[x.replace(str(old),str(new)).replace(str(reference/'outputs.bin'),str(output)) for x in command]
    guarded(name,command)
    assert digest(output)==digest(reference/'outputs.bin')==digest(R/'execution/coord-mx-combined/outputs.bin')
    env=['env','LD_LIBRARY_PATH='+str(new)+':'+str(R.parent/'gpu-stream-20261103/runtime')+':'+str(R.parent/'deepseek41-bringup-20261101/build/bin'),
         'LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/14/libasan.so','ASAN_OPTIONS=detect_leaks=1:abort_on_error=1','UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1']
    guarded('coord-mx-combined-iq-sanitized',[*env,str(R/'sparse-cache/san/test-sparse')])
    result={'passed':True,'library_sha256':build['sha256'],'source_sha256':build['source_sha256'],'MX_graph_cases':6768,'MX_output_values':2180448,'MX_output_sha256':digest(output),'IQ_cache_cases':4488,'IQ_output_values':1066032,'asan_ubsan_leak':'passed','normal_validation':['execution/coord-mx-combined','execution/coord-mx-combined-iq'],'san_validation':['execution/coord-mx-combined-mx-sanitized','execution/coord-mx-combined-iq-sanitized']}
    (group/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
