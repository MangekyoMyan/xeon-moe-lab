"""Stop an idle authorized API, then build/test the dedicated dispatcher in isolation."""
from pathlib import Path
import argparse,fcntl,hashlib,json,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn';CONFIG=R.parent/'deepseek41-bringup-20261101/server-config.json'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args();assert 1<=args.attempt<=9
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=(R/'profiles/final.json').read_bytes();assert CONFIG.read_bytes()==expected
    config=json.loads(expected);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers={'Authorization':'Bearer '+Path(config['api_key_file']).read_text().strip()}
    if subprocess.run(['systemctl','is-active','--quiet','llm-deepseek41-api.service']).returncode==0:
        with opener.open(urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",headers=headers),timeout=5) as response:assert not any(s.get('is_processing') for s in json.load(response))
        subprocess.run(['systemctl','stop','llm-deepseek41-api.service'],check=True,timeout=60)
    assert subprocess.run(['systemctl','is-active','--quiet','llm-deepseek41-api.service']).returncode!=0
    # Build helpers take the same lock themselves; the controller remains the sole caller.
    fcntl.flock(lock,fcntl.LOCK_UN)
    attempt_suffix='' if args.attempt==1 else f'-r{args.attempt}'
    def guard(name,*cmd):subprocess.run(['python3',str(R/'guard.py'),'--name',name+attempt_suffix,'--timeout','300','--',*cmd],check=True)
    subprocess.run(['python3',str(R/'prepare_hot_backend.py')],check=True)
    guard('hot-backend-engine-build','python3',str(R/'build_hot_backend.py'))
    rows=[]
    for sanitize in [False,True]:
        suffix='-san' if sanitize else ''
        guard('hot-backend-test-build'+suffix,'python3',str(R/'build_hot_backend_test.py'),*(['--sanitize'] if sanitize else []))
        env=['env','LD_LIBRARY_PATH='+config['environment']['LD_LIBRARY_PATH'],'CUDA_VISIBLE_DEVICES='+config['environment']['CUDA_VISIBLE_DEVICES']]
        if sanitize:env+=['LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/14/libasan.so','ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:protect_shadow_gap=0','UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1']
        cpu=next(x['file'] for x in config['study_libraries'] if Path(x['file']).name.startswith('libggml-cpu.'))
        gpu=next(x['file'] for x in config['study_libraries'] if Path(x['file']).name.startswith('libggml-cuda.'))
        for layer in [4,24]:
            name=f'hot-backend-L{layer}'+suffix
            guard(name,*env,'numactl','--physcpubind=0-23','--interleave=0,1',str(OUT/('test-backend'+suffix)),str(Path(config['model']).parent),str(layer),cpu,gpu)
            path=R/'execution'/(name+attempt_suffix)/'stdout.log';result=json.loads(path.read_text());assert result['passed'];rows.append({'source':str(path),**result})
    build=json.loads((OUT/'ENGINE_BUILD.json').read_text());normal=json.loads((OUT/'TEST_BUILD.json').read_text());san=json.loads((OUT/'TEST_BUILD-san.json').read_text())
    assert normal['source_sha256']==san['source_sha256']
    assert build['sources']['hot-backend.cpp']==normal['source_sha256']['hot-backend.cpp']
    (OUT/'BACKEND_VALIDATION.json').write_text(json.dumps({'passed':True,'finished_unix':time.time(),'library_sha256':build['sha256'],'tests':rows,'sanitizer_scope':san['scope'],'ASAN_CUDA_compatibility_record':str(OUT/'ASAN_CUDA_COMPAT.json'),'ASAN_options':'detect_leaks=1:abort_on_error=1:protect_shadow_gap=0','API_state':'stopped for isolated tests; next authorized step is experimental all-miss comparison'},indent=2)+'\n')
    print('Dedicated backend validation passed; API remains stopped for next controlled profile.',flush=True)
if __name__=='__main__':main()
