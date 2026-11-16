"""Before backend initialization: resolve route/libllama/base pointers and hash backing files."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path(__file__).resolve().parent;LAB=R.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_','LD_','CUDA_'))}
    env['LD_LIBRARY_PATH']=':'.join(map(str,[R/'runtime',LAB/'deepseek41-bringup-20261101/build/bin',LAB/'v100-prep-20261024/cuda-12.9.2/lib']))
    env['LD_BIND_NOW']='1';return env
def main():
    suffix='_FINAL' if '--final' in sys.argv else ''
    out=R/('PROVIDERS'+suffix+'.json');assert not out.exists()
    subprocess.run([str(R/'driver'),'--check-providers',str(out)],env=environment(),check=True,timeout=30)
    d=json.loads(out.read_text());b=json.loads((R/'LINK_BUILD.json').read_text());c=json.loads((R/'CLIENT_BUILD.json').read_text())
    expected_cpu=Path(b['runtime_libraries']['libggml-cpu.so.0.28.0']['path']).resolve();expected_base=Path(b['runtime_libraries']['libggml-base.so.0.28.0']['path']).resolve();expected_llama=(R/'runtime/libllama.so.0.8.0').resolve()
    assert d['backend_init_called'] is False and d['model_loaded'] is False
    observed={}
    for x in d['providers']:
        p=Path(x['path']);st=p.stat();assert (st.st_dev,st.st_ino,st.st_size)==(x['device'],x['inode'],x['bytes'])
        want=expected_cpu if x['symbol'].startswith(('xeon_route37_','ggml_backend_cpu_')) else expected_base if x['symbol'].startswith('ggml_backend_sched_') else expected_llama
        assert p.resolve()==want,(x,want);observed[str(p)]=sha(p)
    assert observed[str(expected_cpu)]==b['runtime_libraries']['libggml-cpu.so.0.28.0']['sha256'] and observed[str(expected_base)]==b['runtime_libraries']['libggml-base.so.0.28.0']['sha256'] and observed[str(expected_llama)]==b['diagnostic_llama_sha256']
    assert sha(R/'driver')==c['binaries']['driver']
    (R/('PROVIDER_VERIFICATION'+suffix+'.json')).write_text(json.dumps({'passed':True,'provider_count':len(d['providers']),'backing_files_sha256':observed,'driver_sha256':sha(R/'driver'),'LD_BIND_NOW':True,'backend_init_called':False,'model_loaded':False,'scope':'Function pointer canonical paths/inodes/hash; no backend enumeration or model/compute initialization called by this check'},indent=2)+'\n');print('All16 provider pointers and backing hashes verified before backend initialization')
if __name__=='__main__':main()
