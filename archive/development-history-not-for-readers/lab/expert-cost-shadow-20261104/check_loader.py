from pathlib import Path
import hashlib,json,os,signal,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101'
def main():
    config=json.loads((E/'server-config.json').read_text());key=Path(config['api_key_file']).read_text().strip()
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def check():
        req=urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",headers={'Authorization':'Bearer '+key})
        with http.open(req,timeout=3) as f:
            if any(s.get('is_processing') for s in json.load(f)):raise RuntimeError('Production API busy')
        d=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());g=d['gpu']
        if time.time()-d['unix']>45 or d['errors'] or d['active_alerts'] or g['ecc_mode']!='Enabled' or abs(g['power_limit_W']-150)>.5 or g['gpu_C']>=58 or g['hbm_C']>=63:raise RuntimeError('Monitor boundary')
    check();folder=R/'loader-check'/str(time.time_ns());folder.mkdir(parents=True)
    results=[]
    for mode in ['original','observer_off','observer_on']:
        check();env=os.environ.copy();env.update(config['environment']);env.pop('XEON_SHADOW_LOG',None);env.pop('LD_PRELOAD',None)
        if mode!='original':env['LD_LIBRARY_PATH']=str(R/'runtime')+':'+env['LD_LIBRARY_PATH']
        if mode=='observer_on':env['XEON_SHADOW_LOG']=str(folder/'routes.jsonl')
        env['XEON_TEST_LOGITS_FILE']=str(folder/(mode+'.f32'))
        cmd=['python3',str(L/'moe-implementation-20261026/release/without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1',str(R/'capture_loader'),str(L/'iq-cache-next-20261103/tiny-iq3.gguf'),'gpu-stream']
        started=time.monotonic();child=None
        try:
            with (folder/(mode+'.stdout')).open('w') as out,(folder/(mode+'.stderr')).open('w') as err:
                child=subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
                while child.poll() is None:
                    check()
                    if time.monotonic()-started>90:raise RuntimeError('tiny loader time bound')
                    try:child.wait(timeout=1)
                    except subprocess.TimeoutExpired:pass
                if child.returncode:raise RuntimeError('loader failed: '+str(folder/(mode+'.stderr')))
        finally:
            if child and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        paths=[line.split(': ',1)[1] for line in (folder/(mode+'.stdout')).read_text().splitlines() if line.startswith('LLAMA provider: ')]
        if len(paths)!=1:raise RuntimeError('Actual llama library origin missing')
        mapped=Path(paths[0]).resolve();expected=(L/'iq-cache-next-20261103/runtime/libllama.so.0.8.0' if mode=='original' else R/'runtime/libllama.so.0.8.0').resolve()
        if mapped!=expected:raise RuntimeError('Unexpected mapped llama library: '+str(mapped))
        raw=(folder/(mode+'.f32')).read_bytes()
        if len(raw)!=(512+3)*32*4:raise RuntimeError('Missing tiny-model logits')
        results.append({'mode':mode,'elapsed_s':time.monotonic()-started,'logits_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
            'mapped_llama_library':str(mapped),'mapped_llama_sha256':hashlib.sha256(mapped.read_bytes()).hexdigest()})
    if len({x['logits_sha256'] for x in results})!=1:raise RuntimeError('Observer changed tiny-model logits')
    lines=[json.loads(x) for x in (folder/'routes.jsonl').read_text().splitlines()];obs=[x for x in lines if x['kind']=='route_observation']
    if not obs or not lines[-1].get('final'):raise RuntimeError('No complete callback observations')
    for row in obs:
        assert sum(row['histogram'])==row['tokens']*row['topk'] and not row['assignment_changed']
        assert row['tokens']==512
        expected=sum(n>0 for n in row['histogram'])*row['expert_bytes']
        expected+=sum(bool(n) and e<row['experts']-1 and not row['histogram'][e+1] for e,n in enumerate(row['histogram']))*min(row['expert_bytes'],512)
        assert row['submitted_H2D_bytes']==expected
    final={'passes':results,'logits_bit_identical':True,'observations':len(obs),'capture_status':lines[-1],'folder':str(folder),
      'observer_library_sha256':results[-1]['mapped_llama_sha256'],
      'loader_source_sha256':hashlib.sha256((R/'capture_loader.cpp').read_bytes()).hexdigest(),
      'fixture_sha256':hashlib.sha256((L/'iq-cache-next-20261103/tiny-iq3.gguf').read_bytes()).hexdigest(),
      'scope':'tiny synthetic model; not full DeepSeek speed or quality validation'}
    (R/'LOADER_CHECK.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2))
if __name__=='__main__':main()
