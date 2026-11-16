"""One bounded positive-residency pilot after all-miss and source/scheduler validation."""
from pathlib import Path
import fcntl,hashlib,json,re,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn';ENGINE=R.parent/'deepseek41-bringup-20261101'
def run(name,*args):subprocess.run(['python3',str(R/name),*args],check=True)
def main():
    state={'started_unix':time.time(),'complete':False,'error':None,'phase':'dispatch-proof'};path=OUT/'FINAL_PILOT_CONTROLLER.json';assert not path.exists()
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        offset=(ENGINE/'server.stderr.log').stat().st_size
        run('switch_and_wait.py','hot-miss-proof')
        run('bench_api.py','--label','hot-miss-dispatch-proof','--context','128','--tokens','16')
        with (R/'.benchmark.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            raw=(ENGINE/'server-config.json').read_bytes();c=json.loads(raw)
            assert c['optimization_study']['profile']=='hot-miss-proof'
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));headers={'Authorization':'Bearer '+Path(c['api_key_file']).read_text().strip()}
            with opener.open(urllib.request.Request(f"http://{c['host']}:{c['port']}/slots",headers=headers),timeout=5) as response:assert not any(x.get('is_processing') for x in json.load(response))
            subprocess.run(['systemctl','stop','llm-deepseek41-api.service'],check=True,timeout=60)
            with (ENGINE/'server.stderr.log').open('rb') as f:f.seek(offset);text=f.read().decode(errors='replace')
            rows=[{'layer':int(m[1]),'calls':int(m[2]),'H_histogram':list(map(int,m[3].split(',')))} for m in re.finditer(r'xeon-hot-stats layer=(\d+) calls=(\d+) H=\[([0-9,]+)\]',text)]
            assert len(rows)==37 and {x['layer'] for x in rows}==set(range(37))
            assert len({x['calls'] for x in rows})==1 and all(x['calls']>=15 and x['H_histogram']==[x['calls'],0,0,0,0,0,0] for x in rows)
            proof={'passed':True,'config_sha256':hashlib.sha256(raw).hexdigest(),'layers':rows,'scope':'Same frozen engine/profile as hot-miss,only -lv4 added; library INFO maps to TRACE4 so earlier -lv3 ABBA suppressed counters. This separate short run proves dispatch, not a replacement speed measurement.'}
            (OUT/'MISS_DISPATCH.json').write_text(json.dumps(proof,indent=2)+'\n')
        state['phase']='hot14-start';save();run('switch_and_wait.py','hot14-trace')
        state['phase']='hot14-evaluation';save()
        run('bench_api.py','--label','hot14-warm','--context','1024','--tokens','8')
        run('fresh_probe.py','--label','hot14-r1')
        run('hot_quality.py','--label','hot-quality14')
        run('analyze_hot_quality.py','--control','hot-quality-control','--candidate','hot-quality14')
        run('bench_api.py','--label','hot14-prefill','--context','4096','--tokens','16')
        run('vision_probe.py','--label','hot14-newspaper')
        run('long_cache_probe.py','--label','hot14-16k')
        state['complete']=True;state['phase']='awaiting-root-decision-no-automatic-adoption'
    except BaseException as e:state['error']=repr(e);raise
    finally:state['finished_unix']=time.time();save()
if __name__=='__main__':main()
