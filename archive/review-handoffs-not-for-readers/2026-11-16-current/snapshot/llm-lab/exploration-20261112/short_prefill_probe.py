"""Small batch boundary probe, complete native output IDs, no settings changes."""
from pathlib import Path
import argparse,fcntl,hashlib,json,time,urllib.request
from heldout_probe import ROOT,CONFIG,helpers,private_write
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();assert a.label.replace('-','').isalnum()
    lock=(ROOT/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();config=json.loads(raw);key=Path(config['api_key_file']).read_text().strip()
    url=f"http://{config['host']}:{config['port']}";headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def query(route,body=None):
        with opener.open(urllib.request.Request(url+route,headers=headers,data=None if body is None else json.dumps(body).encode()),timeout=180) as response:return json.load(response)
    assert query('/health')['status']=='ok' and not any(s.get('is_processing') for s in query('/slots'))
    def tokens(text,special=False):return query('/tokenize',{'content':text,'add_special':special})['tokens']
    prefix=tokens('Notes:',True);tail=tokens('\nContinue: 1, 2, 3, 4,');filler=tokens(' A numbered list follows.')
    assert len(prefix)+len(tail)<32
    folder=ROOT/'short-prefill'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}';folder.mkdir(parents=True)
    pid=helpers.api_pid();record={'label':a.label,'pid':pid,'started_unix':time.time(),'config_sha256':hashlib.sha256(raw).hexdigest(),'cases':[],'error':None,'complete':False,'timing_scope':'Server prompt_ms; nonstream full output tokens'}
    private_write(folder/'RESULT.json',record)
    try:
        for count in [32,48,63,64]:
            initial=helpers.watch();assert CONFIG.read_bytes()==raw and helpers.api_pid()==pid
            assert not any(s.get('is_processing') for s in query('/slots'))
            fill=count-len(prefix)-len(tail);prompt=prefix+(filler*((fill+len(filler)-1)//len(filler)))[:fill]+tail
            start=time.monotonic();result=query('/completion',{'prompt':prompt,'n_predict':16,'ignore_eos':True,'temperature':0,'seed':42,'stream':False,'return_tokens':True,'cache_prompt':False})
            assert not result.get('error') and not result.get('truncated',False)
            timing=result['timings'];assert timing['cache_n']==0 and timing['prompt_n']==count and timing['predicted_n']==len(result['tokens'])==16
            after=helpers.watch();assert initial['host_counters']==after['host_counters']
            record['cases'].append({'context':count,'prompt_tokens':prompt,'timings':timing,'wall_seconds':time.monotonic()-start,'output':result['content'],'tokens':result['tokens'],'watch_after':after})
            private_write(folder/'RESULT.json',record,replace=True)
            print(json.dumps({'context':count,'prompt_ms':timing['prompt_ms'],'decode_tok_s':timing['predicted_per_second']}),flush=True)
        record['complete']=True
    except BaseException as error:record['error']=repr(error);raise
    finally:private_write(folder/'RESULT.json',record,replace=True)
    print('Saved '+str(folder),flush=True)
if __name__=='__main__':main()
