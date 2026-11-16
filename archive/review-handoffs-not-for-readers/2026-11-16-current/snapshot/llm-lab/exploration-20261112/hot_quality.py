"""Fixed-prefix T=1 replay: numerical/greedy comparison, not a quality benchmark."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,threading,time,urllib.request
from heldout_probe import ROOT as R, CONFIG, helpers, private_write
OUT=R/'hot-ffn'
COMMON={'id_slot':0,'stream':False,'return_tokens':True,'temperature':0,'samplers':['temperature'],'mirostat':0,'seed':42,'ignore_eos':True,'stop':[],'logit_bias':[],'backend_sampling':False,'post_sampling_probs':False,'n_cache_reuse':0}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--freeze',action='store_true');parser.add_argument('--steps',type=int,default=32);a=parser.parse_args()
    assert a.label.replace('-','').isalnum() and 1<=a.steps<=32
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();c=json.loads(raw);profile=c['optimization_study']['profile'];pid=helpers.api_pid()
    key=Path(c['api_key_file']).read_text().strip();headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url=f"http://{c['host']}:{c['port']}"
    def query(route,body=None):
        with opener.open(urllib.request.Request(url+route,headers=headers,data=None if body is None else json.dumps(body).encode()),timeout=240) as response:return json.load(response)
    assert query('/health')['status']=='ok' and not any(x.get('is_processing') for x in query('/slots'))
    inputs=R/'FRESH_INPUTS.json';source=json.loads(inputs.read_text());frozen=OUT/'QUALITY_SEQUENCES.json'
    folder=OUT/'quality'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}';folder.mkdir(parents=True,mode=0o700)
    if os.geteuid()==0:os.chown(folder,R.stat().st_uid,R.stat().st_gid)
    initial=helpers.watch();stop=threading.Event();failures=[]
    record={'label':a.label,'profile':profile,'pid':pid,'config_sha256':hashlib.sha256(raw).hexdigest(),'input_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),'steps_per_case':a.steps,'n_probs':100,'scope':'Append known baseline tokens, never candidate samples. Every scored request processes exactly1newtoken. Partial-topK,not PPL/TVD/KL/generalquality.', 'cases':[],'complete':False,'error':None,'started_unix':time.time()}
    private_write(folder/'RESULT.json',record)
    def monitor():
        while not stop.wait(1):
            try:
                state=helpers.watch()
                if state['host_counters']!=initial['host_counters']:raise RuntimeError('Hardware counters changed')
            except Exception as e:failures.append(repr(e));return
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    def completion(body):
        if failures:raise RuntimeError(failures[0])
        assert CONFIG.read_bytes()==raw
        response=query('/completion',{**COMMON,**body})
        assert not response.get('error') and not response.get('truncated',False)
        if failures:raise RuntimeError(failures[0])
        return response
    try:
        if a.freeze:
            assert profile in ['final','hot-disabled'] and not frozen.exists()
            data={'source_profile':profile,'source_config_sha256':record['config_sha256'],'input_sha256':record['input_sha256'],'sampling':COMMON,'n_probs':100,'cases':[]}
            for case in source['cases']:
                response=completion({'prompt':case['tokens'],'n_predict':64,'n_probs':100,'cache_prompt':False})
                assert len(response['tokens'])==response['timings']['predicted_n']==64 and response['timings']['cache_n']==0
                data['cases'].append({'name':case['name'],'prompt_tokens':case['tokens'],'continuation_tokens':response['tokens']})
            private_write(frozen,data)
        data=json.loads(frozen.read_text());assert data['input_sha256']==record['input_sha256'] and data['n_probs']==100 and data['sampling']==COMMON
        record['sequence_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest()
        for case in data['cases']:
            assert helpers.api_pid()==pid and not any(x.get('is_processing') for x in query('/slots'))
            prompt=case['prompt_tokens'];known=case['continuation_tokens'];length=len(prompt)
            warm=completion({'prompt':prompt,'n_predict':1,'n_probs':0,'cache_prompt':False})
            assert len(warm['tokens'])==1 and warm['timings']['cache_n']==0 and warm['timings']['prompt_n']==length and warm['tokens_cached']==length
            row={'name':case['name'],'warm_timings':warm['timings'],'points':[]}
            for j in range(a.steps):
                start=time.monotonic();reply=completion({'prompt':prompt+known[:j+1],'n_predict':1,'n_probs':100,'cache_prompt':True});wall=time.monotonic()-start
                t=reply['timings'];assert len(reply['tokens'])==t['predicted_n']==1 and t['cache_n']==length+j and t['prompt_n']==1 and reply['tokens_cached']==length+j+1
                assert reply['stop_type']=='limit'
                probabilities=reply.get('completion_probabilities') or []
                assert len(probabilities)<=1
                if probabilities:assert probabilities[0]['id']==reply['tokens'][0]
                row['points'].append({'j':j,'known_next_id':known[j+1],'generated_id':reply['tokens'][0],'matches_frozen_baseline':reply['tokens'][0]==known[j+1],'probability':probabilities[0] if probabilities else None,'timings':t,'wall_s':wall})
            record['cases'].append(row);private_write(folder/'RESULT.json',record,replace=True)
            print(json.dumps({'case':case['name'],'steps':len(row['points']),'known_token_agreement':sum(x['matches_frozen_baseline'] for x in row['points']),'probability_records':sum(x['probability'] is not None for x in row['points'])}),flush=True)
        assert helpers.api_pid()==pid and not failures
        record['complete']=True
    except BaseException as e:record['error']=repr(e);raise
    finally:
        stop.set();watcher.join(timeout=2);record['finished_unix']=time.time();private_write(folder/'RESULT.json',record,replace=True)
    print('Saved '+str(folder),flush=True)
if __name__=='__main__':main()
