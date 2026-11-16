"""Final frozen prompt set, complete native output tokens via nonstreaming responses."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,time,urllib.request
from heldout_probe import ROOT,CONFIG,helpers,private_write
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--freeze-only',action='store_true');a=parser.parse_args()
    assert a.label.replace('-','').isalnum()
    lock=(ROOT/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();config=json.loads(raw);key=Path(config['api_key_file']).read_text().strip()
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    url=f"http://{config['host']}:{config['port']}";opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def query(route,body=None):
        with opener.open(urllib.request.Request(url+route,headers=headers,data=None if body is None else json.dumps(body,ensure_ascii=False).encode()),timeout=240) as response:return json.load(response)
    assert query('/health')['status']=='ok' and not any(s.get('is_processing') for s in query('/slots'))
    source=ROOT/'FRESH_PROMPTS.json';spec=json.loads(source.read_text());fixed=ROOT/'FRESH_INPUTS.json'
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if not fixed.exists():
        cases=[]
        for case in spec['cases']:
            formatted=query('/apply-template',{'messages':[{'role':'user','content':case['text']}]})['prompt']
            tokens=query('/tokenize',{'content':formatted,'add_special':True,'parse_special':True})['tokens']
            assert 32<=len(tokens)<1024
            cases.append({'name':case['name'],'tokens':tokens})
        private_write(fixed,{'source_sha256':source_sha,'cases':cases,'generated_tokens':spec['generated_tokens']})
    manifest=json.loads(fixed.read_text());assert manifest['source_sha256']==source_sha
    if a.freeze_only:
        print('Frozen final input token IDs: '+str(fixed),flush=True);return
    folder=ROOT/'heldout'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}';folder.mkdir(parents=True,mode=0o700)
    if os.geteuid()==0:os.chown(folder,ROOT.stat().st_uid,ROOT.stat().st_gid)
    pid=helpers.api_pid()
    record={'label':a.label,'started_unix':time.time(),'config_sha256':hashlib.sha256(raw).hexdigest(),'input_sha256':hashlib.sha256(fixed.read_bytes()).hexdigest(),'cases':[],'complete':False,'error':None,
            'token_collection':'Complete generated token IDs from native nonstream final response','ttft_measurement':'Server prompt_ms proxy; no streaming wall TTFT measured',
            'libraries':sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines() if 'libggml' in line or 'libllama' in line})}
    private_write(folder/'RESULT.json',record)
    try:
        for case in manifest['cases']:
            initial=helpers.watch();assert CONFIG.read_bytes()==raw and helpers.api_pid()==pid
            assert not any(s.get('is_processing') for s in query('/slots'))
            body={'prompt':case['tokens'],'n_predict':manifest['generated_tokens'],'ignore_eos':True,'temperature':0,'seed':42,'cache_prompt':False,'stream':False,'return_tokens':True}
            start=time.monotonic();result=query('/completion',body);wall=time.monotonic()-start
            assert not result.get('error') and not result.get('truncated',False)
            timing=result['timings'];assert timing['cache_n']==0 and timing['prompt_n']==len(case['tokens'])
            assert len(result['tokens'])==timing['predicted_n']==manifest['generated_tokens']
            after=helpers.watch();assert initial['host_counters']==after['host_counters']
            record['cases'].append({'name':case['name'],'prompt_tokens':len(case['tokens']),'timings':timing,'stop_type':result.get('stop_type'),'ttft_s':timing['prompt_ms']/1000,'elapsed_s':wall,
                'output_sha256':hashlib.sha256(result['content'].encode()).hexdigest(),'token_sha256':hashlib.sha256(json.dumps(result['tokens']).encode()).hexdigest(),'output_token_count':len(result['tokens'])})
            private_write(folder/'RESULT.json',record,replace=True)
            print(json.dumps({'name':case['name'],'decode_tok_s':timing['predicted_per_second'],'tokens':len(result['tokens'])}),flush=True)
        record['complete']=True
    except BaseException as error:record['error']=repr(error);raise
    finally:
        record['finished_unix']=time.time();private_write(folder/'RESULT.json',record,replace=True)
    print('Saved '+str(folder),flush=True)
if __name__=='__main__':main()
