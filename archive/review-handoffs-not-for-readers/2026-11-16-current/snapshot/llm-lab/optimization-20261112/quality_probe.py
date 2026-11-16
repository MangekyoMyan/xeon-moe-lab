#!/usr/bin/env python3
"""Full-vocabulary next-token distributions on controlled synthetic prefixes only."""
from pathlib import Path
import argparse,array,fcntl,hashlib,importlib.util,json,math,sys,time,urllib.request,zlib
R=Path(__file__).resolve().parent;LAB=R.parent;CONFIG=LAB/'deepseek41-bringup-20261101/server-config.json'
spec=importlib.util.spec_from_file_location('probe_helpers',LAB/'postcompact-20261103/live_probe.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    assert a.label.replace('-','').replace('_','').isalnum()
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();c=json.loads(raw);key=Path(c['api_key_file']).read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));base=f"http://{c['host']}:{c['port']}"
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    def query(path,body=None):
        req=urllib.request.Request(base+path,headers=headers,data=None if body is None else json.dumps(body).encode())
        with opener.open(req,timeout=240) as res:return json.load(res)
    assert query('/health').get('status')=='ok'
    if any(x.get('is_processing') for x in query('/slots')):raise RuntimeError('API is busy')
    vocab=int(query('/v1/models')['data'][0]['meta']['n_vocab']);assert 0<vocab<=500000
    inputs=R/'QUALITY_INPUTS.json'
    if not inputs.exists():
        cases=[]
        def add(name,pattern):
            files=list((R/'runs').glob(pattern+'/RESULT.json'));assert len(files)==1,(name,len(files))
            j=json.loads(files[0].read_text());cases.append({'name':name,'prompt':j['request']['prompt'],'source':str(files[0].relative_to(R))})
        add('code128','*cache120-r3-code-plain-c128-n128')
        add('japanese128','*cache120-r3-japanese-plain-c128-n96')
        add('count128','*baseline-warm-count-plain-c128-n64')
        add('code64-control','*prefill128-cpu64-code-plain-c64-n64')
        add('count1024-control','*cache120-r3-count-plain-c1024-n96')
        extra=query('/tokenize',{'content':'def merge_sorted_lists(a, b):\n    result = []\n    i = j = 0\n','add_special':False})['tokens']
        cases.append({'name':'code-continuation','prompt':cases[0]['prompt']+extra,'source':'controlled manually authored continuation'})
        inputs.write_text(json.dumps({'vocab':vocab,'cases':cases},indent=2,ensure_ascii=False)+'\n')
    fixed=json.loads(inputs.read_text());assert fixed['vocab']==vocab
    folder=R/'quality'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}';folder.mkdir(parents=True,exist_ok=False)
    record={'label':a.label,'config_sha256':hashlib.sha256(raw).hexdigest(),'inputs_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),
            'vocab':vocab,'dtype':'native float64','byteorder':sys.byteorder,'cases':[],'complete':False,
            'scope':'Full pre-sampling next-token probability distributions on synthetic fixed prefixes; not a general quality benchmark.'}
    def save():
        p=folder/'RESULT.tmp';p.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n');p.replace(folder/'RESULT.json')
    save()
    for case in fixed['cases']:
        helper.watch()
        if CONFIG.read_bytes()!=raw:raise RuntimeError('Configuration changed during comparison')
        request={'prompt':case['prompt'],'n_predict':1,'temperature':0,'seed':42,'ignore_eos':True,
                 'cache_prompt':False,'stream':False,'return_tokens':True,'n_probs':vocab,'post_sampling_probs':False}
        started=time.monotonic();reply=query('/completion',request)
        probs=reply['completion_probabilities'];assert len(probs)==1
        entries=probs[0]['top_logprobs'];assert len(entries)==vocab
        values=array.array('d',[0.0])*vocab;seen=set()
        for e in entries:
            i=e['id'];lp=e['logprob'];assert isinstance(i,int) and 0<=i<vocab and i not in seen
            assert math.isfinite(lp) and lp<=1e-5
            values[i]=math.exp(lp);seen.add(i)
        assert len(seen)==vocab
        # Server accumulates softmax in float across the whole vocabulary.
        # Preserve raw mass; compare_quality normalizes both full distributions.
        total=math.fsum(values);assert abs(total-1)<1e-3,total
        assert reply['timings']['cache_n']==0 and reply['timings']['prompt_n']==len(case['prompt'])
        assert reply['timings']['predicted_n']==1
        file=folder/(case['name']+'.f64.z');file.write_bytes(zlib.compress(values.tobytes()))
        row={'name':case['name'],'prompt_tokens':len(case['prompt']),'elapsed_s':time.monotonic()-started,
             'prompt_ms':reply['timings']['prompt_ms'],'probability_sum':total,'sampled_token':probs[0]['id'],
             'argmax_token':max(range(vocab),key=values.__getitem__),'probability_file':file.name,
             'probability_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        record['cases'].append(row);save();print(json.dumps({'label':a.label,**row}),flush=True)
        helper.watch()
    record['complete']=True;save();print('Full-vocabulary comparison input capture complete:',folder,flush=True)
if __name__=='__main__':main()
