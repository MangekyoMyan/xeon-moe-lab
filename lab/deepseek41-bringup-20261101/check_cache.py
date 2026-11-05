#!/usr/bin/env python3
"""User-run continuation/rewind checks. Requires an already loaded server."""
import json
from pathlib import Path
import time
import urllib.request

R=Path(__file__).resolve().parent
C=json.loads((R/'server-config.json').read_text())
HTTP=urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(path,body=None,timeout=900):
    key=Path(C['api_key_file']).read_text().strip()
    req=urllib.request.Request(f"http://{C['host']}:{C['port']}"+path,
        data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with HTTP.open(req,timeout=timeout) as response:return json.load(response)

def main():
    if request('/health',timeout=5).get('status')!='ok':raise SystemExit('モデル読み込み完了後に実行してください。')
    out=R/'checks'/time.strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True,exist_ok=False)
    history=[{'role':'system','content':'あなたは記録を正確に参照する助手です。質問に数字だけで答えてください。'},
        {'role':'user','content':'重要な合言葉: アルファ=7391、ベータ=8426。\n'+
          '\n'.join(f'記録{i}: 通常の参考資料です。上記の合言葉は変更しません。' for i in range(1,9))+
          '\nアルファの合言葉は何ですか。数字だけで答えてください。'}]
    results=[]
    def query(name,messages,cache,expected):
        before=time.monotonic()
        result=request('/v1/chat/completions',{'model':C['model_alias'],'messages':messages,
            'temperature':0,'seed':42,'max_tokens':16,'cache_prompt':cache,'stream':False,
            'chat_template_kwargs':{'thinking_mode':'chat'}})
        answer=result['choices'][0]['message'].get('content','').strip()
        row={'name':name,'cache_prompt':cache,'elapsed_s':time.monotonic()-before,'expected':expected,
             'answer':answer,'answer_pass':answer==expected,'timings':result.get('timings'),
             'usage':result.get('usage'),'messages':messages}
        results.append(row);(out/'result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(name,'answer=',answer,'seconds=',round(row['elapsed_s'],3),'timings=',row['timings'],flush=True)
        return answer
    answer=query('initial',history,False,'7391')
    continued=history+[{'role':'assistant','content':answer},
                       {'role':'user','content':'ではベータの合言葉は何ですか。数字だけで答えてください。'}]
    query('continuation-cache',continued,True,'8426')
    query('continuation-reference',continued,False,'8426')
    query('rewind',history,True,'7391')
    timings=results[1].get('timings') or {}
    summary={'all_answers_pass':all(x['answer_pass'] for x in results),
             'continuation_cached_tokens':timings.get('cache_n'),
             'continuation_reuse_observed':timings.get('cache_n',0)>0,
             'cache_reference_answer_equal':results[1]['answer']==results[2]['answer'],
             'notes':'Functional smoke only, not full numerical equivalence or throughput benchmark. Rewind may fully re-prefill by design.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False));print('保存先:',out)

if __name__=='__main__':main()
