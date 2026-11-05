#!/usr/bin/env python3
"""User-run bounded observation of existing traffic; never sends inference or changes state."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT.parent/'deepseek41-bringup-20261101/server-config.json'

def prefix_chars(a,b):
    i=0
    for x,y in zip(a,b):
        if x!=y:break
        i+=1
    return i

def compact(slot, previous):
    prompt=slot.get('prompt')
    generated=slot.get('generated')
    if not isinstance(prompt,str):prompt=None
    if not isinstance(generated,str):generated=None
    result={k:slot.get(k) for k in ['id','id_task','is_processing','n_prompt_tokens',
                                   'n_prompt_tokens_processed','n_prompt_tokens_cache']}
    result.update(unix=time.time(),prompt_text_available=prompt is not None,
                  input_characters=len(prompt) if prompt is not None else None,
                  input_fingerprint_sha256=hashlib.sha256(prompt.encode()).hexdigest() if prompt is not None else None,
                  generated_characters=len(generated) if generated is not None else None,
                  stream=slot.get('params',{}).get('stream'))
    if previous and previous['id_task']!=slot.get('id_task') and previous['prompt'] is not None and prompt is not None:
        old=previous['prompt']
        old_generated=previous['generated']
        result['previous']={'id_task':previous['id_task'],'observed_complete':previous['complete'],
            'input_characters':len(old),'generated_characters_at_last_sample':len(old_generated) if old_generated is not None else None,
            'common_input_prefix_characters':prefix_chars(old,prompt),
            'common_input_plus_output_prefix_characters':prefix_chars(old+old_generated,prompt) if old_generated is not None else None}
    # Raw strings live only in this process; never serialize them.
    private={'id_task':slot.get('id_task'),'prompt':prompt,'generated':generated,
             'complete':not slot.get('is_processing',False)}
    return result,private

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seconds',type=int,default=180)
    args=ap.parse_args()
    if not 10<=args.seconds<=600:raise SystemExit('--seconds must be10..600')
    config=json.loads(CONFIG.read_text())
    key=Path(config['api_key_file']).read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    out=ROOT/'observations'/str(time.time_ns());out.mkdir(parents=True)
    result={'purpose':'Read-only current client traffic; no generated requests or model changes',
            'raw_conversation_saved':False,'seconds_requested':args.seconds,'samples':[],'errors':[],
            'limits':['Character-prefix comparison is not exact tokenizer-level n_past.',
                      'The default metrics-only slots response omits prompt text; missing text yields null, never an empty-prefix claim.',
                      'Requests can finish between samples; this is not a complete HTTP request log.',
                      'cache_prompt is not exposed by this version of GET /slots.',
                      'Reading slots may wait for the current processing batch.']}
    old={};last_signature={};deadline=time.monotonic()+args.seconds
    print('再起動・推論テストなしで記録します。この間にOpen WebUIで同じ会話を2往復ほど続けてください。',flush=True)
    print('本文は保存せず、長さ・指紋・キャッシュ数だけを記録します。',flush=True)
    try:
        while time.monotonic()<deadline:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            req=urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",
                                       headers={'Authorization':'Bearer '+key})
            try:
                with opener.open(req,timeout=min(20,max(1,remaining))) as response:slots=json.load(response)
                for slot in slots:
                    if not isinstance(slot,dict) or 'id_task' not in slot:continue
                    sid=slot['id'];row,private=compact(slot,old.get(sid))
                    signature=tuple(row.get(k) for k in ['id_task','is_processing','n_prompt_tokens_processed','n_prompt_tokens_cache'])
                    changed_task=old.get(sid,{}).get('id_task')!=row['id_task']
                    if signature!=last_signature.get(sid):
                        result['samples'].append(row);last_signature[sid]=signature
                        if changed_task:
                            print('task',row['id_task'],'processed',row['n_prompt_tokens_processed'],
                                  'cached',row['n_prompt_tokens_cache'],'stream',row['stream'],flush=True)
                    old[sid]=private
            except urllib.error.HTTPError as e:
                result['errors'].append({'unix':time.time(),'HTTP_status':e.code})
                if e.code in [401,403,404,503]:break
            except (OSError,ValueError) as e:
                result['errors'].append({'unix':time.time(),'type':type(e).__name__})
            (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            time.sleep(min(2,max(0,deadline-time.monotonic())))
    except KeyboardInterrupt:
        result['interrupted']=True
    finally:
        result['finished_unix']=time.time();old.clear()
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('保存先:',out/'result.json',flush=True)

if __name__=='__main__':main()
