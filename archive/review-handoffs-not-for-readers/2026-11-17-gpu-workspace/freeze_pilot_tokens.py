"""Freeze supplied 20-case corpus with the restored API's template/tokenizer only.

No completion/inference requests, no route collection, no bank fitting. Corpus
splits stay as supplied. Credentials are read locally and never saved or printed.
"""
from pathlib import Path
import array,hashlib,json,time,urllib.request
R=Path(__file__).resolve().parent;E=R.parent/'deepseek41-bringup-20261101'
P=R/'incoming/xeon_moe_route_calibration_review_2026-11-17/pilot'
def digest(b):return hashlib.sha256(b).hexdigest()
def ids_hash(ids):return digest(array.array('i',ids).tobytes())
def main():
    out=R/'PILOT_TOKENS.json';assert not out.exists()
    restored=json.loads((R/'RESTORED_RUNTIME.json').read_text());assert restored['API_health']=='ok'
    cfg=json.loads((E/'server-config.json').read_text());assert digest((E/'server-config.json').read_bytes())==restored['config_sha256']
    key=Path(cfg['api_key_file']).read_text().strip();base='http://'+cfg['host']+':'+str(cfg['port'])
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,body=None):
        req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with opener.open(req,timeout=20) as f:return json.load(f)
    corpus=json.loads((P/'PILOT_CASES_JA_EN_CODE.json').read_text());stage=json.loads((P/'PILOT_STAGE_MANIFEST.json').read_text())
    assert digest((P/'PILOT_CASES_JA_EN_CODE.json').read_bytes())==stage['corpus_sha256']
    props=request('/props');template=props['chat_template'];assert isinstance(template,str)
    data=dict(schema=1,passed=False,unix=time.time(),corpus_sha256=stage['corpus_sha256'],API_config_sha256=restored['config_sha256'],chat_template_sha256=digest(template.encode()),chat_template_caps=props.get('chat_template_caps'),template_kwargs=dict(thinking_mode='chat'),no_system_message=True,request_tools=None,add_generation_prompt=True,tokenizer_flags=dict(add_special=True,parse_special=True),endpoints=['/props','/apply-template','/tokenize'],model_inference_requested=False,route_collected=False,bank_fitted=False,split_policy='Original family splits retained; no outcome-based adjustment',cases=[])
    try:
        for family in corpus['families']:
            for case in family['variants']:
                prompt,reference=case['prompt_utf8'],case['reference_utf8']
                assert digest(prompt.encode())==case['prompt_sha256'] and digest(reference.encode())==case['reference_sha256']
                body=dict(messages=[dict(role='user',content=prompt)])
                rendered=request('/apply-template',body)['prompt']
                assert rendered.endswith('<｜Assistant｜></think>'),'Unexpected assistant channel boundary; investigate'
                prefix=request('/tokenize',dict(content=rendered,add_special=True,parse_special=True,with_pieces=False))['tokens']
                joined=rendered+reference
                pieces=request('/tokenize',dict(content=joined,add_special=True,parse_special=True,with_pieces=True))['tokens']
                full=[x['id'] for x in pieces]
                assert full[:len(prefix)]==prefix,'Prefix token IDs changed on concatenation'
                cont=full[len(prefix):];assert len(cont)>=65,'Insufficient natural reference; no padding'
                assert all(type(x)is int and 0<=x<129280 for x in full)
                raw=b''.join(x['piece'].encode() if isinstance(x['piece'],str) else bytes(x['piece']) for x in pieces[len(prefix):len(prefix)+64])
                row=dict(case_id=case['case_id'],family_id=family['family_id'],split=family['split'],language_or_content=case['language_or_content'],prompt_sha256=case['prompt_sha256'],reference_sha256=case['reference_sha256'],rendered_prefix=rendered,rendered_prefix_sha256=digest(rendered.encode()),joined_text_sha256=digest(joined.encode()),prefix_ids=prefix,prefix_count=len(prefix),prefix_ids_i32_sha256=ids_hash(prefix),full_reference_continuation_ids=cont,reference_count=len(cont),continuation_first65_ids=cont[:65],used_T1_ids=cont[:64],used_T1_ids_i32_sha256=ids_hash(cont[:64]),first_T1_position=len(prefix),next_targets_prefix_and64T1=cont[:65],decoded_first64=raw.decode('utf-8',errors='replace'),decoded_first64_bytes_hex=raw.hex(),strict_prefix_verified=True,assistant_boundary_suffix='<｜Assistant｜></think>')
                data['cases'].append(row)
                out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        assert len(data['cases'])==20 and len({x['family_id'] for x in data['cases']})==12
        assert {s:sum(x['split']==s for x in data['cases']) for s in ['train','validation','holdout']}==dict(train=10,validation=5,holdout=5)
        assert {x['case_id'] for x in data['cases']}=={c for s in stage['stages'] for c in s['cases']}
        data['passed']=True;out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(passed=True,cases=20,prefix_token_range=[min(x['prefix_count'] for x in data['cases']),max(x['prefix_count'] for x in data['cases'])],reference_token_range=[min(x['reference_count'] for x in data['cases']),max(x['reference_count'] for x in data['cases'])],model_inference_requested=False,route_collected=False)))
    except Exception as error:
        data['error']=str(error);out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');raise
if __name__=='__main__':main()
