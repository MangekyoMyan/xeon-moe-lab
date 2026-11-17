"""Verify all generated IDs and journals, then render readable copies losslessly when valid."""
from pathlib import Path
import hashlib,json,sys
from verify_evidence import R,load_run,common,read,write,sha

def main():
    names=['answer-0-F','answer-1-P','answer-2-F','answer-3-P'];runs=[load_run(n) for n in names];common(runs)
    plan=read(R/'caller/FROZEN_PLAN.json');cases={c['case_id']:c['execution_input'] for c in plan['cases']}
    expected=[('text_plan_constraints_ja','F'),('text_plan_constraints_ja','P'),('text_plan_constraints_en','P'),('text_plan_constraints_en','F'),('code_bounded_queue_python','F'),('code_bounded_queue_python','P')]
    all_answers=[];prev_finish=0
    for p,j,m,launch in runs:
        contract=j['generation_contract'];assert contract==dict(policy='greedy',temperature=0,seed=42,seed_used=False,ignore_eos=False,max_generated_ids=1024,reference_supplied=False,all_generated_ids_including_control_counted=True)
        assert j['answer_function_providers'] and all(x['path']==j['providers'][0]['path'] for x in j['answer_function_providers']) # llama_decode is provider0
        eogs=set(j['model_eog_ids']);assert eogs
        for a in j['answers']:
            ci=a['case_id'];mode=a['mode'];assert a['prefix_ids']==cases[ci]['prefix_ids'] and not a['reference_supplied']
            ids=a['generated_ids'];assert len(ids)==a['generated_count'] and 1<=len(ids)<=1024 and all(type(t)==int and 0<=t<129280 for t in ids)
            assert a['decoded_T1_calls']==len(ids)-1 and not (set(ids[:-1])&eogs)
            if a['stop_reason']=='eog':assert ids[-1] in eogs and not a['truncated'] and a['terminal_token_id']==ids[-1]
            else:assert a['stop_reason']=='length' and len(ids)==1024 and ids[-1] not in eogs and a['truncated'] and a['terminal_token_id'] is None
            for k in ['all_text_file','visible_text_file','events_file']:assert a[k] in m and Path(a[k]).name==a[k]
            events=[json.loads(line) for line in (p/a['events_file']).read_text().splitlines()];samples=[x for x in events if x['kind']=='sample'];decodes=[x for x in events if x['kind']=='decode']
            assert len(samples)==len(ids) and len(decodes)==len(ids)-1
            assert [(x['kind'],x['ordinal']) for x in events]==[(kind,i) for i in range(len(ids)) for kind in (['sample','decode'] if i<len(ids)-1 else ['sample'])]
            for i,x in enumerate(samples):
                assert x['ordinal']==i and x['token_id']==ids[i] and x['is_eog']==(ids[i] in eogs)
                prior=a['prefill_ready_ns'] if i==0 else decodes[i-1]['decode_ready_ns']
                assert x['logits_ready_ns']==prior and x['sample_end_ns']>=x['sample_begin_ns']>=prior
            for i,x in enumerate(decodes):assert x['status']==0 and x['input_token_id']==ids[i] and x['position']==len(a['prefix_ids'])+i and x['decode_ready_ns']>x['decode_begin_ns']>=samples[i]['sample_end_ns']
            allbytes=b''.join(bytes.fromhex(x['piece_all_hex']) for x in samples);visible=b''.join(bytes.fromhex(x['piece_visible_hex']) for x in samples)
            assert allbytes==(p/a['all_text_file']).read_bytes() and visible==(p/a['visible_text_file']).read_bytes()
            assert a['prefill_begin_ns']>prev_finish and a['finish_ns']>=samples[-1]['sample_end_ns'];prev_finish=a['finish_ns']
            out=R/'answers'/ci;out.mkdir(parents=True,exist_ok=True);rendered=out/(mode+'.txt');assert not rendered.exists()
            try:text=visible.decode('utf-8');valid=True
            except UnicodeDecodeError:text=visible.decode('utf-8',errors='replace');valid=False
            rendered.write_text(text);rawpath=out/(mode+'-raw.utf8');rawpath.write_bytes(visible)
            all_answers.append(dict(case_id=ci,mode=mode,run=str(p),generated_count=len(ids),decoded_T1_calls=len(decodes),stop_reason=a['stop_reason'],truncated=a['truncated'],
                UTF8_valid=valid,rendered_file=str(rendered.relative_to(R)),raw_file=str(rawpath.relative_to(R)),raw_sha256=hashlib.sha256(visible).hexdigest(),generated_ids=ids,
                native_prefill_ms=(a['prefill_ready_ns']-a['prefill_begin_ns'])/1e6,first_sample_ms=(samples[0]['sample_end_ns']-a['prefill_begin_ns'])/1e6,
                elapsed_from_prefill_ms=(a['finish_ns']-a['prefill_begin_ns'])/1e6,decode_ms=[(x['decode_ready_ns']-x['decode_begin_ns'])/1e6 for x in decodes],
                source_json=str((p/('answer-'+ci+'-'+mode+'.json'))),journal_sha256=sha(p/a['events_file'])))
    assert [(a['case_id'],a['mode']) for a in all_answers]==expected
    equality={ci:next(x['generated_ids'] for x in all_answers if x['case_id']==ci and x['mode']=='F')==next(x['generated_ids'] for x in all_answers if x['case_id']==ci and x['mode']=='P') for ci in cases}
    result=dict(passed=True,plan_sha256=sha(R/'caller/FROZEN_PLAN.json'),responses=all_answers,generated_IDs_F_P_equal=equality,
        scope='Only completeness/provenance/journal checks here. Parent grades actual task answers separately. Visible text is token_to_piece(special=false), not WebUI rendering; free-answer times are descriptive and unbalanced.')
    write(R/'ANSWER_VERIFICATION.json',result);print(json.dumps([dict(case_id=a['case_id'],mode=a['mode'],generated_count=a['generated_count'],stop=a['stop_reason'],UTF8=a['UTF8_valid']) for a in all_answers]))
if __name__=='__main__':main()
