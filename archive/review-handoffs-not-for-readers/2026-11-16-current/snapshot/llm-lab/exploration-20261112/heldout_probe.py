"""Frozen evaluation prompts not used to select cache entries; synthetic text only."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT.parent/'deepseek41-bringup-20261101/server-config.json'
spec=importlib.util.spec_from_file_location('helpers',ROOT.parent/'postcompact-20261103/live_probe.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
PROMPTS=[
 ('interval-code','Write a complete Python function that merges overlapping closed integer intervals. Each interval is represented by a pair [start, end]. The input may be unsorted, may contain duplicates, and may be empty. Do not modify the input. Include a docstring, the implementation, several normal examples, and tests for empty input, touching intervals, nested intervals, and negative coordinates. Explain the invariant maintained by the loop and the time complexity. Use only built-in Python facilities.'),
 ('csv-code','Explain how to parse a small CSV document whose quoted fields may contain commas, escaped double quotes, and line breaks. Give a Python example using the standard csv module and a StringIO object, invent several sample records, and explain why splitting each line on commas is incorrect. Include a validation step for missing columns and describe how you would report an invalid record without silently changing its data. All records must be fictional.'),
 ('japanese-process','OSの「プロセス」と「スレッド」の違いを、日本語で初心者に説明してください。アドレス空間、共有できるもの、独立しているもの、コンテキスト切り替え、競合状態を順に扱ってください。二つの作業者が同じノートを更新する架空の例を使い、なぜロックが必要になるのかを説明してください。特定のOSで設定変更を行う手順ではなく、仕組みを理解するための数段落の説明にしてください。'),
 ('japanese-water','水が地表と大気の間を循環する仕組みを日本語で説明してください。蒸発、凝結、降水、地表流、地下への浸透を、架空の山と湖のある地域の一年を例にしてつなげてください。雲の中の水がすべて水蒸気であるという誤解にも触れてください。実在の地域の観測値や災害予測は使わず、一般的な物理の説明に限定し、数段落の読みやすい文章にしてください。'),
 ('binary-search-de','Erkläre die binäre Suche auf einer sortierten Liste für eine Person, die gerade programmieren lernt. Beschreibe die Grenzen des Suchbereichs, die Wahl der Mitte und die beiden möglichen Aktualisierungen. Zeige ein Beispiel mit den Zahlen 3, 7, 11, 18, 24, 31 und 42 sowie einen Fall, in dem der gesuchte Wert fehlt. Ergänze Pseudocode und erkläre, warum die Sortierung eine Voraussetzung ist. Antworte auf Deutsch in mehreren klaren Absätzen.'),
 ('moon-phases','Explain why the Moon has phases. Use the geometry of the Sun, Earth, and Moon, distinguish the illuminated half from the part visible to an observer on Earth, and explain why ordinary phases are not caused by Earth casting a shadow. Include a description of new moon, first quarter, full moon, and last quarter, then distinguish a lunar eclipse from these phases. Use a self-contained explanation without relying on a diagram or specific dates.'),
 ('garden-fiction','Write an original fictional scene set in a quiet community garden after a windy night. A retired clockmaker and a young gardener discover that several plant labels have been rearranged. Let them solve the small puzzle through observation and conversation. Use invented names, include sensory details, and keep the tone calm and curious. Begin with a short description of the setting, then alternate dialogue with brief actions. Do not refer to any existing fictional characters or stories.'),
 ('database-keys','Explain candidate keys, primary keys, foreign keys, and unique constraints using an invented library catalog. Include books, physical copies, readers, and loans. Distinguish a book title from an individual physical copy, explain why a title is not a reliable key, and give examples of valid and invalid foreign-key relationships. Discuss what happens when a referenced record is deleted. Keep the explanation conceptual and do not execute any database commands.'),
]


def private_write(path,data,replace=False):
    owner=ROOT.stat()
    temp=path.with_suffix(path.suffix+'.tmp') if replace else path
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC|(os.O_TRUNC if replace else os.O_EXCL)
    fd=os.open(temp,flags,0o600)
    if os.geteuid()==0:os.fchown(fd,owner.st_uid,owner.st_gid)
    with os.fdopen(fd,'w') as stream:json.dump(data,stream,ensure_ascii=False,indent=2);stream.write('\n')
    if replace:temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    assert args.label.replace('-','').isalnum()
    lock=(ROOT/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();config=json.loads(raw)
    key=Path(config['api_key_file']).read_text().strip()
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    url=f"http://{config['host']}:{config['port']}";opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def query(route,body=None):
        with opener.open(urllib.request.Request(url+route,headers=headers,data=None if body is None else json.dumps(body,ensure_ascii=False).encode()),timeout=240) as response:return json.load(response)
    assert query('/health')['status']=='ok'
    assert not any(s.get('is_processing') for s in query('/slots'))
    fixed=ROOT/'HELDOUT_INPUTS.json'
    if not fixed.exists():
        cases=[]
        for name,text in PROMPTS:
            formatted=query('/apply-template',{'messages':[{'role':'user','content':text}]})['prompt']
            tokens=query('/tokenize',{'content':formatted,'add_special':True,'parse_special':True})['tokens']
            assert 32<=len(tokens)<1024
            cases.append({'name':name,'synthetic_prompt':text,'tokens':tokens})
        private_write(fixed,{'scope':'Frozen evaluation inputs excluded from the training census','cases':cases})
    cases=json.loads(fixed.read_text())['cases']
    folder=ROOT/'heldout'/f'{time.strftime("%Y%m%d-%H%M%S")}-{args.label}'
    folder.mkdir(parents=True,mode=0o700)
    if os.geteuid()==0:os.chown(folder,ROOT.stat().st_uid,ROOT.stat().st_gid)
    pid=helpers.api_pid()
    record={'label':args.label,'config_sha256':hashlib.sha256(raw).hexdigest(),'input_sha256':hashlib.sha256(fixed.read_bytes()).hexdigest(),
            'libraries':sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines() if 'libggml' in line or 'libllama' in line}),
            'cases':[],'complete':False,'error':None}
    private_write(folder/'RESULT.json',record)
    try:
        for case in cases:
            initial=helpers.watch();assert CONFIG.read_bytes()==raw
            assert not any(s.get('is_processing') for s in query('/slots'))
            body={'prompt':case['tokens'],'n_predict':192,'temperature':0,'seed':42,'cache_prompt':False,'stream':True,'return_tokens':True}
            start=time.monotonic();first=None;output=[];returned=[];final=None
            with opener.open(urllib.request.Request(url+'/completion',headers=headers,data=json.dumps(body).encode()),timeout=240) as response:
                for line in response:
                    if time.monotonic()-start>240:raise TimeoutError('Evaluation request exceeded bound')
                    line=line.decode().strip()
                    if not line.startswith('data:'):continue
                    payload=line[5:].strip()
                    if payload=='[DONE]':break
                    event=json.loads(payload)
                    if event.get('error'):raise RuntimeError('API error')
                    content=event.get('content','');ids=event.get('tokens') or []
                    output.append(content);returned.extend(ids)
                    if first is None and (content or ids):first=time.monotonic()
                    if event.get('stop'):final=event;break
            assert final is not None
            timing=final['timings'];assert timing['cache_n']==0 and timing['prompt_n']==len(case['tokens'])
            after=helpers.watch();assert after['host_counters']==initial['host_counters']
            row={'name':case['name'],'prompt_tokens':len(case['tokens']),'timings':timing,'stop_type':final.get('stop_type'),
                 'ttft_s':None if first is None else first-start,'elapsed_s':time.monotonic()-start,
                 'output_sha256':hashlib.sha256(''.join(output).encode()).hexdigest(),'token_sha256':hashlib.sha256(json.dumps(returned).encode()).hexdigest()}
            record['cases'].append(row);private_write(folder/'RESULT.json',record,replace=True)
            print(json.dumps({'name':case['name'],'tokens':timing['predicted_n'],'decode_tok_s':timing['predicted_per_second'],'ttft_s':row['ttft_s']}),flush=True)
        record['complete']=True
    except BaseException as error:
        record['error']=repr(error);raise
    finally:private_write(folder/'RESULT.json',record,replace=True)
    print('Saved '+str(folder),flush=True)


if __name__=='__main__':main()
