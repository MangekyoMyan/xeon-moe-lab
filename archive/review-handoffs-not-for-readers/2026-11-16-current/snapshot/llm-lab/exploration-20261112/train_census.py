"""Aggregate routes over fixed synthetic training prompts; never retain route order."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parent/'deepseek41-bringup-20261101/server-config.json'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helpers = module('helpers', ROOT.parent/'postcompact-20261103/live_probe.py')
reader = module('reader', ROOT/'census/read_census.py')
PROMPTS = [
    ('memory', 'Explain to a systems programmer how CPU cache locality, memory bandwidth, instruction dependencies, and NUMA placement interact when multiplying matrices. Use a concrete toy example with two CPU sockets and a batch of independent vectors. Distinguish measurements from assumptions. Discuss why adding cores sometimes helps and sometimes does not. Write several connected paragraphs with enough detail to guide a small experiment, using only general computer-science reasoning.'),
    ('python', 'Write a small Python implementation of a bounded least-recently-used cache with get and put operations, then explain its invariants. Use a dictionary and an ordered list so the example is readable without imports. Describe the empty-cache case, updating an existing key, eviction when capacity is reached, and capacity zero. Include a short example sequence and explain which key is evicted at each step. This is fictional educational code, not a production deployment.'),
    ('japanese-network', '架空の家庭内ネットワークを説明してください。PCが2台とサーバーが1台あり、全て有線スイッチにつながっています。初心者向けに、IPアドレス、サブネット、デフォルトゲートウェイ、名前解決の役割を順番に説明し、同じネットワーク内の通信とインターネットへの通信を具体例で比較してください。設定変更を指示するのではなく、通信がどの順番で進むかを文章で説明してください。説明は数段落で、用語は初出時に短く定義してください。'),
    ('math', 'Give an elementary proof that the sum of the first n odd positive integers equals n squared. Start with a geometric picture described in words, then give an algebraic induction proof. Explain the base case and the induction step separately. Include checks for n equal to 1, 2, and 5, and finish by explaining why examples alone do not prove the claim for every positive integer. Keep the derivation self-contained and use clear mathematical notation.'),
    ('sql', 'Create a fictional SQL example with customers and orders. Show a small schema, a query for the total completed-order amount per customer, and an explanation of why a left join can preserve customers with no completed orders. Explain the difference between putting the order-status condition in the join and in the where clause. Include null handling and a stable ordering of the result. Do not connect to any real database; all names and records should be invented.'),
    ('schedule', 'Design a simple algorithm for scheduling independent jobs on two identical workers. Each job has a known duration, jobs cannot be split, and workers start together. Explain a greedy longest-job-first method, walk through durations 9, 8, 7, 6, and 4, and discuss why the heuristic is not a proof of global optimality. Include pseudocode, an invariant about assigned work, and how you would test the algorithm on edge cases. Keep the example purely hypothetical.'),
    ('japanese-story', '短い創作を書いてください。舞台は雨の日の小さな図書館です。古い地図を整理している司書と、町の橋について調べに来た学生の会話を中心にしてください。実在の人物や地名は使わず、会話の途中で二人が地図の書き込みに気づく展開にしてください。派手な事件ではなく、観察と対話で小さな謎が解ける話にします。最初に情景を一段落書き、その後に会話と短い動作描写を交互に続けてください。'),
    ('testing', 'Describe a test plan for a fictional text editor undo-and-redo feature. Define the expected behavior when edits are made, several undo actions occur, and a new edit is made before redo. Include examples for insertion, deletion, replacing a selection, empty history, and consecutive edits grouped into one action. Explain which properties can be checked automatically and which user-interface details require separate inspection. Use clear prose and small examples without depending on any particular editor implementation.'),
]


def private_json(path, data):
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    lock = (ROOT/'.benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw = CONFIG.read_bytes()
    assert raw == (ROOT/'profiles/census.json').read_bytes()
    config = json.loads(raw)
    path = Path(config['environment']['XEON_MOE_CENSUS_PATH'])
    key = Path(config['api_key_file']).read_text().strip()
    headers = {'Authorization':'Bearer '+key, 'Content-Type':'application/json'}
    base = f"http://{config['host']}:{config['port']}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def query(route, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        with opener.open(urllib.request.Request(base+route, data=data, headers=headers), timeout=240) as response:
            return json.load(response)

    assert query('/health')['status'] == 'ok'
    assert not any(slot.get('is_processing') for slot in query('/slots'))
    before = reader.read(path)
    assert before['pid'] == helpers.api_pid()
    folder = ROOT/'training'/time.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, mode=0o700)
    private_json(folder/'AGGREGATE_BEFORE.json', before)
    manifest = {'config_sha256':hashlib.sha256(raw).hexdigest(), 'prompts':[], 'complete':False}
    for name, prompt in PROMPTS:
        helpers.watch()
        assert CONFIG.read_bytes() == raw
        assert not any(slot.get('is_processing') for slot in query('/slots'))
        formatted = query('/apply-template', {'messages':[{'role':'user','content':prompt}]})['prompt']
        ids = query('/tokenize', {'content':formatted,'add_special':True,'parse_special':True})['tokens']
        request = {'prompt':ids,'n_predict':256,'temperature':0,'seed':73,'cache_prompt':False,'stream':False,'return_tokens':True}
        result = query('/completion', request)
        if result.get('error'):raise RuntimeError('Training request failed')
        content = result.get('content','')
        timing = result['timings']
        assert timing['cache_n'] == 0 and timing['prompt_n'] == len(ids)
        row = {'name':name,'synthetic_prompt':prompt,'prompt_tokens':len(ids),'timings':timing,
               'stop_type':result.get('stop_type'),'output_sha256':hashlib.sha256(content.encode()).hexdigest()}
        manifest['prompts'].append(row)
        print(json.dumps({'name':name,'prompt_tokens':len(ids),'generated_tokens':timing['predicted_n']},ensure_ascii=False),flush=True)
    assert not any(slot.get('is_processing') for slot in query('/slots'))
    after = reader.read(path)
    assert after['pid'] == before['pid'] == helpers.api_pid()
    previous = {r['name']:r for r in before['records']}
    delta = []
    for record in after['records']:
        old = previous.get(record['name'])
        if old:
            assert all(old[k] == record[k] for k in ['type','experts','k','rows'])
        row = dict(record)
        row['calls'] -= old['calls'] if old else 0
        row['assignments'] -= old['assignments'] if old else 0
        row['counts'] = [v-(old['counts'][i] if old else 0) for i,v in enumerate(record['counts'])]
        assert row['calls'] >= 0 and min(row['counts']) >= 0 and sum(row['counts']) == row['assignments']
        delta.append(row)
    private_json(folder/'AGGREGATE_AFTER.json', after)
    private_json(folder/'TRAINING_COUNTS.json', {'scope':before['scope'],'records':delta})
    manifest['complete'] = True
    manifest['collector_pid'] = before['pid']
    manifest['aggregate_records'] = len(delta)
    manifest['single_token_calls_range'] = [min(r['calls'] for r in delta),max(r['calls'] for r in delta)]
    private_json(folder/'MANIFEST.json', manifest)
    helpers.watch()
    print(json.dumps({'saved':str(folder),'complete':True,'records':len(delta),'calls_range':manifest['single_token_calls_range']}),flush=True)


if __name__ == '__main__':main()
