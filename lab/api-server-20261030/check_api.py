#!/usr/bin/python3
"""Exercise authenticated OpenAI chat and SSE on the private API."""
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = 'http://10.10.10.2:8000'
MODEL = 'glm-4.5-air-q8'
KEY = (ROOT / 'api-key.txt').read_text().strip()
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, payload=None, key=KEY, timeout=900):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    return HTTP.open(req, timeout=timeout)


def read_json(path, payload=None, key=KEY):
    with request(path, payload, key) as response:
        return json.load(response)


def main():
    results = {'started_unix': time.time(), 'base_url': BASE + '/v1', 'model': MODEL}
    results['health'] = read_json('/health', key=None)
    results['models'] = read_json('/v1/models')
    assert any(x['id'] == MODEL for x in results['models']['data'])
    results['props'] = read_json('/props')
    assert results['props']['default_generation_settings']['n_ctx'] == 131072
    assert results['props']['total_slots'] == 1
    payload = dict(model=MODEL, messages=[{'role': 'user', 'content': '2+3はいくつですか？数字だけ答えてください。'}],
                   max_tokens=32, temperature=0, seed=42)
    for label, key in [('missing_key', None), ('wrong_key', 'invalid-test-key')]:
        try:
            read_json('/v1/chat/completions', payload, key)
            raise AssertionError('Unauthenticated inference was accepted')
        except urllib.error.HTTPError as error:
            assert error.code in (401, 403), error.code
            results[label] = error.code
    start = time.monotonic()
    results['chat'] = read_json('/v1/chat/completions', payload)
    results['chat_wall_seconds'] = time.monotonic() - start
    assert '5' in results['chat']['choices'][0]['message']['content']
    print('Authenticated chat passed', flush=True)
    streaming = dict(model=MODEL, messages=[{'role': 'user', 'content': '日本語で、CPUとGPUの得意な処理の違いを3文で説明してください。'}],
                     max_tokens=160, temperature=0, seed=42, stream=True, stream_options={'include_usage': True})
    chunks = []
    text = ''
    done = False
    first_content_seconds = None
    start = time.monotonic()
    with request('/v1/chat/completions', streaming) as response:
        results['stream_content_type'] = response.headers.get('Content-Type')
        assert 'text/event-stream' in results['stream_content_type']
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            item = json.loads(data)
            chunks.append(item)
            for choice in item.get('choices', []):
                content = choice.get('delta', {}).get('content') or ''
                if content and first_content_seconds is None:
                    first_content_seconds = time.monotonic() - start
                text += content
    assert done and text.strip(), 'Incomplete SSE response'
    results['stream'] = dict(chunks=chunks, text=text, first_content_seconds=first_content_seconds,
                             total_seconds=time.monotonic() - start, done=done)
    print('Streaming passed', flush=True)
    multi = dict(model=MODEL, messages=[
        {'role': 'system', 'content': '日本語で簡潔に答えてください。'},
        {'role': 'user', 'content': '合言葉はFROST-271です。覚えてください。'},
        {'role': 'assistant', 'content': '合言葉を覚えました。'},
        {'role': 'user', 'content': '先ほどの合言葉だけを出力してください。'}], max_tokens=32, temperature=0)
    results['multi_turn'] = read_json('/v1/chat/completions', multi)
    assert 'FROST-271' in results['multi_turn']['choices'][0]['message']['content']
    results['repeat_chat'] = read_json('/v1/chat/completions', payload)
    assert results['repeat_chat']['choices'][0]['message']['content'] == results['chat']['choices'][0]['message']['content']
    results['passed'] = True
    results['finished_unix'] = time.time()
    (ROOT / 'api-check-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': True, 'chat_usage': results['chat'].get('usage'),
                      'stream_first_content_seconds': first_content_seconds,
                      'stream_total_seconds': results['stream']['total_seconds'],
                      'stream_text': text, 'multi_turn': results['multi_turn']['choices'][0]['message']['content']},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
