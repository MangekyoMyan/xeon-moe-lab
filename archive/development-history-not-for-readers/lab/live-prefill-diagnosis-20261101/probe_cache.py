#!/usr/bin/env python3
"""Small synthetic requests to an already loaded server; never starts a model."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parent / 'deepseek41-bringup-20261101/server-config.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['chat', 'native', 'flow', 'stream', 'long'])
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    key = Path(config['api_key_file']).read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None):
        req = urllib.request.Request(
            f"http://{config['host']}:{config['port']}" + path,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        started = time.monotonic()
        with opener.open(req, timeout=120) as response:
            if not (body and body.get('stream')):
                return json.load(response)
            content = ''
            first_content = None
            usage = timings = finish = None
            done = False
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    done = True
                    break
                event = json.loads(payload)
                if 'error' in event:
                    raise RuntimeError('Server returned a streaming error')
                usage = event.get('usage') or usage
                timings = event.get('timings') or timings
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {}).get('content') or ''
                    if delta and first_content is None:
                        first_content = time.monotonic() - started
                    content += delta
                    finish = choice.get('finish_reason') or finish
            if not done or not finish:
                raise RuntimeError('Incomplete streaming response; do not use as conversation history')
            return {'choices': [{'message': {'role': 'assistant', 'content': content},
                                 'finish_reason': finish}], 'usage': usage, 'timings': timings,
                    'measured_first_content_s': first_content}

    if request('/health').get('status') != 'ok':
        raise SystemExit('Server is not ready; no inference sent.')
    if any(s.get('is_processing') for s in request('/slots')):
        raise SystemExit('Existing request is active; no inference sent.')
    out = ROOT / 'probes' / (time.strftime('%Y%m%d-%H%M%S') + '-' + args.mode)
    out.mkdir(parents=True, exist_ok=False)
    rows = []

    def query(name, body, path):
        # An idle check is a precaution, not an exclusive reservation of the slot.
        if any(s.get('is_processing') for s in request('/slots')):
            raise SystemExit('Another client is active; stopping the probe.')
        start = time.monotonic()
        result = request(path, body)
        row = {'name': name, 'elapsed_s': time.monotonic() - start,
               'synthetic_request': body, 'response': result}
        rows.append(row)
        (out / 'result.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
        content = result.get('content')
        if 'choices' in result:
            content = result['choices'][0]['message'].get('content')
        print(json.dumps({'name': name, 'elapsed_s': row['elapsed_s'], 'content': content,
                          'timings': result.get('timings'), 'usage': result.get('usage'),
                          'first_content_s': result.get('measured_first_content_s')},
                         ensure_ascii=False), flush=True)
        return result

    if args.mode in ['chat', 'flow', 'stream', 'long']:
        history = [{'role': 'system', 'content': 'Answer with the requested number only.'},
                   {'role': 'user', 'content': 'Alpha is 7391. Beta is 8426. What is Alpha?'}]
        if args.mode == 'long':
            records = '\n'.join(f'Entry {i}: rain is wet; these notes do not change the codes.' for i in range(1, 31))
            history[1]['content'] = 'Alpha is 7391. Beta is 8426.\n' + records + '\nWhat is Alpha?'

        def chat(name, messages, cache):
            return query(name, {'model': config['model_alias'], 'messages': messages,
                               'temperature': 0, 'seed': 42,
                               'max_tokens': 48 if args.mode in ['flow', 'stream'] else 8,
                               'cache_prompt': cache, 'stream': args.mode == 'stream',
                               'stream_options': {'include_usage': True}, 'return_tokens': True,
                               'chat_template_kwargs': {'thinking_mode': 'chat'}},
                         '/v1/chat/completions')

        if args.mode in ['flow', 'stream']:
            history = [{'role': 'system', 'content': '日本語で簡潔に、一文で答えてください。'},
                       {'role': 'user', 'content': '雨の日に家でできる趣味を一つ教えて。'}]
            initial = chat('japanese-initial', history, False)
            continued = history + [initial['choices'][0]['message'],
                                   {'role': 'user', 'content': 'それを始めるには何を準備しますか？'}]
            answer = chat('japanese-continuation', continued, True)
            third = continued + [answer['choices'][0]['message'],
                                 {'role': 'user', 'content': '最初の一歩を教えてください。'}]
            if args.mode == 'flow':
                chat('independent-auxiliary', [{'role': 'user', 'content': '会話「雨の日の趣味」の短いタイトルだけを答えて。'}], True)
            chat('japanese-after-auxiliary' if args.mode == 'flow' else 'japanese-third-continuation', third, True)
            chat('japanese-reference', third, False)
        else:
            initial = chat('initial', history, False)
            continued = history + [initial['choices'][0]['message'],
                                   {'role': 'user', 'content': 'What is Beta?'}]
            chat('continuation-cache', continued, True)
            chat('continuation-reference', continued, False)
            if args.mode != 'long':
                chat('repeat-cache', continued, True)
    else:
        tokens = request('/tokenize', {'content': 'Numbers: 10, 11, 12, 13,',
                                       'add_special': True})['tokens']

        def native(name, prompt, cache):
            return query(name, {'prompt': prompt, 'n_predict': 4, 'temperature': 0,
                               'seed': 42, 'cache_prompt': cache, 'return_tokens': True,
                               'stream': False}, '/completion')

        initial = native('native-initial', tokens, False)
        if not initial.get('tokens'):
            raise SystemExit('No returned token IDs; stopping without guessing token boundaries.')
        continued = tokens + initial['tokens']
        native('native-exact-token-append', continued, True)
        native('native-append-reference', continued, False)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
