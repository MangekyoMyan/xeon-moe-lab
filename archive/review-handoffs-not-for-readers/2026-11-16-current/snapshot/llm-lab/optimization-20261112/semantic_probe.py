"""Frozen, objectively scored synthetic tasks spanning changed GPU thresholds."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import time
import urllib.request

R = Path(__file__).resolve().parent
CONFIG = R.parent / 'deepseek41-bringup-20261101/server-config.json'
spec = importlib.util.spec_from_file_location('helpers', R.parent / 'postcompact-20261103/live_probe.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def tasks():
    rows = [{'id': i, 'qty': (i * 7) % 13 + 1, 'price': (i * 11) % 19 + 2} for i in range(1, 37)]
    items = ['brass screw', 'paper clip', 'steel washer', 'rubber seal', 'wooden peg', 'glass bead']
    table = '\n'.join(f"P{x['id']:02d} | {items[(x['id'] - 1) % len(items)]} | {x['qty']} | {x['price']}" for x in rows)
    return [
        dict(name='merge_sorted', kind='code', lo=128, hi=512, limit=384, text='''Write exactly one Python function named merge_sorted(a, b). Both arguments are lists of integers already sorted in nondecreasing order. Return a new sorted list containing every element of both lists, preserving all duplicates. Do not modify either input. Handle empty lists, negative integers, and unequal list lengths. Use a linear two-pointer merge, not sorting the concatenated inputs. The running time must be O(len(a) + len(b)). Do not import modules or use recursion. Use only ordinary list operations, loops, comparisons, indexing, and len. Return only the complete function definition, with no example, markdown, explanation, annotations, or surrounding test program. Ensure every branch returns the full merged list, including any remaining elements after one input is exhausted.'''),
        dict(name='run_lengths', kind='code', lo=128, hi=512, limit=384, text='''Write exactly one Python function named run_lengths(values). Its argument is a list of integers. Return a list of two-element lists [value, count], describing each consecutive run of equal values in the original order. For example, input [2, 2, -1, -1, 2] must produce [[2, 2], [-1, 2], [2, 1]]. Separate runs of the same value must remain separate. An empty input must return an empty list. Do not modify the input, sort the values, or count frequencies across the whole list. Do not import modules or use recursion. Use a single traversal with ordinary list operations. Return only the complete Python function definition, with no markdown, explanation, annotations, sample invocation, or surrounding tests. Remember to include the last run in the result.'''),
        dict(name='japanese-extract', kind='json', lo=128, hi=512, limit=192,
             expected={'accepted': ['A17', 'C04'], 'total': 21}, text='''次の架空の受付記録から、状態が「確定」で、個数が10個以上の記録だけを選んでください。
A17：確定、12個。B09：保留、30個。C04：確定、9個。D11：取消、18個。
訂正：C04の個数は9個ではなく11個です。さらにA17の個数は12個ではなく10個に訂正されました。訂正後の個数だけを使い、訂正前の数は足しません。
選ばれた記録のIDを元の記載順でacceptedに並べ、個数の合計をtotalに入れてください。条件に合わない記録は合計に含めません。
回答はacceptedとtotalだけを持つJSONオブジェクトにしてください。説明、単位、マークダウンは不要です。'''),
        dict(name='short-selection', kind='json', lo=64, hi=128, limit=96,
             expected=['B', 'D'], text='''Select only records whose status is ready and quantity is strictly greater than 4. Records: A, ready, 4; B, ready, 7; C, held, 12; D, ready, 5. Keep their original order. Return only a JSON array containing their letter identifiers, with no markdown or explanation. Do not include held records or quantities equal to four.'''),
        dict(name='long-invoice', kind='json', lo=512, hi=1024, limit=512,
             expected=[[f"P{x['id']:02d}", x['qty'] * x['price']] for x in rows],
             text='''Calculate each line total in this fictional invoice. The columns are ID, item description, quantity, and unit price in whole credits. For each row multiply quantity by unit price. Keep every row in the listed order. Return only a JSON array of two-element arrays [ID, total], with the ID as a string and total as an integer. There are no taxes, discounts, rounding rules, or shipping charges. Do not combine rows or add a grand total. Include every ID exactly once, even when totals are equal. Item descriptions do not affect the calculation. No markdown or explanation.
ID | description | quantity | unit_price
''' + table),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert args.label.replace('-', '').replace('_', '').isalnum()
    lock = (R / '.benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    key = Path(config['api_key_file']).read_text().strip()
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    base = f"http://{config['host']}:{config['port']}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def query(path, body=None):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        with opener.open(urllib.request.Request(base + path, headers=headers, data=data), timeout=240) as response:
            return json.load(response)

    assert query('/health').get('status') == 'ok'
    if any(slot.get('is_processing') for slot in query('/slots')):
        raise RuntimeError('API busy')
    inputs = R / 'SEMANTIC_INPUTS.json'
    if not inputs.exists():
        cases = tasks()
        for case in cases:
            formatted = query('/apply-template', {'messages': [{'role': 'user', 'content': case['text']}]})['prompt']
            case['tokens'] = query('/tokenize', {'content': formatted, 'add_special': True, 'parse_special': True})['tokens']
            assert case['lo'] <= len(case['tokens']) < case['hi'], (case['name'], len(case['tokens']))
        inputs.write_text(json.dumps({'cases': cases}, ensure_ascii=False, indent=2) + '\n')
    fixed = json.loads(inputs.read_text())
    if args.prepare_only:
        print(json.dumps({c['name']: len(c['tokens']) for c in fixed['cases']}))
        return
    folder = R / 'semantic' / f'{time.strftime("%Y%m%d-%H%M%S")}-{args.label}'
    folder.mkdir(parents=True)
    record = {'label': args.label, 'config_sha256': hashlib.sha256(raw).hexdigest(),
              'inputs_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
              'cases': [], 'complete': False, 'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scorer_sha256': hashlib.sha256((R / 'score_generated_function.py').read_bytes()).hexdigest()}

    def save():
        temp = folder / 'RESULT.tmp'
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        temp.replace(folder / 'RESULT.json')

    save()
    for case in fixed['cases']:
        helpers.watch()
        assert CONFIG.read_bytes() == raw
        body = {'prompt': case['tokens'], 'n_predict': case['limit'], 'temperature': 0,
                'seed': 42, 'cache_prompt': False, 'stream': True, 'return_tokens': True}
        start = time.monotonic()
        first = None
        pieces = []
        final = None
        with opener.open(urllib.request.Request(base + '/completion', data=json.dumps(body).encode(), headers=headers), timeout=240) as response:
            for line in response:
                if time.monotonic() - start > 240:
                    raise TimeoutError('Bounded semantic request exceeded 240 seconds')
                line = line.decode().strip()
                if not line.startswith('data:'):
                    continue
                payload = line[5:].strip()
                if payload == '[DONE]':
                    break
                event = json.loads(payload)
                if event.get('error'):
                    raise RuntimeError('API error')
                content = event.get('content', '')
                pieces.append(content)
                if first is None and (content or event.get('tokens')):
                    first = time.monotonic()
                if event.get('stop'):
                    final = event
                    break
        assert final is not None
        content = ''.join(pieces).strip()
        if case['kind'] == 'code':
            options = dict(user=65534, group=65534, extra_groups=[]) if os.geteuid() == 0 else {}
            completed = subprocess.run(['/usr/bin/python3', '-I', '-c', (R / 'score_generated_function.py').read_text()],
                                       input=json.dumps({'name': case['name'], 'content': content}), text=True,
                                       capture_output=True, timeout=5, cwd='/tmp', env={'PATH': '/usr/bin:/bin'}, **options)
            scored = json.loads(completed.stdout) if completed.returncode == 0 else {
                'correct': False, 'reason': 'Scoring process failed',
                'returncode': completed.returncode, 'stderr': completed.stderr[:2000]}
        else:
            try:
                scored = {'correct': json.loads(content) == case['expected']}
            except json.JSONDecodeError:
                scored = {'correct': False, 'reason': 'Not strict JSON'}
        row = {'name': case['name'], 'prompt_tokens': len(case['tokens']), 'content': content,
               'score': scored, 'stop_type': final.get('stop_type'),
               'correct': scored['correct'] and final.get('stop_type') == 'eos',
               'ttft_s': None if first is None else first - start,
               'elapsed_s': time.monotonic() - start, 'timings': final['timings'],
               'output_sha256': hashlib.sha256(content.encode()).hexdigest()}
        assert row['timings']['cache_n'] == 0 and row['timings']['prompt_n'] == len(case['tokens'])
        record['cases'].append(row)
        save()
        helpers.watch()
        print(json.dumps({k: v for k, v in row.items() if k != 'content'}), flush=True)
    record['complete'] = True
    record['all_correct'] = all(c['correct'] for c in record['cases'])
    save()
    print('Saved ' + str(folder), flush=True)


if __name__ == '__main__':
    main()
