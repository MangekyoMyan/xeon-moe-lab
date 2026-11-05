#!/usr/bin/env python3
"""Aggregate optional CPU routing counts; never infer inference speed from them."""
from pathlib import Path
import argparse
import json
import time

ROOT = Path(__file__).resolve().parent
TYPE_NAMES = {18: 'IQ3_XXS', 21: 'IQ3_S', 22: 'IQ2_S', 39: 'MXFP4'}
BINS = ['0', '1', '2', '3', '4-7', '8-15', '16-31', '32+']

def summarize(rows):
    bins = [sum(r['hist_0_1_2_3_4to7_8to15_16to31_32plus'][i] for r in rows) for i in range(8)]
    experts = sum(r['experts'] for r in rows)
    active = sum(r['active_experts'] for r in rows)
    assigned = sum(r['assigned_rows'] for r in rows)
    eligible = [r for r in rows if r['mx421_plan_eligible']]
    planned = [sum(r['planned_rows_4_2_1'][i] for r in eligible) for i in range(3)]
    planned_total = sum(planned)
    return {'operations': len(rows), 'expert_observations': experts,
            'assigned_rows': assigned, 'active_expert_observations': active,
            'rows_per_expert_including_zero': assigned / experts if experts else None,
            'rows_per_active_expert': assigned / active if active else None,
            'bins': dict(zip(BINS, bins)), 'eligible_mxfp4_operations': len(eligible),
            'planned_rows_4_2_1': planned,
            'planned_percent_4_2_1': [100 * n / planned_total for n in planned] if planned_total else None}

def analyze(capture):
    result = {'capture': str(capture), 'status': 'waiting_for_ready', 'groups': {},
              'scope': 'CPU MUL_MAT_ID only; rows are counted per tensor operation, not unique conversation tokens.',
              'plan_meaning': 'Prospective 4/2/1 routed-row coverage including worker chunk and 16-row tile boundaries; not executed kernel calls, FLOP fractions or predicted tok/s.'}
    ready_path = capture / 'READY.json'
    if not ready_path.exists():return result
    ready = json.loads(ready_path.read_text())
    if not ready.get('ready'):
        result['status'] = 'readiness_failed';return result
    boundary = ready['monotonic_s']
    path = capture / 'histogram.jsonl'
    if not path.exists():
        result['status'] = 'ready_waiting_for_requests';return result
    lines = path.read_text().splitlines(keepends=True)
    rows = []; malformed = 0; partial = 0; before_ready = 0; truncated = False
    for line in lines:
        if not line.endswith('\n'):
            partial += 1;continue
        try:
            row = json.loads(line)
            if row.get('truncated'):
                truncated = True;continue
            if row['monotonic_s'] < boundary:
                before_ready += 1;continue
            bins = row['hist_0_1_2_3_4to7_8to15_16to31_32plus']
            assert len(bins) == 8 and all(isinstance(n, int) and n >= 0 for n in bins)
            assert sum(bins) == row['experts'] and row['active_experts'] == row['experts'] - bins[0]
            plan = row['planned_rows_4_2_1']
            assert len(plan) == 3 and all(isinstance(n, int) and n >= 0 for n in plan)
            if row['mx421_plan_eligible']:assert sum(plan) == row['assigned_rows']
            assert row['batch_tokens'] >= 1
            rows.append(row)
        except (ValueError, KeyError, TypeError, AssertionError):malformed += 1
    groups = {}
    for row in rows:
        # A one-token graph can also be a prefill tail; do not call this decode.
        batch = 'batch=1' if row['batch_tokens'] == 1 else 'batch>=2'
        quant = TYPE_NAMES.get(row['weight_type'], 'type-' + str(row['weight_type']))
        groups.setdefault((batch, quant), []).append(row)
    result.update(status='recorded' if rows else 'ready_waiting_for_requests',
                  truncated=truncated, warmup_operations_excluded=before_ready,
                  malformed_complete_lines=malformed, partial_lines_ignored=partial,
                  total=summarize(rows),
                  groups={f'{b}/{q}': summarize(v) for (b, q), v in sorted(groups.items())},
                  tensors={name: summarize([r for r in rows if r['tensor'] == name])
                           for name in sorted({r['tensor'] for r in rows})})
    return result

def display(result):
    print('記録:', result['capture'], '\n状態:', result['status'])
    if result.get('truncated'):print('注意: 記録上限に到達。先頭部分の分布です。')
    for label, s in result['groups'].items():
        plan = s['planned_percent_4_2_1']
        fmt = lambda x: '-' if x is None else f'{x:.3f}'
        print(f"{label}: {s['operations']}演算 / expert平均 {fmt(s['rows_per_expert_including_zero'])} / 使用expertのみ {fmt(s['rows_per_active_expert'])}")
        print('  expert入力数の分布:', s['bins'])
        if plan:print('  4/2/1入力処理に割り当て可能な行の割合: ' + ' / '.join(f'{p:.1f}%' for p in plan))
    if result.get('malformed_complete_lines'):print('不正な記録行:', result['malformed_complete_lines'])
    print('CPU側のみ。batch=1にはprefill端数も含まれます。割合は推定行数で、実行回数や速度予測ではありません。')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture', nargs='?', type=Path)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--watch', type=float, metavar='SECONDS')
    a = ap.parse_args()
    if a.watch is not None and a.watch < 1:ap.error('--watch must be >= 1 second')
    while True:
        captures = sorted(p for p in (ROOT / 'captures').glob('*') if p.is_dir())
        capture = a.capture or (captures[-1] if captures else None)
        if a.watch:print('\033[2J\033[H', end='')
        if capture is None:print('実モデルの記録はまだありません。ユーザーによるstock-hist起動後に記録されます。')
        else:
            result = analyze(capture)
            if a.json:print(json.dumps(result, ensure_ascii=False, indent=2))
            else:display(result)
        if not a.watch:break
        time.sleep(a.watch)

if __name__ == '__main__':main()
