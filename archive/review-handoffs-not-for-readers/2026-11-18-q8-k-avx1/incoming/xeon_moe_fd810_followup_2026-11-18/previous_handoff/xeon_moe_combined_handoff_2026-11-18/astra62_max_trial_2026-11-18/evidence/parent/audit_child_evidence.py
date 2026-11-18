#!/usr/bin/env python3
"""Parent-side audit of candidate artifacts; no benchmark execution."""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import statistics


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def timings(path):
    rows = list(csv.DictReader(path.open()))
    conditions = []
    for case in sorted({r['case'] for r in rows}):
        cases = [r for r in rows if r['case'] == case]
        pairs = []
        for rnd in sorted({int(r['round']) for r in cases}):
            pair = [r for r in cases if int(r['round']) == rnd]
            variants = sorted({r['variant'] for r in pair})
            assert len(variants) == 2 and 'baseline' in variants
            candidate = next(v for v in variants if v != 'baseline')
            stats = {}
            assert len({r['seed'] for r in pair}) == 1
            samples = {r['sample'] for r in pair}
            for sample in samples:
                assert len({r['output_hash'] for r in pair if r['sample'] == sample}) == 1
            for variant in variants:
                vr = [r for r in pair if r['variant'] == variant]
                assert len(vr) == len(samples) and len({r['sample'] for r in vr}) == len(vr)
                assert len({r['work_bytes'] for r in vr}) == 1
                assert len({r['order'] for r in vr}) == 1
                stats[variant] = {
                    'median_ns': statistics.median(float(r['ns_per_op']) for r in vr),
                    'order': int(vr[0]['order']), 'work_bytes': int(vr[0]['work_bytes']),
                    'samples': len(vr),
                }
            ratio = stats[candidate]['median_ns'] / stats['baseline']['median_ns']
            pairs.append({'round': rnd, 'seed': int(pair[0]['seed']), 'ratio': ratio, 'statistics': stats})
        ratios = [p['ratio'] for p in pairs]
        med = statistics.median(ratios)
        conditions.append({
            'case': case, 'candidate': candidate, 'pairs': len(pairs),
            'median_ratio': med, 'time_reduction_percent': 100 * (1 - med),
            'candidate_faster_pairs': sum(x < 1 for x in ratios),
            'ratio_min': min(ratios), 'ratio_max': max(ratios), 'rounds': pairs,
        })
    return {
        'input': str(path), 'sha256': sha(path), 'raw_rows': len(rows),
        'method_ja': '各プロセスのサンプル中央値を求め、同じround/seedの候補÷基準を算出し、そのペア中央値を報告。プロセス内サンプルを独立試行数に数えない。',
        'all_paired_output_hashes_equal': True, 'conditions': conditions,
    }


def dumps(work):
    groups = collections.defaultdict(list)
    for p in sorted(work.glob('check-*.bin')) + sorted(work.glob('edges-*.bin')):
        group = 'check'
        if p.name.startswith('edges-'):
            group = 'edges'
            if p.name.startswith('edges-disabled-'):
                group = 'edges-disabled'
            elif p.name.startswith('edges-forced-'):
                group = 'edges-forced'
        groups[group].append({'file': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    for group, files in groups.items():
        assert len(files) >= 2
        assert len({(f['bytes'], f['sha256']) for f in files}) == 1, group
    return {
        'method_ja': '同じテスト集合・同じ環境変数設定内で全出力ダンプの長さとSHA256を比較。子の演算を新たに実行したわけではない。TILED_MM disabledとdefaultは別の算術経路のため混ぜて比較しない。',
        'all_equal_within_suite': True, 'groups': dict(groups),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timing', type=pathlib.Path)
    parser.add_argument('--work', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    a = parser.parse_args()
    result = {}
    if a.timing:
        result['timing'] = timings(a.timing)
    if a.work:
        result['output_dumps'] = dumps(a.work)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(a.output), 'timing_rows': result.get('timing', {}).get('raw_rows'),
                      'dump_groups': list(result.get('output_dumps', {}).get('groups', {}))}))
