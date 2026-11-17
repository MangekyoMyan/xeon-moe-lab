"""Offline audit of completed captures; never starts models or accesses hardware.

Route statistics describe this fixed teacher-forced fixture only. They are not
cache-hit predictions, throughput, model-quality, or collector-overhead tests.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    root = Path(root)
    d = json.loads((root / 'RUN.json').read_text())
    h = json.loads((root / 'health.json').read_text())
    launch = json.loads((root / 'LAUNCH_MANIFEST.json').read_text())
    t = d['trace']
    n = d['planned_capture_calls']
    assert n in (4, 68) and n == launch['planned_calls']
    assert d['passed'] is True and d['model_routes'] is True
    assert d['session_closed_before_owner_destruction'] is True
    assert h['status'] == 'exited' and h['child_returncode'] == 0
    assert not (root / 'STOP_REQUESTED').exists()
    assert len(d['tokens']) == 64 + n and d['prefix_tokens'] == 64
    assert d['vision_weights_loaded'] is True
    assert d['placement'] == [dict(layer=i, expert_weights_host=i < 37) for i in range(40)]
    assert len(d['providers']) == 16
    assert len(t['records']) == n * 37
    assert len(t['calls']) == len(t['attempts']) == len(d['call_results']) == n
    s = t['stats']
    for k, v in dict(records=n*37, attempts=n, calls=n,
                     complete_decode_attempts=n, dropped_events=0, flags=0).items():
        assert s[k] == v, (k, s[k], v)
    assert s['enabled'] is False
    for i in range(n):
        label = dict(request=0, sequence=0, decode_call=i, phase=2, logical_input_token=i)
        a, c, result = t['attempts'][i], t['calls'][i], d['call_results'][i]
        assert a['label'] == c['label'] == label
        for k, v in dict(id=i+1, first_record=37*i, records=37, layers=(1<<37)-1,
                         position=64+i, context=1, ubatch=0, n_tokens=1,
                         flags=0, graph_status=0).items():
            assert a[k] == v, (i, 'attempt', k)
        assert all(a[k] is True for k in ['observing','scope_t1','graph_returned','caller_completed'])
        for k, v in dict(first_attempt=i, attempts=1, flags=0, public_or_driver_status=0).items():
            assert c[k] == v, (i, 'call', k)
        assert c['completed'] is True and c['synchronized'] is True
        assert result['position'] == 64+i and result['logical_input_token'] == i
        assert result['decode_status'] == result['exception_stage'] == 0
        assert result['drain_called'] is False
        assert all(result[k] is True for k in ['label_accepted','decode_returned','output_called',
                   'output_valid','synchronized','finalize_called','finalize_ok','ok'])
        records = t['records'][37*i:37*(i+1)]
        assert sorted(r['layer'] for r in records) == list(range(37))
        for j, r in enumerate(records, 37*i):
            for k, v in dict(ordinal=j, attempt_id=i+1, request=0, sequence=0,
                             decode_call=i, logical_input_token=i, position=64+i,
                             phase=2, projection=1, n_tokens=1, ubatch=0,
                             context=1, generation=a['generation'], flags=0).items():
                assert r[k] == v, (i, 'record', j, k)
            assert len(r['ids']) == len(set(r['ids'])) == 6
            assert all(type(x) is int and 0 <= x < 384 for x in r['ids'])
    smoke = json.loads((root / 'SMOKE4.json').read_text())
    assert smoke['trace']['records'] == t['records'][:148]
    assert smoke['call_results'] == d['call_results'][:4]
    # The old schema's top-level passed stays false until the whole session closes.
    assert smoke['passed'] is False and smoke['trace']['stats']['flags'] == 0
    layer_summary = []
    for layer in range(37):
        rows = [r for r in t['records'] if r['layer'] == layer]
        sets = [set(r['ids']) for r in rows]
        freq = Counter(x for r in rows for x in r['ids'])
        overlap = Counter(len(a & b) for a, b in zip(sets, sets[1:]))
        layer_summary.append(dict(layer=layer, unique_experts=len(freq),
            unique_unordered_top6_sets=len({tuple(sorted(s)) for s in sets}),
            adjacent_overlap_0_to_6=[overlap[k] for k in range(7)],
            frequencies=sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))))
    samples = h['recent_samples']
    def peak(key):
        return max((x['gpu'][key] for x in samples if key in x.get('gpu',{})), default=None)
    result = dict(run=root.name, verified=True, calls=n, records=37*n,
        expert_selections=37*n*6, graph_generations=sorted({r['generation'] for r in t['records']}),
        child_returncode=h['child_returncode'], process_elapsed_seconds=h['finished_unix']-h['started_unix'],
        smoke4_coverage_verified=True,
        smoke4_passed_field_meaning='Whole-run final flag remains false before session close; not a smoke failure',
        sampled_guard=dict(retained_samples=len(samples),gpu_peak_C=peak('gpu_C'),hbm_peak_C=peak('hbm_C'),
            vram_peak_MiB=peak('vram_used_MiB'),power_peak_W=peak('power_W'),
            cpu_peak_C=max((v for x in samples for v in x.get('cpu_C',{}).values()), default=None),
            available_min_GiB=min((x['available_kib']/1024**2 for x in samples), default=None),
            fan_peak_percent=max((v for x in samples for v in x.get('fans',{}).get('percent',[])), default=None),
            gpu_query_timeouts=h['gpu_monitor']['timeout_count']),
        raw_sha256={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()},
        per_layer=layer_summary,
        limits=['Fixed repeated English teacher-forced text, one session, short context.',
                'No throughput, sampled-output equality, collector overhead, GPU readiness or hot-bank tests.',
                'Guard peaks cover retained periodic samples, not all instantaneous extrema.'])
    return result, d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing audit')
    pairs = [audit(p) for p in args.runs]
    out = dict(schema=1, verification_source_sha256=sha(Path(__file__)), runs=[p[0] for p in pairs])
    if len(pairs) == 2:
        a, b = pairs[0][1], pairs[1][1]
        out['same_prefix_and_first4_inputs'] = a['tokens'][:68] == b['tokens'][:68]
        out['first4_ordered_records_equal'] = a['trace']['records'][:148] == b['trace']['records'][:148]
        out['first4_top6_sets_equal'] = [sorted(r['ids']) for r in a['trace']['records'][:148]] == [sorted(r['ids']) for r in b['trace']['records'][:148]]
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(verified=True, runs=[dict(run=r['run'], calls=r['calls'], records=r['records']) for r in out['runs']], output=str(args.output))))


if __name__ == '__main__':
    main()
