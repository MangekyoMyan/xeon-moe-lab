#!/usr/bin/env python3
"""Read archived calibration only; no engine, model, API, supplied script or network."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

PIN = 'fa6671f7b5281bad28947fe5324e2495e792a86f'
REL = Path('archive/review-handoffs-not-for-readers/2026-11-17-route-calibration')
RUN = '20261117T013039Z-D-07d69e25'


def load(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def percentile_linear(values, q):
    a = sorted(values)
    x = (len(a)-1)*q
    i = int(x)
    return a[i] if i == len(a)-1 else a[i]+(a[i+1]-a[i])*(x-i)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.repo / REL
    case_defs = {x['case_id']: x for x in load(root/'CASES.json')['cases']}
    run = load(root/'runs'/RUN/'RUN.json')
    runtime_cases = {x['case_id']: x for x in run['cases']}
    names = [x['name'] for x in run['blocks'] if x['name'].startswith('screen_')]
    assert len(names) == 6
    sources = [root/'CASES.json', root/'runs'/RUN/'RUN.json']
    summaries = {}
    layers_by_case = {}
    matrices = {}
    for name in names:
        file = root/'runs'/RUN/(name+'.json')
        sources.append(file)
        block = load(file)
        case = case_defs[name]
        rc = runtime_cases[name]
        assert sha(case['text_utf8'].encode()) == rc['text_sha256']
        assert len(rc['full_tokens']) == rc['full_token_count']
        assert rc['used_prefix_tokens'] == block['prefix_tokens'] == 64
        assert block['calls'] == len(block['steps']) == 64
        assert block['position_first'] == 64 and block['position_last'] == 127
        assert block['mode'] == 'D2' and block['observing']
        assert 'pointer nonnull only' in block['output_validation']
        trace = block['trace']
        assert trace['stats'] == dict(records=2368, attempts=64, calls=64,
            dropped_events=0, complete_decode_attempts=64, flags=0, enabled=False, observing=True)
        assert len(trace['records']) == 2368
        matrix = [[None]*37 for _ in range(64)]
        for r in trace['records']:
            t,l = r['logical_input_token'], r['layer']
            assert 0 <= t < 64 and 0 <= l < 37 and matrix[t][l] is None
            assert r['position'] == 64+t and r['flags'] == 0 and r['generation'] == 1
            assert len(r['ids']) == len(set(r['ids'])) == 6
            assert all(0 <= e < 384 for e in r['ids'])
            matrix[t][l] = frozenset(r['ids'])
        assert all(v is not None for row in matrix for v in row)
        layers = [Counter(e for row in matrix for e in row[l]) for l in range(37)]
        uniques = [len(x) for x in layers]
        adjacent = [len(matrix[t][l] & matrix[t-1][l]) for t in range(1,64) for l in range(37)]
        dt = []
        for i, s in enumerate(block['steps']):
            assert s['index'] == i and s['position'] == 64+i
            marks = [s[k] for k in ('step_begin_ns','decode_begin_ns','decode_return_ns',
                'output_ready_ns','validation_end_ns','finalize_end_ns','step_end_ns')]
            assert marks == sorted(marks)
            assert s['decode_status'] == 0 and s['synchronized'] and s['ok']
            dt.append((s['output_ready_ns']-s['decode_begin_ns'])/1e6)
        summaries[name] = dict(full_tokens=rc['full_token_count'], used_prefix=64,
            captured_t1=64, text_bytes=len(case['text_utf8'].encode()),
            text_codepoints=len(case['text_utf8']), text_sha256=rc['text_sha256'],
            mean_engine_ms=statistics.mean(dt), median_engine_ms=statistics.median(dt),
            p95_engine_ms_linear=percentile_linear(dt,.95), first_engine_ms=dt[0],
            mean_remaining63_ms=statistics.mean(dt[1:]),
            layer_unique_experts_min=min(uniques), layer_unique_experts_median=statistics.median(uniques),
            layer_unique_experts_max=max(uniques), layer_unique_experts=uniques,
            mean_adjacent_common_experts=statistics.mean(adjacent),
            adjacent_histogram=[Counter(adjacent)[i] for i in range(7)],
            records=len(trace['records']), actual_generated_answer=False,
            numeric_scope=block['output_validation'])
        layers_by_case[name] = layers
        matrices[name] = matrix
    pairs = []
    for stem in ('screen_cache', 'screen_experiment'):
        ja,en = stem+'_ja', stem+'_en'
        # These token indices are intentionally NOT asserted to align semantics.
        positional = [len(matrices[ja][t][l] & matrices[en][t][l])
                      for t in range(64) for l in range(37)]
        jaccard = []
        tv = []
        for x,y in zip(layers_by_case[ja], layers_by_case[en]):
            u = set(x)|set(y)
            jaccard.append(len(set(x)&set(y))/len(u))
            tv.append(sum(abs(x[e]-y[e]) for e in u)/(2*64*6))
        pairs.append(dict(family=stem,
            full_text_ja_over_en_tokens=summaries[ja]['full_tokens']/summaries[en]['full_tokens'],
            ja_over_en_mean_engine_ms=summaries[ja]['mean_engine_ms']/summaries[en]['mean_engine_ms'],
            same_token_index_common_experts_mean=statistics.mean(positional),
            same_token_index_common_experts_histogram=[Counter(positional)[i] for i in range(7)],
            per_layer_union_jaccard_mean=statistics.mean(jaccard),
            per_layer_slot_frequency_total_variation_mean=statistics.mean(tv),
            limits=['Same token index is not semantic token alignment.',
                    'One block per case in a fixed order; content, position, cache and order are confounded.',
                    'Descriptive route differences do not identify a causal language effect or bank generalization.']))
    output = dict(schema=1,pinned_commit=PIN,scope='Independent read-only six-case analysis',
        cases=summaries,pairs=pairs,
        total_screen_calls=sum(x['captured_t1'] for x in summaries.values()),
        total_screen_records=sum(x['records'] for x in summaries.values()),
        bank_selection_performed=False,model_or_gpu_executed=False,
        source_files={str(f.relative_to(a.repo)):{'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())}
                      for f in sources})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'cases':{k:{kk:v[kk] for kk in ('full_tokens','mean_engine_ms','layer_unique_experts_median')} for k,v in summaries.items()},'pairs':pairs,'total_records':output['total_screen_records']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
