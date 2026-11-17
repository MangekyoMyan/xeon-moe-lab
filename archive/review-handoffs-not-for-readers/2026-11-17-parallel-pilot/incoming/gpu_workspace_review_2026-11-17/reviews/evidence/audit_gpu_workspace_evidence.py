#!/usr/bin/env python3
"""Read-only, stdlib-only independent audit of a pinned published handoff.

Does not import or execute repository scripts, binaries, models, GPU code, or API
calls. Writes only the explicitly requested audit output directory.
"""
import argparse
import array
from collections import Counter
import difflib
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import zipfile

PIN = 'd98a17c56fadc92cf79e21ecbe80732dedeed899'
REL = 'archive/review-handoffs-not-for-readers/2026-11-17-gpu-workspace'
ARCH = 'archive/review-handoffs-not-for-readers'
RUNS = {
    'U1024': ('', '20261117T031214Z-U1024-93e7fe85'),
    'capacity32': ('bank-capacity', '20261117T033327Z-U1024-0b6d5d01'),
    'train5': ('pilot-capture', '20261117T035402Z-U1024-d22b3e04'),
}
NORMAL_HASHES = {
    'libggml-cpu.so.0.28.0': '33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3',
    'libllama.so.0.8.0': '1d1e7d722c050856ddff874b5f7b87080925555e7eb9580a7ca37fcafb2a8f26',
    'libggml-cuda.so.0.28.0': '45f6b7d8334c415818f79b13add7c9710d552bbab2ff6c1f4c8bb5770f8de8f0',
    'libggml-base.so.0.28.0': '2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558',
}
CONFIG = '963b652592504501b6ec02880ab7e38303b7da782f7aca8e49ff0a9568d70d43'
SOURCE = {}
CHECKS = Counter()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    p = Path(p)
    b = p.read_bytes()
    SOURCE[str(p)] = {'bytes': len(b), 'sha256': sha_bytes(b)}
    return b


def sha(p):
    return sha_bytes(read(p))


def obj(p):
    return json.loads(read(p))


def check(truth, label):
    if not truth:
        raise AssertionError(label)
    CHECKS[label.split(':', 1)[0]] += 1


def eq(a, b, label):
    check(a == b, label + ': ' + str(a)[:160] + ' vs ' + str(b)[:160])


def ids_sha(ids):
    return sha_bytes(struct.pack('<' + 'i' * len(ids), *ids))


def manifest(root, entries):
    for name, meta in entries.items():
        data = read(root / name)
        expected = meta['sha256'] if isinstance(meta, dict) else meta
        eq(sha_bytes(data), expected, 'file hash: ' + name)
        if isinstance(meta, dict) and 'bytes' in meta:
            eq(len(data), meta['bytes'], 'file size: ' + name)
    return len(entries)


def integers_plan(data, inventory):
    lines = data.decode().splitlines()
    hdr = lines[0].split()
    eq(hdr[:5], ['XEON_HOT_FFN_PLAN_V1', '40', '384', '5120', '2304'], 'plan header')
    eq(len(lines), 41, 'plan 40 layers')
    result = {}
    for layer, line in enumerate(lines[1:]):
        fields = line.split()
        eq(int(fields[0]), layer, 'plan layer order')
        eq(fields[1:4], [p['type'].upper() for p in inventory[layer]['projections']], 'plan native types')
        n, ids = int(fields[4]), list(map(int, fields[5:]))
        eq(n, len(ids), 'plan count')
        eq(ids, sorted(set(ids)), 'plan sorted unique IDs')
        check(all(0 <= e < 384 for e in ids), 'plan ID range')
        result[layer] = ids
    count = sum(map(len, result.values()))
    payload = sum(len(ids) * inventory[l]['native_triplet_bytes'] for l, ids in result.items())
    eq([count, payload], list(map(int, hdr[5:])), 'plan total count payload')
    return result, count, payload


def verify_ledger(t, prefix, count, prefill):
    s = t['stats']
    eq(s['enabled'], False, 'ledger closed')
    eq([s['flags'], s['dropped_events']], [0, 0], 'ledger flags drops')
    n = 1 if prefill else count
    eq([s['calls'], len(t['calls'])], [n, n], 'ledger call count')
    eq(s['observing'], not prefill, 'ledger observing')
    eq([s['records'], len(t['records'])], [0, 0] if prefill else [37 * count] * 2, 'ledger record count')
    eq(s['complete_decode_attempts'], 0 if prefill else count, 'ledger complete decode count')
    eq(s['attempts'], len(t['attempts']), 'ledger attempt count')
    if not prefill:
        eq(len(t['attempts']), count, 'ledger one T1 attempt')
    position = 0
    for i, a in enumerate(t['attempts']):
        label = {'request': 0, 'sequence': 0, 'decode_call': 0 if prefill else i,
                 'phase': 1 if prefill else 2, 'logical_input_token': 0 if prefill else i}
        eq(a['label'], label, 'attempt label')
        eq([a['id'], a['context'], a['flags'], a['graph_status']], [i+1, 1, 0, 0], 'attempt identity status')
        check(a['graph_returned'] and a['caller_completed'], 'attempt completed')
        check(a['generation'] >= 1, 'attempt generation positive')
        eq([a['observing'], a['scope_t1']], [not prefill, not prefill], 'attempt scope')
        eq(a['position'], position if prefill else prefix+i, 'attempt position')
        check(a['n_tokens'] > 0, 'attempt positive tokens')
        if prefill:
            eq([a['first_record'], a['records'], a['layers'], a['ubatch']], [0, 0, 0, i], 'prefill no route record')
            position += a['n_tokens']
        else:
            eq([a['first_record'], a['records'], a['layers'], a['ubatch'], a['n_tokens']],
               [37*i, 37, (1 << 37)-1, 0, 1], 'T1 layer mask')
        if not prefill:
            for j, r in enumerate(t['records'][37*i:37*(i+1)]):
                expected = {'ordinal': 37*i+j, 'attempt_id': i+1, 'request': 0, 'sequence': 0,
                            'decode_call': i, 'ubatch': 0, 'logical_input_token': i,
                            'position': prefix+i, 'generation': a['generation'], 'context': 1,
                            'layer': j, 'phase': 2, 'projection': 1, 'n_tokens': 1, 'flags': 0}
                for key, value in expected.items():
                    eq(r[key], value, 'record ' + key)
                eq(len(r['ids']), 6, 'record six experts')
                eq(len(set(r['ids'])), 6, 'record unique experts')
                check(all(0 <= e < 384 for e in r['ids']), 'record ID range')
    if prefill:
        eq(position, prefix, 'prefill sum tokens')
    for i, c in enumerate(t['calls']):
        label = {'request': 0, 'sequence': 0, 'decode_call': 0 if prefill else i,
                 'phase': 1 if prefill else 2, 'logical_input_token': 0 if prefill else i}
        eq(c['label'], label, 'call label')
        eq([c['flags'], c['public_or_driver_status']], [0, 0], 'call status')
        check(c['synchronized'] and c['completed'], 'call completed')
        eq([c['first_attempt'], c['attempts']], [0, len(t['attempts'])] if prefill else [i, 1], 'call attempts')


def f32(path, nv):
    data = read(path)
    eq(len(data), 5 * nv * 4, 'F32 byte size')
    values = array.array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    check(all(map(math.isfinite, values)), 'saved full F32 finite')
    return data, values


def verify_case(runpath, b, full_tokens, nv):
    prefix, n = b['prefix_tokens'], len(b['T1'])
    check(n in (4, 64), 'case bounded length')
    eq(b['logical_prefill_output_positions'], [prefix-1], 'case final prefill position')
    eq(b['continuation_inputs'], full_tokens[prefix:prefix+n], 'case input token identity')
    targets = full_tokens[prefix:prefix+n+1]
    target_key = 'next_token_targets_prefix_and_all_T1' if 'next_token_targets_prefix_and_all_T1' in b else 'next_token_targets_for_rows'
    eq(b[target_key], targets, 'case next token targets')
    if 'next_token_targets_for_saved_rows' in b:
        eq(b['next_token_targets_for_saved_rows'], targets[:5], 'case saved row targets')
    verify_ledger(b['prefill_trace'], prefix, 1, True)
    verify_ledger(b['T1_trace'], prefix, n, False)
    actual = [a['n_tokens'] for a in b['prefill_trace']['attempts']]
    eq(actual, b['actual_ubatches'], 'case raw actual ubatches')
    eq(actual, b['expected_ubatches'], 'case expected ubatches')
    eq(actual, [prefix], 'case single actual ubatch')
    ts = [b[k] for k in ['prefill_decode_begin_ns', 'prefill_decode_return_ns', 'prefill_output_ready_ns', 'prefill_finite_end_ns']]
    eq(ts, sorted(ts), 'prefill monotonic timestamps')
    for i, step in enumerate(b['T1']):
        eq([step['index'], step['position']], [i, prefix+i], 'T1 step identity')
        eq(step['boundaries_ns'], sorted(step['boundaries_ns']), 'T1 monotonic timestamps')
        eq(step['decode_status'], 0, 'T1 decode success')
        check(step['full_logits_finite'] and step['synchronized'], 'T1 recorded finite synchronized')
        if 'next_target_token_id' in step:
            eq(step['next_target_token_id'], targets[i+1], 'T1 predicts subsequent token')
    eq(b['logits_shape'], [5, nv], 'case saved logits shape')
    check(b['all_saved_logits_finite'], 'case recorded saved finite')
    data, values = f32(runpath/b['logits_file'], nv)
    gens = [a['generation'] for a in b['T1_trace']['attempts']]
    changes = [{'decode_call': i, 'position': prefix+i, 'before': gens[i-1], 'after': gens[i]}
               for i in range(1, n) if gens[i] != gens[i-1]]
    result = {'case_id': b['case_id'], 'prefix_tokens': prefix, 'T1_calls': n,
              'actual_ubatches': actual, 'logits_sha256': sha_bytes(data), 'saved_F32_scalars': len(values),
              'position_range': [prefix, prefix+n-1], 'generation_transitions': changes,
              'generation_call_counts': dict(Counter(gens)),
              'prefill_ms': (ts[2]-ts[0])/1e6,
              'T1_engine_ms': [(s['boundaries_ns'][3]-s['boundaries_ns'][1])/1e6 for s in b['T1']],
              'independently_checked_finite_rows': 5, 'recorded_finite_T1_calls': n}
    return result, data, values


def miss_runs(seq):
    lengths, run = [], 0
    for h in seq:
        if h == 0:
            run += 1
        elif run:
            lengths.append(run); run = 0
    if run:
        lengths.append(run)
    return lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--previous-zip', type=Path)
    args = p.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    w = repo / REL
    check(not out.is_relative_to(repo), 'output outside repository')
    result = {'schema': 1, 'review_pin': PIN,
              'method': 'Independent stdlib-only read/recompute; no supplied code execution/import, engine, model, GPU, or API execution.',
              'source_scope': REL}
    shared = obj(w/'FILES_SHA256.json')
    result['manifest_files'] = manifest(w, shared)
    actual = {str(x.relative_to(w)) for x in w.rglob('*') if x.is_file()}
    eq(actual, set(shared) | {'FILES_SHA256.json'}, 'shared exact fileset')
    mapping = obj(w/'RAW_MAPPING.json')
    for entry in mapping:
        eq(sha(w/entry['shared']), entry['sha256'], 'raw mapping hash')
        check(entry['byte_identical'], 'raw mapping recorded byte identity')
    eq({x['shared'] for x in mapping}, set(shared)-{'RAW_MAPPING.json'}, 'raw mapping full coverage')
    result['raw_mapping_entries'] = len(mapping)

    incoming = w/'incoming/xeon_moe_route_calibration_review_2026-11-17'
    inc = obj(incoming/'FILES_SHA256.json')
    eq(inc['review_pin'], 'fa6671f7b5281bad28947fe5324e2495e792a86f', 'incoming prior pin')
    incoming_entries = {x['path']: x for x in inc['files']}
    eq(manifest(incoming, incoming_entries), 34, 'incoming 34 entries')
    eq({str(x.relative_to(incoming)) for x in incoming.rglob('*') if x.is_file()}, set(incoming_entries)|{'FILES_SHA256.json'}, 'incoming exact 35 files')
    intake = obj(w/'INTAKE.json')
    result['incoming'] = {'manifest_entries': 34, 'total_with_manifest': 35, 'receipt_zip_sha256': intake['sha256'], 'zip_independently_available': False}
    if args.previous_zip and args.previous_zip.exists():
        zdata = read(args.previous_zip)
        eq(sha_bytes(zdata), intake['sha256'], 'previous ZIP hash')
        with zipfile.ZipFile(args.previous_zip) as z:
            names = [n for n in z.namelist() if not n.endswith('/')]
            check(len(names) == 35, 'previous ZIP member count')
            for name in names:
                matching = [p for p in (set(incoming_entries)|{'FILES_SHA256.json'}) if name == p or name.endswith('/'+p)]
                eq(len(matching), 1, 'previous ZIP unique local match')
                eq(z.read(name), read(incoming/matching[0]), 'previous ZIP member identical')
        result['incoming'].update(zip_independently_available=True, zip_bytes=len(zdata), zip_members=len(names))

    old_tree = repo/ARCH/'2026-11-16-current/snapshot/llm-lab'
    old_hot = read(old_tree/'exploration-20261112/hot-ffn/hot-backend.cpp')
    new_hot = read(w/'bank-capacity/hot-backend.cpp')
    bank_design = obj(w/'bank-capacity/BANK_DESIGN.json')
    eq(sha_bytes(old_hot),bank_design['original_hot_backend_sha256'],'bank original source identity')
    patch = ''.join(difflib.unified_diff(old_hot.decode().splitlines(True),new_hot.decode().splitlines(True),
                                      fromfile='original/hot-backend.cpp',tofile='capacity/hot-backend.cpp'))
    eq(patch.encode(),read(w/'bank-capacity/hot-backend-capacity.patch'),'bank published patch is exact diff')
    for text in ['require(free>payload+size_t(3072)*1024*1024,', 'require(free>size_t(2560)*1024*1024,']:
        check(text in old_hot.decode() and text in new_hot.decode(),'bank strict guard source unchanged')
    ggml_common = read(old_tree/'deepseek41-bringup-20261101/source/ggml/src/ggml-common.h').decode()
    for text in ['#define QK_K 256','#define QK_MXFP4 32',
                 'sizeof(block_iq2_s) == sizeof(ggml_half) + QK_K/4 + QK_K/16',
                 'sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8)',
                 'sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE',
                 'sizeof(block_mxfp4) == sizeof(uint8_t) + QK_MXFP4/2']:
        check(text in ggml_common,'native block definition source')
    result['source_provenance']={'hot_backend_patch_exact':True,'original_hot_sha256':sha_bytes(old_hot),
                                 'capacity_hot_sha256':sha_bytes(new_hot),'strict_guards_unchanged':True,
                                 'native_block_element_and_byte_sizes':{'iq2_s':[256,82],'iq3_xxs':[256,98],'iq3_s':[256,110],'mxfp4':[32,17]}}

    old = repo/ARCH/'2026-11-17-route-calibration/runs/20261117T012730Z-F-before-a30e72f7'
    old_f32 = read(old/'correct_F.f32')
    eq(sha_bytes(old_f32), '95430744558e7414b336f6548a6c03e064e12e1adc03cb8256fc45dcfc3187b0', 'previous F bytes')
    old_run = obj(old/'RUN.json')
    caseobjs, arrays, raws, reports = {}, {}, {}, {}
    result['runs'] = {}
    for tag, (sub, runname) in RUNS.items():
        base, rd = w/sub, w/sub/'runs'/runname
        rr = obj(rd/'RUN.json'); reports[tag] = rr
        build, launch, pin = obj(rd/'BUILD.json'), obj(rd/'LAUNCH_MANIFEST.json'), obj(rd/'PROVIDER_EXPECTED.json')
        eq(read(base/'BUILD.json'), read(rd/'BUILD.json'), 'build copied exactly')
        eq(build['returncode'], 0, 'build return code')
        manifest(base, build['source_sha256'])
        eq(sha(base/'run_capacity.py'), launch['runner_sha256'], 'runner source pin')
        eq(sha(rd/'BUILD.json'), launch['build_sha256'], 'launch build pin')
        eq(sha(old/'RESULTS_SHA256.json'), launch['reference_manifest_sha256'], 'launch previous results pin')
        eq(launch['baseline_config_sha256'], CONFIG, 'launch baseline config')
        eq(launch['reserve_GiB'], 64, 'launch RAM guard')
        eq(launch['ubatch'], 1024, 'launch ubatch')
        eq(launch['bank'], tag == 'capacity32', 'launch bank status')
        if tag == 'capacity32':
            check(not launch['bank_used_by_model'] and not rr['bank_used_by_model'] and rr['capacity_bank_initialized'], 'fixture not connected to model')
            eq(launch['bank_plan_sha256'], sha(base/'capacity.plan'), 'fixture plan launch pin')
        if tag == 'train5':
            eq(launch['pilot_design_sha256'], sha(base/'PILOT_DESIGN.json'), 'pilot design launch pin')
        hashes = obj(rd/'RESULTS_SHA256.json')
        reconstructed = (json.dumps({'command': launch['command'], 'environment': launch['environment']}, indent=2)+'\n').encode()
        eq(sha_bytes(reconstructed), hashes['server-config.json'], 'omitted config reconstructed hash')
        for n, h in hashes.items():
            if n != 'server-config.json':
                eq(sha(rd/('guard-health.json' if n == 'health.json' else n)), h, 'run results hash')
        eq(rr['providers'], pin['providers'], 'actual symbol providers')
        eq(rr['active_backend_providers'], pin['active_backend_providers'], 'actual graph providers')
        eq([x['name'] for x in rr['active_backend_providers']], ['CUDA0', 'CPU'], 'two model backends')
        eq(rr['cparams'], {'n_ctx':131072,'n_batch':1024,'n_ubatch':1024,'n_seq_max':1,'n_outputs_max':1,'n_outputs_max_per_seq':1,'threads':12,'threads_batch':24,'n_rs_seq':0}, 'actual context settings')
        check(rr['passed'] and rr['session_closed_before_owner_destruction'] and rr['vision_weights_loaded'], 'run completion')
        eq(rr['vocab_size'], 129280, 'vocabulary')
        eq(rr['placement'], [{'layer':i,'host':i<37} for i in range(40)], 'expert placement')
        for layer in rr['native_expert_inventory']:
            total = 0
            for projection in layer['projections']:
                bs, size = {'iq2_s':(256,82),'iq3_xxs':(256,98),'iq3_s':(256,110),'mxfp4':(32,17)}[projection['type']]
                ne = projection['ne']
                eq(ne[0] % bs, 0, 'native block aligned')
                eq(ne[2:], [384, 1], 'native expert dimensions')
                native = ne[0]//bs*size*ne[1]
                eq(native, projection['native_expert_bytes'], 'native projection bytes independent')
                eq(native, projection['source_expert_stride_bytes'], 'native source stride')
                check(projection['contiguous'], 'native contiguous')
                total += native
            eq(total, layer['native_triplet_bytes'], 'native triplet sum')
        raw_cases = obj(base/'CASES.json')['cases']
        eq(len(raw_cases), len(rr['cases']), 'raw case count')
        for inp, actual_case in zip(raw_cases, rr['cases']):
            eq(sha_bytes(inp['text_utf8'].encode()), inp['sha256_utf8'], 'input text hash')
            eq(actual_case['full_tokens'], inp['expected_token_ids'], 'runtime tokenizer identity receipt')
            eq(actual_case['full_token_count'], len(inp['expected_token_ids']), 'runtime full token count')
            eq(actual_case['text_sha256'], inp['sha256_utf8'], 'runtime text hash')
        eq(rr['cases'][0]['full_tokens'], old_run['cases'][0]['full_tokens'], 'short control original tokens')
        lookup = {c['case_id']:c['full_tokens'] for c in rr['cases']}
        cases = {}
        for name in rr['completed_cases']:
            b = obj(rd/(name+'.json')); caseobjs[(tag,name)] = b
            cs, data, values = verify_case(rd, b, lookup[b['case_id']], rr['vocab_size'])
            cases[name], raws[(tag,name)], arrays[(tag,name)] = cs, data, values
            if name == 'short64':
                eq(data[rr['vocab_size']*4:], old_f32, 'short control exact old T1 bytes')
        memory = []
        for phase in rr['memory_phases']:
            c, n = phase['cuda'], phase['nvml_cli']
            eq(n['status'], 0, 'memory NVML status')
            ts = [c['begin_ns'], c['end_ns'], n['begin_ns'], n['end_ns']]
            eq(ts, sorted(ts), 'memory queries sequential')
            csv = [v.strip() for v in n['raw_csv'].strip().split(',')]
            total, used, free = map(float, csv[-3:])
            check(total == used+free+273, 'NVML reserved accounting')
            memory.append({'phase': phase['phase'], 'cuda_free_MiB': c['free_bytes']/2**20,
                           'cuda_total_MiB': c['total_bytes']/2**20, 'nvml_total_MiB':total,
                           'nvml_used_MiB':used, 'nvml_free_MiB':free,
                           'query_span_ms':(n['end_ns']-c['begin_ns'])/1e6})
        health = obj(rd/'guard-health.json')
        eq([health['status'],health['child_returncode']], ['exited',0], 'guard child exited normally')
        eq(health['limits'], {'gpu_C':60,'hbm_C':65,'cpu_C':60,'peripheral_C':45,'pch_C':55,'fan_percent':70}, 'guard unchanged limits')
        samples = health['recent_samples']
        for s in samples:
            eq([s['gpu']['limit_W'],s['gpu']['ecc'],s['gpu']['ecc_pending']], [150.0,'Enabled','Enabled'], 'saved power ECC settings')
            eq([s['gpu']['ecc_single'], s['gpu']['ecc_double']], [0.0,0.0], 'saved ECC errors')
            check(s['available_kib'] >= 64*2**20, 'saved available RAM guard')
        hs = {'samples_saved':len(samples), 'gpu_C_max':max(s['gpu']['gpu_C'] for s in samples),
              'hbm_C_max':max(s['gpu']['hbm_C'] for s in samples),
              'cpu_C_max':max(v for s in samples for v in s['cpu_C'].values()),
              'fan_percent_max':max(v for s in samples for v in s['fans']['percent']),
              'RAM_GiB_min':min(s['available_kib']/2**20 for s in samples),
              'monitor':health['gpu_monitor']}
        result['runs'][tag] = {'run':runname,'subdir':sub,'cases':cases,'memory':memory,'health_saved_tail':hs,
                               'run_result_hash_entries':len(hashes),'omitted_server_config_hash_reconstructed':True,
                               'engine_libraries_recorded':build['engine_libraries'],'active_backend_providers':rr['active_backend_providers']}

    inventory = reports['train5']['native_expert_inventory']
    eq(inventory, reports['U1024']['native_expert_inventory'], 'same inventory U1024 pilot')
    eq(inventory, reports['capacity32']['native_expert_inventory'], 'same inventory capacity pilot')
    fixture, nfixture, payload = integers_plan(read(w/'bank-capacity/capacity.plan'), inventory)
    eq({l:es for l,es in fixture.items() if es}, {l:list(range(8)) for l in [0,4,23,24]}, 'fixture predeclared membership')
    stderr = read(w/'bank-capacity/runs'/RUNS['capacity32'][1]/'server.stderr.log').decode(errors='replace')
    pre = re.findall(r'capacity-bank-Gpre free=(\d+) total=(\d+) payload=(\d+) margin=(\d+)', stderr)
    post = re.findall(r'capacity-bank-Gpost free=(\d+) total=(\d+) margin=(\d+)', stderr)
    warm = [tuple(map(int,x)) for x in re.findall(r'capacity-bank-warm layer=(\d+) H=(\d+) repeats=(\d+) zero_output=(\d+)', stderr)]
    ready = re.findall(r'xeon-hot-ready experts=(\d+) native_payload=(\d+) GPU_bank_allocation=(\d+) GPU_free_before_vision=(\d+) CPUthreads=(\d+) original-weights-retained=(\d+)', stderr)
    eq([len(pre),len(post),len(ready)],[1,1,1], 'single actual admission logs')
    pf,pt,pp,pm = map(int,pre[0]); qf,qt,qm = map(int,post[0]); count,np,alloc,fr,threads,retained = map(int,ready[0])
    eq([pp,np,count,threads,retained],[payload,payload,32,12,1], 'bank ready identity')
    eq([pm,qm],[3072*2**20,2560*2**20], 'bank unchanged margins')
    check(pf>payload+pm and qf>qm, 'bank strict guards pass')
    eq(warm,[(l,h,3,1) for l in [0,4,23,24] for h in range(1,7)], 'bank reachable warmup cells')
    eq(fr,qf,'bank ready post free')
    fixture_result = {'experts':count,'selected_layers':[0,4,23,24],'native_payload_bytes':payload,
                      'bank_buffer_allocation_bytes':alloc,'padding_bytes':alloc-payload,
                      'Gpre_free_bytes':pf,'Gpre_margin_bytes':pm,'Gpre_headroom_bytes':pf-payload-pm,
                      'Gpost_free_bytes':qf,'Gpost_margin_bytes':qm,'Gpost_headroom_bytes':qf-qm,
                      'native_payload_MiB':payload/2**20,'bank_buffer_allocation_MiB':alloc/2**20,
                      'initial_total_free_delta_MiB':(pf-qf)/2**20,
                      'delta_minus_buffer_MiB':(pf-qf-alloc)/2**20,'GPU_graph_calls_recorded':sum(x[2] for x in warm),
                      'zero_output_readback_cells_recorded':len(warm),'all_72_intermediate_outputs_saved_or_checked':False,
                      'model_used_bank':False,'actual_strict_Gpre_pass':True,'actual_strict_Gpost_pass':True}
    result['capacity_fixture'] = fixture_result

    comparison = obj(w/'BANK_COMPARISON.json')
    eq(sha(w/'analyze_capacity.py'),comparison['source_sha256'],'bank comparison source pin')
    numeric = []
    for name, orig in zip(['short64','long1024'],comparison['comparisons']):
        eq(raws[('U1024',name)],raws[('capacity32',name)],'bank main logits byte equality')
        b = caseobjs[('U1024',name)]; a = arrays[('U1024',name)]; nv = 129280
        targets = b['next_token_targets_for_rows']
        rows = []
        for i in range(5):
            vals = a[i*nv:(i+1)*nv]
            top2 = sorted(range(nv), key=lambda k: (-vals[k],k))[:2]
            largest = vals[top2[0]]
            logp = vals[targets[i]] - largest - math.log(math.fsum(math.exp(v-largest) for v in vals))
            expected = orig['rows'][i]
            eq(top2, expected['top2_baseline'], 'comparison top2')
            eq(top2, expected['top2_candidate'], 'comparison candidate top2')
            eq(vals[top2[0]]-vals[top2[1]],expected['top1_margin_baseline'], 'comparison top1 margin')
            check(abs(logp-expected['target_logprob_baseline'])<2e-12,'comparison target logprob independent')
            eq([expected['max_abs'],expected['RMSE'],expected['relative_L2']],[0.,0.,0.], 'comparison zero error')
            rows.append({'target':targets[i],'top2':top2,'margin':vals[top2[0]]-vals[top2[1]],'target_logprob':logp})
        numeric.append({'case':name,'byte_equal':True,'rows':rows,'scalars':len(a)})
    result['main_model_saved_logits_comparison'] = numeric
    for tag, saved in zip(['U1024','capacity32'],comparison['runs']):
        eq(result['runs'][tag]['memory'],saved['memory'],'BANK_COMPARISON memory recompute')
        for c in saved['cases']:
            own = result['runs'][tag]['cases'][c['name']]
            for k in ['prefix_tokens','actual_ubatches','logits_sha256','prefill_ms','T1_engine_ms']:
                eq(own[k],c[k],'BANK_COMPARISON '+k)

    frozen = obj(w/'PILOT_TOKENS.json')
    corpus = obj(incoming/'pilot/PILOT_CASES_JA_EN_CODE.json')
    stage = obj(incoming/'pilot/PILOT_STAGE_MANIFEST.json')
    design = obj(w/'pilot-capture/PILOT_DESIGN.json')
    template = obj(w/'TEMPLATE_SCREEN.json')
    eq(sha(incoming/'pilot/PILOT_CASES_JA_EN_CODE.json'),frozen['corpus_sha256'],'frozen corpus identity')
    eq(sha(w/'PILOT_TOKENS.json'),design['all_token_manifest_sha256'],'frozen token manifest design pin')
    eq(sha_bytes(template['template_properties']['chat_template'].encode()),frozen['chat_template_sha256'],'chat template identity')
    eq(frozen['API_config_sha256'],CONFIG,'template normal config identity')
    eq(frozen['endpoints'],['/props','/apply-template','/tokenize'],'template endpoints only')
    eq([frozen['model_inference_requested'],frozen['route_collected'],frozen['bank_fitted']],[False]*3,'template no inference receipt')
    eq(frozen['template_kwargs'],{'thinking_mode':'chat'},'template chat mode receipt')
    variants = {}
    for family in corpus['families']:
        for v in family['variants']:
            variants[v['case_id']] = (family,v)
    eq(len(variants),20,'corpus 20 cases')
    eq(len(corpus['families']),12,'corpus 12 families')
    eq(set(variants),{c['case_id'] for c in frozen['cases']},'frozen all original cases')
    token_cases = {}
    splits = Counter()
    for c in frozen['cases']:
        f,v = variants[c['case_id']]
        eq([c['family_id'],c['split'],c['language_or_content']],[f['family_id'],f['split'],v['language_or_content']], 'frozen original identity split')
        eq(c['prompt_sha256'],sha_bytes(v['prompt_utf8'].encode()),'frozen prompt hash')
        eq(c['reference_sha256'],sha_bytes(v['reference_utf8'].encode()),'frozen reference hash')
        eq(c['rendered_prefix_sha256'],sha_bytes(c['rendered_prefix'].encode()),'frozen rendered hash')
        eq(c['joined_text_sha256'],sha_bytes((c['rendered_prefix']+v['reference_utf8']).encode()),'frozen joined hash')
        eq(c['prefix_count'],len(c['prefix_ids']),'frozen prefix length')
        eq(c['reference_count'],len(c['full_reference_continuation_ids']),'frozen reference length')
        eq(c['prefix_ids_i32_sha256'],ids_sha(c['prefix_ids']),'frozen prefix IDs hash')
        ids = c['full_reference_continuation_ids']
        check(len(ids)>=65,'frozen at least 65 targets')
        eq(c['continuation_first65_ids'],ids[:65],'frozen first65')
        eq(c['used_T1_ids'],ids[:64],'frozen 64 inputs')
        eq(c['used_T1_ids_i32_sha256'],ids_sha(ids[:64]),'frozen T1 IDs hash')
        eq(c['next_targets_prefix_and64T1'],ids[:65],'frozen next token offset')
        eq(c['first_T1_position'],len(c['prefix_ids']),'frozen first T1 position')
        check(all(isinstance(x,int) and 0<=x<129280 for x in c['prefix_ids']+ids),'frozen ID range')
        check(c['rendered_prefix'].endswith('<｜Assistant｜></think>'),'frozen assistant boundary')
        check(c['strict_prefix_verified'],'frozen strict prefix receipt')
        decoded = bytes.fromhex(c['decoded_first64_bytes_hex'])
        eq(decoded.decode('utf-8',errors='replace'),c['decoded_first64'],'frozen displayed first64 bytes')
        check(v['reference_utf8'].encode().startswith(decoded),'frozen first64 byte prefix of reference')
        token_cases[c['case_id']] = c
        splits[c['split']] += 1
    eq(dict(splits),{'train':10,'validation':5,'holdout':5},'frozen split counts')
    first5 = next(s['cases'] for s in stage['stages'] if s['stage']=='train_first5')
    eq(first5,design['case_ids'],'pilot predeclared first5 order')
    eq(reports['train5']['completed_cases'],['short64']+first5,'pilot only first5 plus control')
    runtime_cases = {c['case_id']:c for c in reports['train5']['cases']}
    for name in first5:
        c = token_cases[name]; b = caseobjs[('train5',name)]
        eq(c['split'],'train','pilot exclusively train')
        eq(runtime_cases[name]['full_tokens'],c['prefix_ids']+c['full_reference_continuation_ids'],'pilot exact frozen full IDs')
        eq(b['prefix_tokens'],c['prefix_count'],'pilot full prompt prefix')
        eq(b['continuation_inputs'],c['used_T1_ids'],'pilot first64 frozen input IDs')
        eq(b['next_token_targets_prefix_and_all_T1'],c['continuation_first65_ids'],'pilot 65 target correspondence')
    check(frozen['unix'] < obj(w/'pilot-capture/runs'/RUNS['train5'][1]/'guard-health.json')['started_unix'], 'token manifest before pilot run receipt')
    result['tokens'] = {'cases':20,'families':12,'splits':dict(splits),'prefix_token_range':[min(c['prefix_count'] for c in token_cases.values()),max(c['prefix_count'] for c in token_cases.values())],
                        'reference_token_range':[min(c['reference_count'] for c in token_cases.values()),max(c['reference_count'] for c in token_cases.values())],
                        'first5_order':first5,'withheld_15_route_results_in_this_handoff':False,
                        'independently_retokenized':False,'strict_prefix_supported_by_source_and_success_receipt':True,
                        'template_sha256':frozen['chat_template_sha256'],'corpus_sha256':frozen['corpus_sha256'],
                        'per_case':[{'case_id':c['case_id'],'family':c['family_id'],'split':c['split'],'prefix':c['prefix_count'],'reference':c['reference_count']} for c in token_cases.values()]}

    proposal = obj(w/'TRAIN5_PROPOSAL.json')
    eq(sha(w/'analyze_pilot.py'),proposal['analysis_source_sha256'],'proposal source pin')
    for name in first5:
        rd=w/'pilot-capture/runs'/RUNS['train5'][1]
        eq(proposal['sources'][name],{'result_sha256':sha(rd/(name+'.json')),'logits_sha256':sha(rd/(name+'.f32')),'prefix_tokens':token_cases[name]['prefix_count']},'proposal raw source hashes')
    freq = Counter((r['layer'],e) for name in first5 for r in caseobjs[('train5',name)]['T1_trace']['records'] for e in r['ids'])
    ranks = sorted(freq,key=lambda le:(-Fraction(freq[le],inventory[le[0]]['native_triplet_bytes']),le[0],le[1]))
    ranked_rows = [{'layer':l,'expert':e,'count':freq[l,e],'native_triplet_bytes':inventory[l]['native_triplet_bytes']} for l,e in ranks]
    eq(ranked_rows,proposal['ranked_training_pairs'],'entire exact Fraction ranking')
    budget = design['bank_selection_budget_bytes']; eq(budget,payload,'candidate predeclared native budget')
    used,chosen = 0,[]
    for le in ranks:
        size = inventory[le[0]]['native_triplet_bytes']
        if used+size<=budget:
            chosen.append(le); used += size
    selected = [{'layer':l,'expert':e,'training_inclusion_count':freq[l,e],'native_triplet_bytes':inventory[l]['native_triplet_bytes']} for l,e in sorted(chosen)]
    eq(selected,proposal['candidate']['selected'],'candidate greedy native budget selection')
    candplan, n, planbytes = integers_plan(read(w/'TRAIN5_CANDIDATE.plan'),inventory)
    eq({(l,e) for l,es in candplan.items() for e in es},set(chosen),'candidate plan exact chosen IDs')
    eq([n,planbytes,used,budget],[35,475453440,475453440,480706560],'candidate byte arithmetic')
    eq(sha(w/'TRAIN5_CANDIDATE.plan'),proposal['candidate']['plan_sha256'],'candidate plan hash')
    plan_lines = [f'XEON_HOT_FFN_PLAN_V1 40 384 5120 2304 {n} {used}']
    for l in range(40):
        types = ' '.join(p['type'].upper() for p in inventory[l]['projections'])
        es = sorted(e for ll,e in chosen if ll==l)
        plan_lines.append(f'{l} {types} {len(es)}'+(' '+' '.join(map(str,es)) if es else ''))
    eq(('\n'.join(plan_lines)+'\n').encode(),read(w/'TRAIN5_CANDIDATE.plan'),'candidate independently regenerated plan bytes')
    htotal, allmiss, percase = Counter(),Counter(),[]
    chosen_set=set(chosen); selectedlayers={l for l,e in chosen}
    selected_h=Counter()
    for name,orig in zip(first5,proposal['per_case']):
        h_by_layer={l:[None]*64 for l in range(37)}
        for r in caseobjs[('train5',name)]['T1_trace']['records']:
            h = sum((r['layer'],e) in chosen_set for e in r['ids'])
            h_by_layer[r['layer']][r['decode_call']] = h
            if r['layer'] in selectedlayers: selected_h[h]+=1
        hist=Counter(v for hs in h_by_layer.values() for v in hs)
        htotal.update(hist)
        remaining=[222-sum(h_by_layer[l][i] for l in range(37)) for i in range(64)]
        layers=[]
        for l, hs in h_by_layer.items():
            runs=miss_runs(hs); allmiss.update(runs)
            layers.append({'layer':l,'H_each':hs,'all_miss_run_lengths':runs})
        own={'case_id':name,'family_id':token_cases[name]['family_id'],'lane':token_cases[name]['language_or_content'],
             'H_histogram':[hist[h] for h in range(7)],'mean_H':sum(h*n for h,n in hist.items())/(37*64),
             'remaining_CPU_expert_slots_per_token':remaining,'per_layer':layers}
        eq(own,orig,'entire per case H distribution and miss runs')
        percase.append(own)
    hist=[htotal[h] for h in range(7)]; hit=sum(h*n for h,n in htotal.items()); cells=sum(htotal.values())
    geom={'H_histogram':hist,'mean_H':hit/cells,'selected_slot_fraction':hit/(cells*6),
          'H0_fraction':htotal[0]/cells,'H6_fraction':htotal[6]/cells,
          'all_miss_run_histogram':{str(k):v for k,v in sorted(allmiss.items())},'miss_runs_reset_at_each_case_and_layer':True}
    eq(geom,proposal['training_only_geometry'],'full aggregate H geometry')
    eq([proposal['training_calls'],proposal['records'],proposal['expert_selection_slots']],[320,cells,sum(freq.values())],'proposal raw denominator counts')
    eq(sum(freq[x] for x in chosen),hit,'selected frequency equals H hits')
    for name in ['saved_time','GPU_ready','actual_candidate_admission','quality_acceptance']:
        eq(proposal[name],None,'proposal unmeasured remains null')
    eq(proposal['validation_or_holdout_routes_used'],False,'proposal no validation holdout')
    eq(proposal['controls_excluded_from_selection'],['short64'],'proposal control excluded')
    result['candidate35']={'native_budget_bytes':budget,'native_payload_bytes':used,'native_payload_MiB':used/2**20,
                           'unspent_native_bytes':budget-used,'selected_experts':n,'selected_layers':len(selectedlayers),'observed_distinct_pairs':len(freq),
                           'selected':selected,'geometry':geom,'hit_slots':hit,'total_slots':sum(freq.values()),
                           'selected_layer_H_histogram':[selected_h[h] for h in range(7)],
                           'empty_layers':[l for l in range(37) if l not in selectedlayers],
                           'all_ranked_7779_exact_Fraction_entries_match':True,'per_case':[dict((k,v) for k,v in c.items() if k!='per_layer') for c in percase],
                           'admitted':None,'quality_accepted':None,'saved_time':None}

    restorations=[]
    for path in [w/'RESTORED_BEFORE_PILOT.json',w/'RESTORED_RUNTIME.json']:
        restored=obj(path)
        eq(restored['config_sha256'],CONFIG,'restore config pin')
        eq({Path(p).name:h for p,h in restored['mapped_runtime_hashes'].items()},NORMAL_HASHES,'restore four runtime pins')
        eq(restored['API_health'],'ok','restore health receipt')
        check(restored['normal_runtime_backing_files_verified'] and restored['normal_API_restored'],'restore verification receipt')
        check(not restored['diagnostic_libraries_loaded'],'restore no diagnostic receipt')
        eq(set(restored['services'].values()),{'active'},'restore active services')
        eq([restored['monitor_errors'],restored['active_alerts']],[{},{}],'restore no monitor errors')
        check(0<=restored['unix']-restored['monitor_timestamp']<=45,'restore recent monitor')
        restorations.append(restored)
    check(restorations[0]['unix']<frozen['unix']<restorations[1]['unix'],'template chronology between restores')
    result['restore_recorded_evidence']=restorations
    result['limits']=[
        'No live engine or API verification here. Hashes prove shared consistency, not independent attestation of remote execution.',
        'Engine ELF backing files are not in this handoff; runtime library hashes are recorded provenance checked against source/launch receipts, not rehashed actual ELF here.',
        'M1 is dummy vision graph reserve, not real-image compute. CUDA/NVML queries are sequential. M3 and saved health tails are not whole-run peaks.',
        'Capacity32 initializes a dormant native bank; model inference does not dispatch to it. Candidate35 covers 23 layers and has not been allocated/admitted.',
        '24 warmup final readback cells were recorded as zero after 3 compute calls each; intermediate 72 outputs are not separately available.',
        'All saved 5-row F32 arrays independently checked. Remaining 60 pilot T1 full outputs only have source plus runtime finite-scan receipt.',
        'Tokenizer and strict prefix use source plus success receipt; no independent model tokenizer run and no raw API token-piece responses.',
        'Candidate uses only 5 training variants in 3 families; H is training-fit geometry, not held-out utility or saved time.',
        'Pilot generation changes are at position256; long1024 control changes occur at positions1025 and1027. Linkage remains intact; metadata alone does not establish the causal trigger.',
        'Teacher-forced reference windows and finite/equal saved logits do not establish sampled-answer quality or API streaming cadence.'
    ]
    result['checks_by_category']=dict(sorted(CHECKS.items()))
    result['check_count']=sum(CHECKS.values())
    result['passed']=True
    out.mkdir(parents=True,exist_ok=True)
    (out/'GPU_WORKSPACE_EVIDENCE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    source={str(Path(k).relative_to(repo)) if Path(k).is_relative_to(repo) else k:v for k,v in sorted(SOURCE.items())}
    (out/'SOURCE_SHA256.json').write_text(json.dumps({'review_pin':PIN,'files':source,'audit_script_sha256':sha_bytes(Path(__file__).read_bytes())},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':result['check_count'],'sources':len(source),'candidate_hits':hit,'H':hist,'out':str(out)},ensure_ascii=False))


if __name__ == '__main__':
    main()
