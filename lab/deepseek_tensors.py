#!/usr/bin/env python3
"""Summarize stored weight formats and approximate active weight bytes per token."""
import collections
import ctypes
import json
import math
from pathlib import Path
from index_model_cache import unpack, string, value

LAB = Path(__file__).resolve().parent
lib = ctypes.CDLL('/opt/llama.cpp/build/bin/libggml-base.so')
lib.ggml_type_name.restype = ctypes.c_char_p
lib.ggml_type_size.restype = ctypes.c_size_t
lib.ggml_blck_size.restype = ctypes.c_int64


def tensors(path):
    with path.open('rb') as stream:
        assert stream.read(4) == b'GGUF'
        assert unpack(stream, 'I') in (2, 3)
        nt, nk = unpack(stream, 'Q'), unpack(stream, 'Q')
        for _ in range(nk):
            string(stream, False)
            value(stream, unpack(stream, 'I'), False)
        rows = []
        for _ in range(nt):
            name = string(stream)
            dims = [unpack(stream, 'Q') for _ in range(unpack(stream, 'I'))]
            kind, offset = unpack(stream, 'I'), unpack(stream, 'Q')
            params = math.prod(dims)
            size = params * lib.ggml_type_size(kind) // lib.ggml_blck_size(kind)
            rows.append({'name': name, 'dims': dims, 'type': lib.ggml_type_name(kind).decode(),
                         'parameters': params, 'bytes': size, 'offset': offset})
        return rows


def main():
    report = {'method': 'One read of each active weight; excludes KV, activations, cache reuse, prefetch and alignment amplification. Expert tensors use 6/256 rows; embedding and token lookup use one row. MTP excluded only if nextn tensors are present. This is not measured DRAM traffic.', 'models': {}}
    for quant in ['IQ2_XXS-XL', 'Q2_K-XL']:
        rows = [r for p in sorted((LAB / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF' / quant).glob('*.gguf')) for r in tensors(p)]
        mtp_present = any(r['name'] == 'blk.42.nextn.eh_proj.weight' for r in rows)
        by_type = collections.defaultdict(lambda: {'storage_bytes': 0, 'active_weight_bytes_estimate': 0, 'parameters': 0})
        active_params = 0
        for r in rows:
            fraction = 1
            if mtp_present and r['name'].startswith('blk.42.'):
                fraction = 0
            elif '_exps.' in r['name']:
                assert r['dims'][-1] == 256
                fraction = 6 / 256
            elif r['name'] == 'token_embd.weight' or 'tid2eid' in r['name']:
                fraction = 1 / r['dims'][-1]
            r['active_fraction_estimate'] = fraction
            dst = by_type[r['type']]
            dst['storage_bytes'] += r['bytes']
            dst['parameters'] += r['parameters']
            dst['active_weight_bytes_estimate'] += r['bytes'] * fraction
            active_params += r['parameters'] * fraction
        report['models'][quant] = {'mtp_present': mtp_present, 'by_type': dict(by_type),
                                  'active_parameters_estimate': active_params,
                                  'active_weight_bytes_estimate': sum(v['active_weight_bytes_estimate'] for v in by_type.values()),
                                  'tensors': rows}
    out = LAB / 'results/deepseek-tensor-inventory.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({q: {k: v for k, v in m.items() if k != 'tensors'} for q, m in report['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
