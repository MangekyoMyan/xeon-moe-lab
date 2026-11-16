"""One aggregate snapshot. No per-request or chronological routing records."""
from pathlib import Path
import argparse
import json
import os
import struct


def read(path):
    with Path(path).open('rb', buffering=0) as stream:
        head = stream.read(64)
        magic, version, stride, capacity, pid, failure, used, dropped = struct.unpack_from('<8sIIIIQQQ', head)
        assert magic == b'XMCENS01' and version == 1 and stride == 4224 and capacity == 128
        assert used <= capacity and dropped == 0 and failure == 0
        rows = []
        for i in range(used):
            for attempt in range(10):
                stream.seek(64+i*stride)
                first = struct.unpack('<Q', stream.read(8))[0]
                stream.seek(64+i*stride)
                raw = stream.read(stride)
                before = struct.unpack_from('<Q', raw)[0]
                stream.seek(64+i*stride)
                after = struct.unpack('<Q', stream.read(8))[0]
                if first == before == after and before % 2 == 0:
                    break
            else:
                raise RuntimeError('Counter busy; take snapshot while API idle')
            name, typ, experts, k, m, calls, assignments = struct.unpack_from('<64siIqqQQ', raw, 8)
            assert 0 < experts <= 512
            counts = list(struct.unpack_from('<'+'Q'*experts, raw, 112))
            assert sum(counts) == assignments
            rows.append({'name': name.split(b'\0', 1)[0].decode('ascii'), 'type': typ,
                         'experts': experts, 'k': k, 'rows': m, 'calls': calls,
                         'assignments': assignments, 'counts': counts})
    return {'version': 1, 'pid': pid, 'scope': 'One model lifetime per process; snapshot only while API idle. Aggregated CPU single-token expert assignments include possible prefill tails. No token IDs, text, requests or execution order.', 'records': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = read(args.input)
    fd = os.open(args.output, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({'records': len(result['records']), 'saved': str(args.output)}))
