#!/usr/bin/env python3
"""Own bounded counterexample to two over-specific grading requirements.

This is not model-generated code and does not execute supplied reference code,
model inference, APIs, GPU work, or any engine. A reverse-direction ring with
None on successful push satisfies the prompt's observable contract, even though
the old rubric demanded True and a forward tail formula.
"""
from collections import deque
from itertools import product
from pathlib import Path
import hashlib
import json


class ReverseRingQueue:
    def __init__(self, capacity):
        if not 1 <= capacity <= 1024:
            raise ValueError('capacity')
        self._a = [0] * capacity
        self._head = 0
        self._count = 0

    def size(self):
        return self._count

    def try_push(self, value):
        if self._count == len(self._a):
            return False
        self._a[(self._head - self._count) % len(self._a)] = value
        self._count += 1
        return None  # Successful push return is unspecified by the prompt.

    def try_pop(self):
        if not self._count:
            return None
        value = self._a[self._head]
        self._head = (self._head - 1) % len(self._a)
        self._count -= 1
        return value


def main():
    sequences = transitions = 0
    for capacity in (1, 2, 3):
        for ops in product(('pop', -1, 0, 1), repeat=8):
            q, expected = ReverseRingQueue(capacity), deque()
            for op in ops:
                if op == 'pop':
                    wanted = expected.popleft() if expected else None
                    assert q.try_pop() == wanted
                elif len(expected) == capacity:
                    assert q.try_push(op) is False
                else:
                    result = q.try_push(op)
                    assert result is None
                    expected.append(op)
                assert q.size() == len(expected)
                transitions += 1
            while expected:
                assert q.try_pop() == expected.popleft()
            assert q.size() == 0 and q.try_pop() is None
            sequences += 1
    rejected = (-(1 << 63), -1, 0, 1025, (1 << 63) - 1)
    for n in rejected:
        try:
            ReverseRingQueue(n)
        except ValueError:
            continue
        raise AssertionError('invalid capacity accepted')
    q = ReverseRingQueue(1024)
    values = [-(1 << 63), (1 << 63) - 1, 0] + list(range(1021))
    for value in values:
        assert q.try_push(value) is None
    assert q.try_push(99) is False and q.size() == 1024
    assert [q.try_pop() for _ in values] == values
    for i in range(4097):
        assert q.try_push(i) is None
        assert q.try_pop() == i
    result = {
        'status': 'passed',
        'purpose': 'Demonstrate that the prior successful-True and forward-tail grading constraints reject a valid alternative.',
        'code_origin': 'reviewer-written counterexample; not a model answer or supplied reference',
        'exhaustive_sequence_length': 8,
        'exhaustive_capacities': [1, 2, 3],
        'alphabet': ['pop', 'push(-1)', 'push(0)', 'push(1)'],
        'sequences_checked': sequences,
        'operation_transitions_checked': transitions,
        'additional_boundary_capacity': 1024,
        'additional_wrap_pairs': 4097,
        'invalid_capacities_checked': list(rejected),
        'successful_push_returns': 'None',
        'physical_direction': 'decreasing index modulo capacity',
        'oracle': 'independent collections.deque with explicit capacity rejection',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_or_engine_or_GPU_executed': False,
        'provided_reference_executed': False,
        'limits': 'A grading counterexample and bounded checks, not a model-quality result or a proof over arbitrary programs.'
    }
    Path(__file__).with_name('GRADING_COUNTEREXAMPLE_VERIFIED.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
