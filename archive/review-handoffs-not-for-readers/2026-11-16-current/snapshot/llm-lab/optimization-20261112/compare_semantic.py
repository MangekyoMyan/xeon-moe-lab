"""Compare completed frozen functional tasks, retaining scorer-repair provenance."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def read(folder):
    raw = (folder / 'RESULT.json').read_bytes()
    result = json.loads(raw)
    assert result['complete']
    rows = {c['name']: dict(c) for c in result['cases']}
    assert len(rows) == len(result['cases'])
    repair_file = folder / 'RESCORE.json'
    if repair_file.exists():
        repair = json.loads(repair_file.read_text())
        assert repair['source_result_sha256'] == hashlib.sha256(raw).hexdigest()
        assert repair['scorer_sha256'] == result['scorer_sha256']
        assert repair['all_correct_after_rescore']
        for corrected in repair['cases']:
            row = rows[corrected['name']]
            assert row['score']['reason'] == 'Scoring process failed'
            assert row['output_sha256'] == corrected['original_output_sha256']
            assert corrected['uid_asserted'] == 65534 and corrected['complete_generation']
            row['score'] = corrected['score']
            row['correct'] = corrected['score']['correct'] and row['stop_type'] == 'eos'
    for row in rows.values():
        assert hashlib.sha256(row['content'].encode()).hexdigest() == row['output_sha256']
        assert row['timings']['cache_n'] == 0
        assert row['timings']['prompt_n'] == row['prompt_tokens']
    return result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    reference, ref_rows = read(args.reference)
    candidate, cand_rows = read(args.candidate)
    assert reference['inputs_sha256'] == candidate['inputs_sha256']
    assert reference['scorer_sha256'] == candidate['scorer_sha256']
    assert ref_rows.keys() == cand_rows.keys()
    for result in [reference, candidate]:
        profile = ROOT / 'profiles' / (result['label'] + '.json')
        assert hashlib.sha256(profile.read_bytes()).hexdigest() == result['config_sha256']
    compared = []
    for name, ref in ref_rows.items():
        cand = cand_rows[name]
        assert ref['prompt_tokens'] == cand['prompt_tokens']
        compared.append({'name': name, 'tokens': ref['prompt_tokens'],
                         'reference_correct': ref['correct'], 'candidate_correct': cand['correct'],
                         'same_output': ref['output_sha256'] == cand['output_sha256'],
                         'reference_ttft_s': ref['ttft_s'], 'candidate_ttft_s': cand['ttft_s'],
                         'ttft_reduction_percent': 100 * (1 - cand['ttft_s'] / ref['ttft_s'])})
    out = {'reference': str(args.reference), 'candidate': str(args.candidate), 'cases': compared,
           'all_correct_both': all(c['reference_correct'] and c['candidate_correct'] for c in compared),
           'scope': 'Five fixed functional tasks, not a broad quality-equivalence benchmark. One run per profile.'}
    (ROOT / 'SEMANTIC_COMPARISON.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
