"""Recovery checks for the single investigated 2026-11-04 incident."""
import json
from pathlib import Path

FIELDS = ['boot_id', 'started_unix', 'finished_unix', 'status', 'error', 'child_returncode', 'pid']


def verify_incident(health, marker, reference, boot_id):
    if not (reference.get('status') == 'stopped_on_error' and reference.get('child_returncode') == -15 and
            str(reference.get('error', '')).startswith("TimeoutExpired(['nvidia-smi', '-i',") and
            str(reference['error']).endswith(', 5)')):
        raise RuntimeError('Reference is not the investigated GPU timeout')
    if any(health.get(k) != reference.get(k) for k in FIELDS):
        raise RuntimeError('Failure record changed; refusing to acknowledge another incident')
    if boot_id != reference['boot_id']:
        raise RuntimeError('Boot changed since incident; inspect current hardware before recovery')
    if marker != (reference['error']+'\n').encode():
        raise RuntimeError('Stop marker changed; refusing recovery')


def reviewed_health(health, archive, now):
    result = dict(health)
    result['status'] = 'stop_reviewed'
    result['reviewed_error'] = result.pop('error')
    result['reviewed_unix'] = now
    result['recovery_archive'] = str(archive)
    result['review_note'] = 'Investigated single GPU-query timeout; bounded retry installed. Model not restarted.'
    return result
