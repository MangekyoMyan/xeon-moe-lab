#!/usr/bin/env python3
"""Deliver local incidents to the existing Codex thread, never poll an LLM."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from monitor import atomic, read_json

HERE = Path(__file__).resolve().parent

def daemon_available(path):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(str(path))
        return True
    except OSError:
        return False

def format_message(events, latest, root):
    rows = []
    for event in events:
        stamp = datetime.datetime.fromtimestamp(event['unix'], datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
        for item in event['items']:
            value = item.get('value')
            limit = item.get('limit')
            rows.append(f"{stamp}: {item['description']}={value}" + (f" / 上限{limit}" if limit is not None else ''))
    test = all(e['kind'] == 'selftest' for e in events)
    return ('[XEON-WATCH ' + ('SELFTEST' if test else 'ALERT') + ']\n' +
            ('通知機能の自己テストです。実際のハードウェア異常はありません。受信確認のみお願いします。\n' if test else '') +
            '\n'.join(rows[:16]) + (f'\nほか{len(rows)-16}件（ログ参照）' if len(rows) > 16 else '') +
            f"\n現在の継続アラーム: {','.join(latest.get('active_alerts', {})) or 'なし'}" +
            f"\nセンサー取得失敗: {','.join(latest.get('errors', {})) or 'なし'}" +
            f"\n最新記録: {root / 'latest.json'}\nイベントID: " + ','.join(e['id'] for e in events) +
            '\n通常監視は常駐ソフトが継続します。必要なら記録を一度確認して簡潔に報告してください。' +
            'ユーザー方針: 長い起動・ダウンロードはユーザーが実行。比較テストの再開や自動再起動は指示されていません。')

def deliver(config, message):
    # The daemon must already be reachable. No automatic daemon launch while offline.
    if not daemon_available(config['codex_socket']):
        return {'accepted': False, 'reason': 'Codex daemon unavailable; retained locally'}
    cmd = [config['codex'], 'queue', '--thread', config['thread_id'], '--message', message]
    try:
        p = subprocess.run(cmd, cwd=config['workspace'], capture_output=True, text=True, timeout=25)
    except (OSError, subprocess.SubprocessError) as e:
        return {'accepted': False, 'reason': type(e).__name__}
    match = re.search(r'Queued message ([0-9a-f-]+) for thread ([0-9a-f-]+)', p.stdout)
    accepted = p.returncode == 0 and match is not None and match.group(2) == config['thread_id']
    return {'accepted': accepted, 'exit_code': p.returncode,
            'queue_id': match.group(1) if accepted else None,
            'reason': 'accepted by existing thread queue' if accepted else 'No successful queue receipt'}

def cycle(config, root, output, state, sender=deliver, now=None):
    now = time.time() if now is None else now
    if now < state.get('next_attempt', 0):
        return False
    pending = []
    acked = state.setdefault('acked', {})
    for path in (root / 'outbox').glob('*.json'):
        if path.stem in acked:
            continue
        pending.append(read_json(path, None))
    latest = read_json(root / 'latest.json', {})
    state.setdefault('started', now)
    stamp = latest.get('unix', state['started'])
    if now - stamp > 120 and now - state.get('process_started', state['started']) >= 120:
        event_id = 'watchdog-' + str(int(stamp))
        if event_id not in acked:
            pending.append({'id': event_id, 'unix': now, 'kind': 'monitor_stale',
                            'items': [{'description': '温度監視記録が120秒以上更新されていません', 'value': round(now - stamp)}]})
    pending.sort(key=lambda e: (e['unix'], e['id']))
    if not pending:
        return False
    # Collapse backlog into one message; details remain in the outbox and local journal.
    message = format_message(pending[-20:], latest, root)
    if len(pending) > 20:
        message += f'\n未通知イベントは計{len(pending)}件。直近20件をまとめて通知しました。'
    receipt = sender(config, message)
    receipt.update(unix=now, event_ids=[e['id'] for e in pending])
    atomic(output / 'last-delivery.json', receipt)
    if receipt['accepted']:
        for event in pending:
            acked[event['id']] = {'unix': now, 'queue_id': receipt.get('queue_id')}
        state['next_attempt'] = now + config['notify_min_gap_seconds']
        print(json.dumps(receipt), flush=True)
    else:
        state['next_attempt'] = now + config['notify_retry_seconds']
    atomic(output / 'state.json', state)
    return receipt['accepted']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=HERE / 'config.json')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    config = read_json(args.config, None)
    root = Path(config['state_dir'])
    output = Path(config['notify_state_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(output / 'state.json', {'acked': {}, 'next_attempt': 0})
        state['process_started'] = time.time()
        stop = False
        def stopped(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGINT, stopped)
        signal.signal(signal.SIGTERM, stopped)
        while not stop:
            cycle(config, root, output, state)
            if args.once:
                break
            for _ in range(20):
                if stop:
                    break
                time.sleep(0.5)

if __name__ == '__main__':
    main()
