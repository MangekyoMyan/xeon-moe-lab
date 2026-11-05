#!/usr/bin/env python3
"""User-run profile switch. Default is read-only; --apply restarts only the model API."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parent / 'deepseek41-bringup-20261101'
CONFIG = ENGINE / 'server-config.json'
SERVICE = 'llm-deepseek41-api.service'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_verified(profile):
    manifest = json.loads((ROOT/'profiles/manifest.json').read_text())
    current = CONFIG.read_bytes()
    known = {v['sha256'] for v in manifest['profiles'].values()}
    if digest(current) not in known:
        raise RuntimeError('現在の設定が準備時と異なります。上書きせず停止します。')
    entry = manifest['profiles'][profile]
    target = (ROOT/'profiles'/entry['file']).read_bytes()
    if digest(target) != entry['sha256']:
        raise RuntimeError('候補ファイルのハッシュが一致しません。')
    return current, target


def atomic_config(data):
    old = CONFIG.stat()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=CONFIG.parent, prefix='.server-profile-', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            os.fchmod(f.fileno(), stat.S_IMODE(old.st_mode))
            os.fchown(f.fileno(), old.st_uid, old.st_gid)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(CONFIG)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def active():
    return subprocess.run(['systemctl','is-active','--quiet',SERVICE]).returncode == 0


def ensure_idle(config):
    key = Path(config['api_key_file']).read_text().strip()
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(f"http://{config['host']}:{config['port']}/slots",
                                 headers={'Authorization':'Bearer '+key})
    with http.open(req, timeout=10) as response:
        slots = json.load(response)
    if any(s.get('is_processing') for s in slots):
        raise RuntimeError('APIが応答中です。モデルを止めずに終了します。')


def ensure_monitor():
    # Do not bypass the existing protection latch, including during rollback.
    if (ENGINE/'STOP_REQUESTED').exists() or (ROOT.parent/'moe-engine-20261029/STOP_REQUESTED').exists():
        raise RuntimeError('保護停止記録があります。原因確認が必要です。設定・サービスは変更しません。')
    state = json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    if time.time()-state['unix'] > 45 or state['errors'] or state['active_alerts']:
        raise RuntimeError('監視に異常または古い記録があります。設定・サービスは変更しません。')
    for unit in ['xeon-watch.service','xeon-watch-notify.service']:
        if subprocess.run(['systemctl','is-active','--quiet',unit]).returncode:
            raise RuntimeError('監視サービスが停止しています。モデルを停止せず終了します。')


def read_monitor():
    return json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())


def wait_for_monitor_release(stopped_unix):
    # Consume existing daemon records; do not run a separate hardware polling command.
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        state = read_monitor()
        if state['unix'] > stopped_unix:
            if state['errors'] or state['active_alerts']:
                raise RuntimeError('停止後の監視記録に異常があります。モデルは再起動していません。')
            if state['gpu']['used_MiB'] <= 128:
                return
        time.sleep(1)
    raise RuntimeError('停止後のGPU解放を監視記録で確認できません。モデルは停止中、設定は元のままです。')


def apply(current, target):
    ensure_monitor()
    if active():
        ensure_idle(json.loads(current))
    # Recheck immediately before the first mutation.
    if CONFIG.read_bytes() != current:
        raise RuntimeError('確認後に設定が変わりました。変更せず終了します。')
    backup = ROOT/'profile-history'/str(time.time_ns())
    backup.mkdir(parents=True, mode=0o700)
    (backup/'before.json').write_bytes(current)
    (backup/'requested.json').write_bytes(target)
    subprocess.run(['systemctl','stop',SERVICE],check=True,timeout=75)
    if active():
        raise RuntimeError('モデル停止を確認できません。設定は変更しません。')
    if CONFIG.read_bytes() != current:
        raise RuntimeError('停止中に設定が変更されました。上書きせず終了します。')
    wait_for_monitor_release(time.time())
    if CONFIG.read_bytes() != current:
        raise RuntimeError('GPU解放の確認中に設定が変更されました。上書きせず終了します。')
    try:
        atomic_config(target)
        subprocess.run(['python3',str(ENGINE/'start.py')],check=True,timeout=45)
    except BaseException as error:
        # Restore the old configuration if startup was rejected. Never retry a model start automatically.
        if not active() and CONFIG.read_bytes() == target:
            atomic_config(current)
            if isinstance(error,subprocess.SubprocessError):
                raise RuntimeError('起動要求に失敗したため、変更前の設定へ戻しました。APIは停止中です。上のエラー内容を確認してください。') from None
        raise
    print('起動を要求しました。モデル読込はバックグラウンドで進みます。自動ベンチマークは起動しません。')
    print('一度だけ状態を見る場合: sudo python3 '+str(ENGINE/'status.py'))
    print('以前のCPU配置に戻す場合: sudo python3 '+str(Path(__file__).resolve())+' baseline --apply')
    print('記録:',backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',choices=['gpu2','baseline'])
    parser.add_argument('--apply',action='store_true',help='stop the idle model API, switch config, and request a new start')
    args = parser.parse_args()
    current,target = load_verified(args.profile)
    print('選択:',args.profile,'context:128k / f16 KV / GPU150W / ECC / fan<=70%')
    if not args.apply:
        print('検証のみ。設定・サービスを変更していません。反映には --apply が必要です。')
        return
    if os.geteuid()!=0:
        raise SystemExit('反映は sudo python3 '+str(Path(__file__).resolve())+' '+args.profile+' --apply')
    with (ROOT/'.profile-switch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apply(current,target)


if __name__=='__main__':
    try:
        main()
    except (RuntimeError,OSError,subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
