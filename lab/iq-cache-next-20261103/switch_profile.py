#!/usr/bin/env python3
"""User-run configuration switch with hash checks and exact config rollback."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time

R=Path(__file__).resolve().parent;ENGINE=R.parent/'deepseek41-bringup-20261101'
CONFIG=ENGINE/'server-config.json';UNIT='llm-deepseek41-api.service'
WATCH=Path('/var/lib/xeon-watch/latest.json')
def sha(data):return hashlib.sha256(data).hexdigest()

def write_config(data):
    stat=CONFIG.stat();fd,name=tempfile.mkstemp(prefix='.iq3-profile-',dir=CONFIG.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
            os.fchmod(f.fileno(),stat.st_mode&0o777);os.fchown(f.fileno(),stat.st_uid,stat.st_gid)
        os.replace(name,CONFIG)
    finally:
        if os.path.exists(name):os.unlink(name)

def wait_release(stopped):
    deadline=time.monotonic()+30
    while True:
        j=json.loads(WATCH.read_text())
        if j['unix']>=stopped and j['gpu']['used_MiB']<=128:return
        if time.monotonic()>deadline:raise RuntimeError('GPU解放の確認ができません。設定は変更していません。')
        time.sleep(1)

def apply(current,desired):
    if subprocess.run(['systemctl','is-active','--quiet','llm-openai-api.service']).returncode==0:
        raise RuntimeError('別のAPIが稼働中のため切替を中止しました。')
    if (ENGINE/'STOP_REQUESTED').exists():raise RuntimeError('前回の保護停止を確認してください。解除はしません。')
    j=json.loads(WATCH.read_text())
    if time.time()-j['unix']>45 or j['errors'] or j['active_alerts']:raise RuntimeError('監視が古いか、異常があります。')
    if j['gpu']['ecc_mode']!='Enabled' or abs(j['gpu']['power_limit_W']-150)>.5:raise RuntimeError('GPU150W/ECC設定を確認してください。')
    if CONFIG.read_bytes()!=current:raise RuntimeError('本番設定が変更されました。')
    subprocess.run(['systemctl','stop',UNIT],check=True,timeout=75)
    wait_release(time.time())
    if CONFIG.read_bytes()!=current:raise RuntimeError('停止確認中に本番設定が変更されました。上書きしません。')
    write_config(desired)
    try:subprocess.run(['python3',str(ENGINE/'start.py')],check=True,timeout=60)
    except BaseException:
        inactive=subprocess.run(['systemctl','is-active','--quiet',UNIT]).returncode!=0
        if inactive and CONFIG.read_bytes()==desired:
            write_config(current)
            print('APIが非稼働であることを確認し、設定を戻しました。')
        else:print('API稼働中または設定変更があるため復元せず終了します。status.pyで確認してください。')
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('profile',choices=['previous','multi64','multi96'])
    ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    manifest=json.loads((R/'PROFILES.json').read_text());profile=manifest['profiles'][a.profile]
    desired=Path(profile['file']).read_bytes();assert sha(desired)==profile['sha256'],'Profile hash changed'
    current=CONFIG.read_bytes()
    accepted={manifest['original_config_sha256']}|{v['sha256'] for v in manifest['profiles'].values()}
    if sha(current) not in accepted:raise SystemExit('別途変更された本番設定です。上書きせず再確認が必要です。')
    if a.profile!='previous':
        assert sha((R/'VALIDATION.json').read_bytes())==manifest['validation_sha256'],'Validation record changed'
        v=json.loads((R/'VALIDATION.json').read_text())
        assert v['passed'] and v['build_sha256']==sha((R/'BUILD.json').read_bytes()),'Validation is missing or stale'
    for library in manifest['libraries']:
        if a.profile!='previous' or '/iq-cache-next-20261103/runtime/' not in library['file']:
            assert sha(Path(library['file']).read_bytes())==library['sha256'],'Overlay hash changed: '+library['file']
    print(f"選択: {a.profile} / GPU expert 3層 / batch 1024 / 128k f16 KV / GPU150W・ECC")
    if a.profile!="previous":print(f"CPU IQ追加配置: 上限{64 if a.profile=='multi64' else 96}GiB RAM、3形式対応。起動時に一度変換します。")
    if not a.apply:
        print('確認のみ。--applyでAPIを停止して設定を切り替え、モデルの起動を要求します。');return
    if os.geteuid()!=0:raise SystemExit('sudoで実行してください。')
    with (R/'.profile.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if CONFIG.read_bytes()!=current:raise SystemExit('確認中に設定が変更されました。')
        apply(current,desired)
    print('起動要求を送りました。大きなモデルの読込み・メモリ登録には時間がかかります。')
    print('表示: sudo python3 /home/codex/projects/llm-lab/xeon_dashboard.py')
    print('戻す: sudo python3 '+str(R/'switch_profile.py')+' previous --apply')

if __name__=='__main__':main()
