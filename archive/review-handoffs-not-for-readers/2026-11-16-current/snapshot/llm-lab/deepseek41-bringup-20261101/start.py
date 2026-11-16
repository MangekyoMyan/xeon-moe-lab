#!/usr/bin/env python3
"""User-run launch: validates environment, installs one unit, starts asynchronously."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

R=Path(__file__).resolve().parent;LAB=R.parent


def check_api_bind(host, port):
    # Match llama-server's socket options: allow TIME_WAIT reuse, not shared listeners.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)


def main():
    if os.geteuid()!=0:raise SystemExit('実行: sudo python3 '+str(Path(__file__).resolve()))
    ready=R/'BUILD_READY.json'
    if not ready.exists():raise SystemExit('先に python3 '+str(R/'build.py')+' を実行してください。')
    j=json.loads(ready.read_text());binary=R/'build/bin/llama-server'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=j['binary_sha256']:raise SystemExit('ビルド後にバイナリが変わっています。')
    config=json.loads((R/'server-config.json').read_text())
    if (R/'STOP_REQUESTED').exists():raise SystemExit('前回の保護停止を確認してください: '+str(R/'STOP_REQUESTED'))
    for name in ['llm-openai-api.service','llm-deepseek41-api.service']:
        if subprocess.run(['systemctl','is-active','--quiet',name]).returncode==0:raise SystemExit(name+' が稼働中です。自動停止しません。')
    for name in ['xeon-watch.service','xeon-watch-notify.service']:
        if subprocess.run(['systemctl','is-active','--quiet',name]).returncode:raise SystemExit('監視が停止しています: '+name)
    s=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    if time.time()-s['unix']>45 or s['errors'] or s['active_alerts']:raise SystemExit('監視記録が古いか、異常があります。status.pyで確認してください。')
    if s['gpu']['ecc_mode']!='Enabled' or abs(s['gpu']['power_limit_W']-150)>.5:raise SystemExit('GPU150W/ECC有効を確認してください。設定は自動変更しません。')
    if s['gpu']['used_MiB']>128:raise SystemExit('GPUが使用中です。')
    mem={x.split(':',1)[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:','Mlocked:'))}
    if mem['MemTotal']<480*1024**3 or mem['Mlocked']>32*1024**3:raise SystemExit('512GB構成と他のメモリ固定処理を確認してください。')
    if mem['MemAvailable']<360*1024**3:raise SystemExit('利用可能RAMが360GiB未満です。大きな別処理を終了してから再実行してください。')
    model_root=Path(config['model']).parent.parent
    manifest=json.loads((model_root/'download-verified.json').read_text())
    for item in manifest['files']:
        if (model_root/item['name']).stat().st_size!=item['size']:raise SystemExit('モデルのサイズ不一致: '+item['name'])
    if not Path(config['api_key_file']).exists():raise SystemExit('既存APIキーのファイルが見つかりません。')
    try:check_api_bind(config['host'],config['port'])
    except OSError as e:raise SystemExit('APIのIP/ポートを使用できません: '+str(e))
    subprocess.run(['install','-m','0644',str(R/'llm-deepseek41-api.service'),'/etc/systemd/system/llm-deepseek41-api.service'],check=True)
    subprocess.run(['systemctl','daemon-reload'],check=True)
    subprocess.run(['systemctl','start','llm-deepseek41-api.service'],check=True)
    print('起動要求を送りました。HDDからの初回読込には時間がかかります。自動起動設定はしていません。')
    print('確認: sudo python3 '+str(R/'status.py'))
    print('ログ: sudo tail -f '+str(R/'server.stderr.log'))
    print('停止: sudo systemctl stop llm-deepseek41-api.service')

if __name__=='__main__':main()
