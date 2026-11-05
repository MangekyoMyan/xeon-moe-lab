#!/usr/bin/env python3
"""User-run profile switch; model startup is intentionally never run by the assistant."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

R=Path(__file__).resolve().parent;LAB=R.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
CONFIG=ENGINE/'server-config.json';UNIT='llm-deepseek41-api.service'
def sha(data):return hashlib.sha256(data).hexdigest()
def write_config(data,stat):
    temporary=CONFIG.with_name(CONFIG.name+'.cpu-profile.tmp')
    with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.chmod(temporary,stat.st_mode&0o777);os.chown(temporary,stat.st_uid,stat.st_gid);temporary.replace(CONFIG)

def wait_for_gpu_release():
    stopped=time.time();deadline=time.monotonic()+30
    while True:
        snapshot=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
        if snapshot['unix']>=stopped and snapshot['gpu']['used_MiB']<=128:return
        if time.monotonic()>=deadline:
            raise SystemExit('API停止後のGPU解放を確認できません。設定は変更していません。status.pyで確認してください。')
        time.sleep(1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('profile',choices=['stock','pair','hybrid','stock-hist']);ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    manifest=json.loads((R/'RUNTIME.json').read_text());baseline=(R/'baseline-server-config.json').read_bytes()
    assert sha(baseline)==manifest['baseline_config_sha256']
    history_path=R/'profile-history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
    current=CONFIG.read_bytes();allowed={manifest['baseline_config_sha256']}|{x['config_sha256'] for x in history}
    if sha(current) not in allowed:raise SystemExit('起動設定が別途変更されています。上書きせず、設定の再確認が必要です。')
    desired=json.loads(baseline);capture=None
    if a.profile!='stock':
        profile=manifest['profiles'][a.profile]
        if sha(Path(profile['library']).read_bytes())!=profile['sha256']:raise SystemExit('CPU library hash mismatch')
        desired['environment']['LD_LIBRARY_PATH']=profile['directory']+':'+desired['environment']['LD_LIBRARY_PATH']
        desired['cpu_profile']={'name':a.profile,'CPU_library_sha256':profile['sha256'],'full_model_validation':'pending'}
    print('選択:',a.profile,'/ context・GPU・電力・ファン・API設定は元の値を使用')
    if not a.apply:
        print('確認のみ。--applyで既存APIを停止し、この構成で起動します。');return
    if os.geteuid()!=0:raise SystemExit('sudoで実行してください。')
    if subprocess.run(['systemctl','is-active','--quiet','llm-openai-api.service']).returncode==0:
        raise SystemExit('別のAPIが稼働しているため切替処理を中止しました。そのAPIは停止していません。')
    owner=LAB.stat()
    if a.profile=='stock-hist':
        capture=R/'captures'/time.strftime('%Y%m%d-%H%M%S')
        capture.mkdir(parents=True,exist_ok=False);os.chown(capture,owner.st_uid,owner.st_gid);capture.chmod(0o750)
        desired['environment']['GGML_MOE_HIST_FILE']=str(capture/'histogram.jsonl')
        desired['environment']['GGML_MOE_HIST_MAX_OPS']='20000'
    data=baseline if a.profile=='stock' else (json.dumps(desired,indent=2)+'\n').encode()
    subprocess.run(['systemctl','stop',UNIT],check=True,timeout=75)
    # start.py checks the persistent monitor snapshot. Wait for a new sample
    # after shutdown, otherwise the old API's VRAM allocation can reject startup.
    wait_for_gpu_release()
    st=CONFIG.stat();write_config(data,st)
    try:
        subprocess.run(['python3',str(ENGINE/'start.py')],check=True,timeout=60)
    except BaseException:
        write_config(current,st)
        print('起動処理が失敗したため設定を戻しました。API状態は既存status.pyで確認してください。')
        raise
    entry={'unix':time.time(),'profile':a.profile,'config_sha256':sha(data),'previous_config_sha256':sha(current),
           'capture':str(capture) if capture else None}
    history.append(entry);history_path.write_text(json.dumps(history,indent=2)+'\n')
    if capture is not None:
        with (capture/'ready-watcher.log').open('w') as log:
            subprocess.Popen(['python3',str(R/'capture_ready.py'),'--config',str(CONFIG),'--capture',str(capture)],
              stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True,user=owner.st_uid,group=owner.st_gid)
        print('記録先:',capture)
        print('READY.jsonのready:trueを確認してから普段の入力を送ってください。本文・token IDは記録しません。')
    print('起動を開始しました。進捗は既存のダッシュボードで確認できます。')

if __name__=='__main__':main()
