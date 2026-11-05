#!/usr/bin/env python3
"""User-run vision switch. No service/configuration changes without --apply."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
ENGINE = LAB / 'deepseek41-bringup-20261101'
CONFIG = ENGINE / 'server-config.json'
UNIT = 'llm-deepseek41-api.service'
WATCH = Path('/var/lib/xeon-watch/latest.json')
MIB = 1024 ** 2


def sha(data):
    return hashlib.sha256(data).hexdigest()


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def active():
    return subprocess.run(['systemctl', 'is-active', '--quiet', UNIT]).returncode == 0


def healthy():
    for p in [ENGINE/'STOP_REQUESTED', LAB/'moe-engine-20261029/STOP_REQUESTED']:
        if p.exists():
            raise RuntimeError('保護停止記録があります。解除せず停止します: ' + str(p))
    for unit in ['xeon-watch.service', 'xeon-watch-notify.service']:
        if subprocess.run(['systemctl', 'is-active', '--quiet', unit]).returncode:
            raise RuntimeError('監視サービスが停止しています。')
    state = json.loads(WATCH.read_text())
    if not 0 <= time.time() - state['unix'] <= 45 or state['errors'] or state['active_alerts']:
        raise RuntimeError('監視に異常があるか、記録が古くなっています。')
    if state['gpu']['ecc_mode'] != 'Enabled' or abs(state['gpu']['power_limit_W'] - 150) > .5:
        raise RuntimeError('GPU150W / ECC設定を確認してください。')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('profile', choices=['vision', 'text'])
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((ROOT/'PROFILES.json').read_text())
    profiles = manifest['profiles']
    current = CONFIG.read_bytes()
    if sha(current) not in {p['sha256'] for p in profiles.values()}:
        raise RuntimeError('本番設定が準備時と異なります。上書きせず停止します。')
    target = (ROOT/profiles[args.profile]['file']).read_bytes()
    if sha(target) != profiles[args.profile]['sha256']:
        raise RuntimeError('候補設定のハッシュが異なります。')
    for item in manifest['tools']:
        if sha(Path(item['file']).read_bytes()) != item['sha256']:
            raise RuntimeError('起動ヘルパーまたは計測プログラムが準備時と異なります。')
    projector = Path(manifest['projector']['file'])
    if args.profile == 'vision' and projector.stat().st_size != manifest['projector']['size']:
        raise RuntimeError('Visionファイルの容量が一致しません。')
    print(f"選択: {args.profile} / 会話128k / CPUキャッシュ96GiB / GPU150W・ECC")
    if args.profile == 'vision':
        print('F16 VisionをGPUへ追加。初期設定は画像ごと256トークン、画像処理バッチ256。')
    if not args.apply:
        print('確認のみ。--apply でモデルAPIを停止し、設定を反映して起動を要求します。')
        return
    if os.geteuid() != 0:
        raise RuntimeError('反映には sudo python3 を使用してください。')
    with (ROOT/'.switch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if current == target and active():
            print('既にこの設定で起動しています。変更しません。')
            return
        healthy()
        if subprocess.run(['systemctl', 'is-active', '--quiet', 'llm-openai-api.service']).returncode == 0:
            raise RuntimeError('別のAPIが稼働しています。')
        idle = helper('vision_idle_check', LAB/'expert-validation-20261101/switch_profile.py')
        switch = helper('vision_config_io', LAB/'iq-cache-next-20261103/switch_profile.py')
        if active():
            idle.ensure_idle(json.loads(current))
        if args.profile == 'vision':
            print('VisionファイルのSHA256を確認中…', flush=True)
            with projector.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != manifest['projector']['sha256']:
                raise RuntimeError('VisionファイルのSHA256不一致。APIは停止していません。')
        healthy()
        if CONFIG.read_bytes() != current:
            raise RuntimeError('確認中に本番設定が変わりました。')
        history = ROOT/'history'/str(time.time_ns())
        history.mkdir(parents=True, mode=0o700)
        (history/'before.json').write_bytes(current)
        (history/'requested.json').write_bytes(target)
        print('モデルAPIを停止します。OSの再起動は行いません。', flush=True)
        subprocess.run(['systemctl', 'stop', UNIT], check=True, timeout=75)
        switch.wait_release(time.time())
        if active():
            raise RuntimeError('APIの停止を確認できません。')
        if args.profile == 'vision':
            print('VisionのGPUメモリ予約を確認中（本体モデルはまだ読み込みません）…', flush=True)
            env = os.environ.copy()
            env.update(json.loads(target)['environment'])
            probe = subprocess.run([str(ROOT/'memory_probe'), str(projector), 'gpu'],
                                   env=env, capture_output=True, text=True, timeout=45)
            (history/'memory-probe.stdout').write_text(probe.stdout)
            (history/'memory-probe.stderr').write_text(probe.stderr)
            if probe.returncode:
                raise RuntimeError('Visionの予約確認に失敗しました。設定は変更していません。ログ: ' + str(history))
            usage = json.loads(probe.stdout)
            expected = manifest['text_gpu_observed_MiB'] * MIB + usage['gpu_bytes'] + 768 * MIB
            if usage['image_max_tokens'] != 256 or usage['gpu_bytes'] <= 0 or expected > usage['gpu_total_bytes']:
                raise RuntimeError('Vision追加後のVRAM余裕を確認できません。設定を変えず停止しました。')
            print(f"Vision予約: {usage['gpu_bytes']/MIB:.1f} MiB。余力768MiBを含む容量確認OK。", flush=True)
            # The monitor must observe the probe context's release before start.py.
            switch.wait_release(time.time())
        healthy()
        if CONFIG.read_bytes() != current:
            raise RuntimeError('停止中に設定が変わりました。上書きしません。')
        switch.write_config(target)
        try:
            subprocess.run(['python3', str(ENGINE/'start.py')], check=True, timeout=60)
        except BaseException:
            if not active() and CONFIG.read_bytes() == target:
                switch.write_config(current)
                print('起動要求が拒否されたため設定を元に戻しました。APIは停止中です。')
            raise
        print('起動を要求しました。モデルの読み込みはバックグラウンドで続きます。')
        print('表示: sudo python3 ' + str(LAB/'xeon_dashboard.py'))
        print('画像なしへ戻す: sudo python3 ' + str(Path(__file__).resolve()) + ' text --apply')


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
