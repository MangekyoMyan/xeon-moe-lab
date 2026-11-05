#!/usr/bin/env python3
"""User-run next-startup observer profile. This command never starts/stops an API."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,stat,subprocess,tempfile,time
R=Path(__file__).resolve().parent;ENGINE=R.parent/'deepseek41-bringup-20261101'
CONFIG=ENGINE/'server-config.json';PRIVATE=Path('/var/lib/xeon-expert-shadow');STATE=PRIVATE/'profile.json'
BASELINE_SHA='11ca8641aee16173e38283407182b6febc264915535df93dc7ce4dd840a2e44b'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def private_directory(create=False):
    if create:PRIVATE.mkdir(mode=0o700,exist_ok=True)
    if PRIVATE.exists() or PRIVATE.is_symlink():
        st=PRIVATE.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o700:raise RuntimeError('Private state directory must be root-owned0700 without symlink')
def atomic(path,raw,mode,uid,gid):
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            os.fchmod(f.fileno(),mode);os.fchown(f.fileno(),uid,gid);f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(name):os.unlink(name)
def inactive():
    for name in ['llm-deepseek41-api.service','llm-openai-api.service']:
        r=subprocess.run(['systemctl','show',name,'--property=ActiveState,MainPID'],check=True,capture_output=True,text=True)
        fields=dict(line.split('=',1) for line in r.stdout.splitlines())
        if fields.get('ActiveState') not in ['inactive','failed'] or fields.get('MainPID')!='0':raise RuntimeError('先にAPIを停止してください: '+name)
    uid=subprocess.run(['systemctl','show','llm-deepseek41-api.service','--property=User','--value'],check=True,capture_output=True,text=True).stdout.strip()
    if uid not in ['', 'root']:raise RuntimeError('Observer private directory expects the current root-run API service')
def verified_build():
    b=json.loads((R/'CAPTURE_BUILD.json').read_text())
    for path,key in [(Path(b['library']),'sha256'),(R/'shadow_capture.h','helper_sha256'),(R/'llama-context-shadow.cpp','source_sha256')]:
        if sha(path.read_bytes())!=b[key]:raise RuntimeError('Observer build/source changed: '+str(path))
    test=json.loads((R/'LOADER_CHECK.json').read_text())
    if not test.get('logits_bit_identical') or test.get('observations',0)<1 or test.get('observer_library_sha256')!=b['sha256']:raise RuntimeError('Actual callback validation missing for this build')
    return b
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['enable','restore']);p.add_argument('--apply',action='store_true');a=p.parse_args()
    print('次回起動の観測だけを設定します。CPU/GPU分担・電力・ファンは変更しません。APIの停止・起動は実行しません。')
    if a.apply and os.geteuid()!=0:raise SystemExit('適用には sudo python3 が必要です。')
    with (R/'.profile.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        private_directory()
        if CONFIG.is_symlink() or not CONFIG.is_file():raise RuntimeError('Expected regular production config')
        current=CONFIG.read_bytes();st=CONFIG.stat()
        if a.action=='enable':
            build=verified_build()
            if sha(current)!=BASELINE_SHA:raise RuntimeError('元の本番設定が変わっています。新設定との整合性を確認してから作り直してください。')
            if STATE.exists():raise RuntimeError('以前の観測設定の記録があります。先に restore を実行してください。')
            folder=PRIVATE/'captures';log=folder/('routes-'+str(time.time_ns())+'.jsonl')
            runtime=PRIVATE/('runtime-'+build['sha256'])
            config=json.loads(current);env=config['environment']
            if 'XEON_SHADOW_LOG' in env:raise RuntimeError('Existing observer environment')
            env['LD_LIBRARY_PATH']=str(runtime)+':'+env['LD_LIBRARY_PATH'];env['XEON_SHADOW_LOG']=str(log)
            target=(json.dumps(config,indent=2,ensure_ascii=False)+'\n').encode()
            print('観測先:',log,'/ 最大2048演算・8MiB、本文保存なし')
            print('1起動分の観測です。再記録はAPI停止後に restore → enable。既存ログへ追記しません。')
            if not a.apply:return
            inactive();private_directory(create=True);folder.mkdir(mode=0o700,exist_ok=True)
            if folder.is_symlink() or folder.stat().st_uid!=0 or stat.S_IMODE(folder.stat().st_mode)!=0o700:raise RuntimeError('Capture directory must be root-owned0700')
            library=Path(build['library']).read_bytes()
            if sha(library)!=build['sha256']:raise RuntimeError('Library changed before private copy')
            runtime.mkdir(mode=0o700,exist_ok=True)
            if runtime.is_symlink() or runtime.stat().st_uid!=0 or stat.S_IMODE(runtime.stat().st_mode)!=0o700:raise RuntimeError('Private runtime directory changed')
            target_library=runtime/'libllama.so.0.8.0'
            if target_library.exists():
                if target_library.is_symlink() or sha(target_library.read_bytes())!=build['sha256']:raise RuntimeError('Private library mismatch')
            else:atomic(target_library,library,0o444,0,0)
            for name in ['libllama.so.0','libllama.so']:
                link=runtime/name
                if link.exists() or link.is_symlink():
                    if not link.is_symlink() or os.readlink(link)!=target_library.name:raise RuntimeError('Private library alias mismatch')
                else:link.symlink_to(target_library.name)
            record={'original_utf8':current.decode(),'original_sha256':sha(current),'configured_sha256':sha(target),
                    'mode':stat.S_IMODE(st.st_mode),'uid':st.st_uid,'gid':st.st_gid,'capture':str(log),'library_sha256':build['sha256']}
            atomic(STATE,(json.dumps(record,indent=2)+'\n').encode(),0o600,0,0)
            if CONFIG.is_symlink() or CONFIG.read_bytes()!=current:raise RuntimeError('Production config changed before apply')
            atomic(CONFIG,target,stat.S_IMODE(st.st_mode),st.st_uid,st.st_gid)
        else:
            ss=STATE.lstat()
            if not stat.S_ISREG(ss.st_mode) or ss.st_uid!=0 or stat.S_IMODE(ss.st_mode)!=0o600:raise RuntimeError('Untrusted restore state')
            record=json.loads(STATE.read_text());target=record['original_utf8'].encode()
            if sha(target)!=BASELINE_SHA or record['original_sha256']!=BASELINE_SHA or sha(current) not in [record['configured_sha256'],record['original_sha256']]:raise RuntimeError('設定が別途変更されています。自動上書きしません。')
            print('保存した元の設定へバイト単位で戻します。観測ログは保持します。')
            if not a.apply:return
            inactive()
            if CONFIG.is_symlink() or CONFIG.read_bytes()!=current:raise RuntimeError('Production config changed before restore')
            atomic(CONFIG,target,record['mode'],record['uid'],record['gid'])
            archive=PRIVATE/('profile.restored-'+str(time.time_ns())+'.json');os.replace(STATE,archive)
        print('設定完了。起動する場合: sudo python3 '+str(ENGINE/'start.py'))
if __name__=='__main__':main()
