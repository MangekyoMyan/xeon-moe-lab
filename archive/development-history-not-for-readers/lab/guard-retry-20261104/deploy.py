#!/usr/bin/env python3
"""Install the tested guard and acknowledge only the investigated stop; never starts a model."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time

from gpu_guard import read_snapshot, validate_snapshot, number
from recovery import verify_incident, reviewed_health

R=Path(__file__).resolve().parent;LAB=R.parent;E=LAB/'deepseek41-bringup-20261101'
INCIDENT=LAB/'guard-incident-20261104/health-evidence.json'
HEALTH=E/'health.json';MARKER=E/'STOP_REQUESTED';ALARM='api_guard:'+str(HEALTH)
BASE_SHA='f31ec8ae7176a549e6ebc2494f3a23e77f92c3bd8d0dca6d8ea4a4512bfdafb7'
CONFIG_SHA='e73c2bd30ac5dc4b31f6d4bf5b3d9e23c8603c7f638114a50cf9541629631240'


def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(path,data,mode=0o600,uid=None,gid=None):
    if isinstance(data,dict):data=(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode()
    tmp=path.with_name('.'+path.name+'.guard-update.tmp')
    created=False
    try:
        fd=os.open(tmp,os.O_CREAT|os.O_EXCL|os.O_WRONLY,mode)
        created=True
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
            if uid is not None:os.fchown(f.fileno(),uid,gid)
            os.fchmod(f.fileno(),mode)
        os.replace(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if created and tmp.exists():tmp.unlink()


def no_active_model():
    for unit in ['llm-deepseek41-api.service','llm-openai-api.service']:
        result=subprocess.check_output(['systemctl','show',unit,'--property=ActiveState,MainPID'],text=True,timeout=5)
        props=dict(line.split('=',1) for line in result.splitlines() if '=' in line)
        if props.get('ActiveState') not in ['inactive','failed'] or props.get('MainPID')!='0':
            raise RuntimeError('API is active/transitioning; refusing guard recovery: '+unit)
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:name=path.read_text().strip()
        except FileNotFoundError:continue
        if name=='llama-server':raise RuntimeError('An inference process is still running')


def monitor_check(reference,boot,allow_old_alarm):
    subprocess.run(['systemctl','is-active','--quiet','xeon-watch.service'],check=True,timeout=5)
    subprocess.run(['systemctl','is-active','--quiet','xeon-watch-notify.service'],check=True,timeout=5)
    sample=read_snapshot();counters=sample.get('host_counters',{})
    # This investigated boot had zero EDAC/throttle counters. Any new counter prevents acknowledgement.
    zero_counters={key:0 for key in counters}
    validate_snapshot(sample,reference['initial_gpu'],zero_counters,boot,time.time(),allowed_alert=ALARM if allow_old_alarm else None)
    if number(sample['gpu'].get('used_MiB'),'VRAM')>128:raise RuntimeError('GPU is still in use')
    alarm=sample['active_alerts'].get(ALARM)
    if alarm is not None and alarm.get('observation',{}).get('value')!=reference['error']:
        raise RuntimeError('Independent API alarm changed')
    return sample


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
    if os.geteuid()!=0:raise SystemExit('Run with sudo; this only installs the guard and reviews the saved stop.')
    validation=json.loads((R/'VALIDATION.json').read_text())
    if not validation['passed']:raise RuntimeError('Tests have not passed')
    required={'supervise.py','gpu_guard.py','recovery.py','deploy.py'}
    if not required.issubset(validation['sources']):raise RuntimeError('Validation manifest is incomplete')
    for name,digest in validation['sources'].items():
        if sha((R/name).read_bytes())!=digest:raise RuntimeError('Source changed after validation: '+name)
    reference_bytes=INCIDENT.read_bytes()
    if sha(reference_bytes)!=validation['incident_sha256']:raise RuntimeError('Incident evidence changed')
    reference=json.loads(reference_bytes);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    lockpath=LAB/'results/glm-air-q8-20261020/.experiment.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        no_active_model()
        if sha((E/'server-config.json').read_bytes())!=CONFIG_SHA:raise RuntimeError('Model profile changed')
        if (LAB/'moe-engine-20261029/STOP_REQUESTED').exists():raise RuntimeError('Another hardware stop is present')
        health_raw=HEALTH.read_bytes();marker_raw=MARKER.read_bytes();health=json.loads(health_raw)
        verify_incident(health,marker_raw,reference,boot)
        monitor_check(reference,boot,True)
        old_supervisor=(E/'supervise.py').read_bytes();new_supervisor=(R/'supervise.py').read_bytes()
        if sha(old_supervisor) not in [BASE_SHA,sha(new_supervisor)]:raise RuntimeError('Supervisor changed independently')
        existing_helper=E/'gpu_guard.py'
        if existing_helper.exists() and existing_helper.read_bytes()!=(R/'gpu_guard.py').read_bytes():raise RuntimeError('An unrelated GPU guard helper exists')
        print('Verified: exact reviewed incident, inactive model, fresh healthy hardware monitor, validated patch.',flush=True)
        if not args.apply:return
        stamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S');backup=R/('applied-'+stamp);backup.mkdir(mode=0o700)
        for name,data in [('supervise.before.py',old_supervisor),('health.original.json',health_raw),('STOP_REQUESTED.original',marker_raw)]:atomic(backup/name,data)
        owner=(E/'supervise.py').stat();atomic(E/'gpu_guard.py',(R/'gpu_guard.py').read_bytes(),0o644,owner.st_uid,owner.st_gid)
        atomic(E/'supervise.py',new_supervisor,owner.st_mode&0o777,owner.st_uid,owner.st_gid)
        no_active_model();monitor_check(reference,boot,True)
        for name in ['supervise.py','gpu_guard.py']:
            if sha((E/name).read_bytes())!=validation['sources'][name]:raise RuntimeError('Installed guard verification failed: '+name)
        if HEALTH.read_bytes()!=health_raw or MARKER.read_bytes()!=marker_raw:raise RuntimeError('Incident changed during install; original stop retained')
        reviewed=reviewed_health(health,backup,time.time());reviewed_raw=(json.dumps(reviewed,indent=2,ensure_ascii=False)+'\n').encode()
        try:
            atomic(HEALTH,reviewed_raw)
            if MARKER.read_bytes()!=marker_raw:raise RuntimeError('Marker changed before clearing')
            os.replace(MARKER,backup/'STOP_REQUESTED.cleared')
        except BaseException:
            # Roll back only this transaction, and never overwrite a newly reported fault.
            if HEALTH.read_bytes()==reviewed_raw:
                atomic(HEALTH,health_raw)
            raise
        for directory in [E,backup]:
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        record={'applied_unix':time.time(),'backup':str(backup),'supervisor_sha256':sha((E/'supervise.py').read_bytes()),
                'helper_sha256':sha((E/'gpu_guard.py').read_bytes()),'old_incident_acknowledged':True,'model_started':False,
                'monitor_alarm_cleared':False,'config_sha256':sha((E/'server-config.json').read_bytes())}
        atomic(R/'APPLIED.json',record,0o644)
        print('Guard installed; original incident archived. Waiting briefly for the existing monitor to clear its old latch.',flush=True)
        deadline=time.monotonic()+50
        while time.monotonic()<deadline:
            sample=monitor_check(reference,boot,True)
            if ALARM not in sample['active_alerts']:
                record['monitor_alarm_cleared']=True;record['monitor_checked_unix']=sample['unix'];break
            time.sleep(2)
        atomic(R/'APPLIED.json',record,0o644)
        print(json.dumps(record),flush=True)
        if not record['monitor_alarm_cleared']:
            raise SystemExit('Guard installed but monitor recovery not yet confirmed. Do not start until its old alarm clears.')
        print('Ready for user-run start.py. No model was started.',flush=True)

if __name__=='__main__':main()
