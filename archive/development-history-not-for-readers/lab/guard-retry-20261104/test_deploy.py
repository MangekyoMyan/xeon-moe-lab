import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import deploy
from test_gpu_guard import initial

class DeployTests(unittest.TestCase):
    def exercise(self,mode):
        with tempfile.TemporaryDirectory() as tmp:
            lab=Path(tmp);r=lab/'study';r.mkdir();e=lab/'engine';e.mkdir()
            lock=lab/'results/glm-air-q8-20261020/.experiment.lock';lock.parent.mkdir(parents=True);lock.touch()
            config=b'{}';(e/'server-config.json').write_bytes(config)
            old=b'old supervisor';new=b'new supervisor';(e/'supervise.py').write_bytes(old);(r/'supervise.py').write_bytes(new);(r/'gpu_guard.py').write_bytes(b'helper')
            (r/'deploy.py').write_bytes(b'deployment stub');(r/'recovery.py').write_bytes(b'recovery stub')
            ref={'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'started_unix':123,'finished_unix':456,'status':'stopped_on_error',
              'error':"TimeoutExpired(['nvidia-smi', '-i', 'GPU-test', '-q', '-x'], 5)",'child_returncode':-15,'pid':100,'initial_gpu':initial()}
            health=e/'health.json';raw=(json.dumps(ref)+'\n').encode();health.write_bytes(raw);marker=e/'STOP_REQUESTED';marker_raw=(ref['error']+'\n').encode();marker.write_bytes(marker_raw)
            incident=lab/'incident.json';incident.write_bytes(raw)
            validation={'passed':True,'sources':{name:deploy.sha((r/name).read_bytes()) for name in ['supervise.py','gpu_guard.py','deploy.py','recovery.py']},'incident_sha256':deploy.sha(raw)}
            if mode=='missing_validation':validation['sources'].pop('gpu_guard.py')
            (r/'VALIDATION.json').write_text(json.dumps(validation))
            calls=0
            def no_model():
                nonlocal calls
                calls+=1
                if mode=='active':raise RuntimeError('API active')
                if mode=='changed_marker' and calls==2:marker.write_bytes(b'new fault\n')
                if mode=='tampered_install' and calls==2:(e/'gpu_guard.py').write_bytes(b'tampered')
            real_atomic=deploy.atomic
            def write(path,data,*args,**kwargs):
                real_atomic(path,data,*args,**kwargs)
                if mode=='mid_marker' and path==health and json.loads(data)['status']=='stop_reviewed':marker.write_bytes(b'new fault\n')
            real_replace=os.replace
            def replace(src,dst):
                if mode=='rename_failure' and Path(src)==marker:raise OSError('simulated marker rename failure')
                return real_replace(src,dst)
            if mode=='changed_evidence':incident.write_bytes(b'{}')
            patches={'R':r,'LAB':lab,'E':e,'HEALTH':health,'MARKER':marker,'INCIDENT':incident,'ALARM':'api_guard:'+str(health),
                     'BASE_SHA':deploy.sha(old),'CONFIG_SHA':deploy.sha(config)}
            with mock.patch.multiple(deploy,**patches),mock.patch.object(sys,'argv',['deploy.py','--apply']),\
                 mock.patch.object(deploy.os,'geteuid',return_value=0),mock.patch.object(deploy,'no_active_model',side_effect=no_model),\
                 mock.patch.object(deploy,'monitor_check',return_value={'unix':time.time(),'active_alerts':{}}),\
                 mock.patch.object(deploy,'atomic',side_effect=write),mock.patch.object(deploy.os,'replace',side_effect=replace),contextlib.redirect_stdout(io.StringIO()):
                if mode=='success':deploy.main()
                else:
                    with self.assertRaises((RuntimeError,OSError)):deploy.main()
            if mode=='success':
                self.assertFalse(marker.exists());self.assertEqual((e/'supervise.py').read_bytes(),new)
                self.assertEqual(json.loads(health.read_text())['status'],'stop_reviewed')
                rec=json.loads((r/'APPLIED.json').read_text());self.assertFalse(rec['model_started']);self.assertTrue(rec['monitor_alarm_cleared'])
                backup=Path(rec['backup']);self.assertEqual((backup/'health.original.json').read_bytes(),raw);self.assertEqual((backup/'STOP_REQUESTED.cleared').read_bytes(),marker_raw)
            else:
                self.assertTrue(marker.exists());self.assertEqual(health.read_bytes(),raw)
                self.assertEqual(marker.read_bytes(),b'new fault\n' if mode in ['changed_marker','mid_marker'] else marker_raw)
                if mode in ['active','changed_evidence','missing_validation']:self.assertEqual((e/'supervise.py').read_bytes(),old)
    def test_success_archives_and_does_not_start(self):self.exercise('success')
    def test_active_model_refuses_install(self):self.exercise('active')
    def test_changed_evidence_refuses_recovery(self):self.exercise('changed_evidence')
    def test_new_marker_not_overwritten(self):self.exercise('changed_marker')
    def test_failed_marker_clear_restores_original_health(self):self.exercise('rename_failure')
    def test_marker_change_during_health_write_does_not_hide_alarm(self):self.exercise('mid_marker')
    def test_modified_installed_code_prevents_recovery(self):self.exercise('tampered_install')
    def test_incomplete_validation_refuses_install(self):self.exercise('missing_validation')
    def test_atomic_does_not_remove_existing_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data';other=p.with_name('.data.guard-update.tmp');other.write_bytes(b'previous transaction')
            with self.assertRaises(FileExistsError):deploy.atomic(p,b'new data')
            self.assertEqual(other.read_bytes(),b'previous transaction')

if __name__=='__main__':unittest.main(verbosity=2)
