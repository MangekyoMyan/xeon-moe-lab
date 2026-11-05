from contextlib import ExitStack,redirect_stdout
from pathlib import Path
from io import StringIO
import json
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import switch_profile as s

class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config=self.root/'server-config.json';self.config.write_bytes(b'old-config')
        self.watch=self.root/'watch.json';self.watch.write_text(json.dumps({'unix':time.time(),'errors':[],
          'active_alerts':[],'gpu':{'ecc_mode':'Enabled','power_limit_W':150}}))
        for k,v in [('ENGINE',self.root),('CONFIG',self.config),('WATCH',self.watch)]:self.stack.enter_context(patch.object(s,k,v))
        self.stack.enter_context(patch.object(s.os,'fchown'))
        self.wait=self.stack.enter_context(patch.object(s,'wait_release'))
        self.run=self.stack.enter_context(patch.object(s.subprocess,'run',side_effect=self.command))
        self.stack.enter_context(redirect_stdout(StringIO()))
        self.fail=False;self.active_after=False;self.concurrent=False
    def command(self,cmd,**kw):
        if cmd[:2]==['systemctl','is-active']:
            active=cmd[-1]==s.UNIT and self.active_after
            return subprocess.CompletedProcess(cmd,0 if active else 3)
        if cmd[0]=='python3' and self.fail:
            if self.concurrent:self.config.write_bytes(b'external-change')
            raise subprocess.CalledProcessError(1,cmd)
        return subprocess.CompletedProcess(cmd,0)
    def test_success(self):
        s.apply(b'old-config',b'new-config');self.assertEqual(self.config.read_bytes(),b'new-config')
    def test_failure_inactive_rolls_back(self):
        self.fail=True
        with self.assertRaises(subprocess.CalledProcessError):s.apply(b'old-config',b'new-config')
        self.assertEqual(self.config.read_bytes(),b'old-config')
    def test_started_service_not_rolled_back(self):
        self.fail=True;self.active_after=True
        with self.assertRaises(subprocess.CalledProcessError):s.apply(b'old-config',b'new-config')
        self.assertEqual(self.config.read_bytes(),b'new-config')
    def test_concurrent_config_not_rolled_back(self):
        self.fail=True;self.concurrent=True
        with self.assertRaises(subprocess.CalledProcessError):s.apply(b'old-config',b'new-config')
        self.assertEqual(self.config.read_bytes(),b'external-change')
    def test_change_during_shutdown_not_overwritten(self):
        self.wait.side_effect=lambda _:self.config.write_bytes(b'external-change')
        with self.assertRaises(RuntimeError):s.apply(b'old-config',b'new-config')
        self.assertEqual(self.config.read_bytes(),b'external-change')
        self.assertFalse(any(c.args[0][0]=='python3' for c in self.run.call_args_list))
    def test_protection_latch_preserved(self):
        (self.root/'STOP_REQUESTED').touch()
        with self.assertRaises(RuntimeError):s.apply(b'old-config',b'new-config')
        self.assertFalse(any(c.args[0][:2]==['systemctl','stop'] for c in self.run.call_args_list))
    def test_stale_monitor_keeps_api_running(self):
        j=json.loads(self.watch.read_text());j['unix']=0;self.watch.write_text(json.dumps(j))
        with self.assertRaises(RuntimeError):s.apply(b'old-config',b'new-config')
        self.assertFalse(any(c.args[0][:2]==['systemctl','stop'] for c in self.run.call_args_list))

if __name__=='__main__':unittest.main()
