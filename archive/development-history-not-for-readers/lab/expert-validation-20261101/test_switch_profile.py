import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('switch', Path(__file__).with_name('switch_profile.py'))
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root/'server-config.json'
        self.before = b'{"profile":"baseline"}\n'
        self.target = b'{"profile":"gpu2"}\n'
        self.config.write_bytes(self.before)
        self.context = contextlib.ExitStack()
        self.context.enter_context(patch.object(switch, 'ROOT', self.root))
        self.context.enter_context(patch.object(switch, 'CONFIG', self.config))
        self.context.enter_context(patch.object(switch, 'ENGINE', self.root))
        self.wait_patch = self.context.enter_context(patch.object(switch, 'wait_for_monitor_release'))
        self.context.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.context.close()
        self.tmp.cleanup()

    def test_busy_api_is_never_stopped(self):
        with patch.object(switch, 'ensure_monitor'), patch.object(switch, 'active', return_value=True), \
             patch.object(switch, 'ensure_idle', side_effect=RuntimeError('busy')), patch.object(switch.subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                switch.apply(self.before,self.target)
            run.assert_not_called()
            self.assertEqual(self.config.read_bytes(),self.before)

    def test_hardware_guard_blocks_all_mutation(self):
        with patch.object(switch,'ensure_monitor',side_effect=RuntimeError('guard')), patch.object(switch.subprocess,'run') as run:
            with self.assertRaises(RuntimeError):
                switch.apply(self.before,self.target)
            run.assert_not_called()
            self.assertEqual(self.config.read_bytes(),self.before)

    def test_rejected_start_restores_config_without_retry(self):
        with patch.object(switch,'ensure_monitor'), patch.object(switch,'active',return_value=False), \
             patch.object(switch.subprocess,'run',side_effect=[subprocess.CompletedProcess([],0),subprocess.CalledProcessError(1,['start'])]) as run:
            with self.assertRaisesRegex(RuntimeError,'変更前の設定へ戻しました'):
                switch.apply(self.before,self.target)
            self.assertEqual(self.config.read_bytes(),self.before)
            self.assertEqual(run.call_count,2)

    def test_success_only_stops_named_api_and_requests_one_start(self):
        with patch.object(switch,'ensure_monitor'), patch.object(switch,'active',return_value=False), \
             patch.object(switch.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as run:
            switch.apply(self.before,self.target)
            self.assertEqual(self.config.read_bytes(),self.target)
            self.assertEqual(run.call_args_list[0].args[0],['systemctl','stop','llm-deepseek41-api.service'])
            self.assertEqual(run.call_count,2)
            self.assertEqual(len(list((self.root/'profile-history').glob('*/before.json'))),1)

    def test_unknown_config_is_not_overwritten(self):
        p=self.root/'profiles';p.mkdir()
        (p/'gpu2.json').write_bytes(self.target)
        (p/'manifest.json').write_text(json.dumps({'profiles':{'gpu2':{'file':'gpu2.json','sha256':switch.digest(self.target)}}}))
        with self.assertRaises(RuntimeError):
            switch.load_verified('gpu2')
        self.assertEqual(self.config.read_bytes(),self.before)

    def test_concurrent_change_is_preserved(self):
        external=b'{"profile":"external"}\n'
        def stop(*args,**kwargs):
            self.config.write_bytes(external)
            return subprocess.CompletedProcess([],0)
        with patch.object(switch,'ensure_monitor'), patch.object(switch,'active',return_value=False), \
             patch.object(switch.subprocess,'run',side_effect=stop) as run:
            with self.assertRaises(RuntimeError):
                switch.apply(self.before,self.target)
            self.assertEqual(self.config.read_bytes(),external)
            self.assertEqual(run.call_count,1)


class MonitorFreshnessTests(unittest.TestCase):
    def test_old_free_and_fresh_busy_records_do_not_allow_start(self):
        records=[{'unix':9,'errors':{},'active_alerts':{},'gpu':{'used_MiB':0}},
                 {'unix':11,'errors':{},'active_alerts':{},'gpu':{'used_MiB':8560}},
                 {'unix':12,'errors':{},'active_alerts':{},'gpu':{'used_MiB':0}}]
        with patch.object(switch,'read_monitor',side_effect=records) as read, patch.object(switch.time,'sleep') as sleep:
            switch.wait_for_monitor_release(10)
            self.assertEqual(read.call_count,3)
            self.assertEqual(sleep.call_count,2)

    def test_fresh_hardware_alert_prevents_restart(self):
        record={'unix':11,'errors':{},'active_alerts':{'gpu':'hot'},'gpu':{'used_MiB':0}}
        with patch.object(switch,'read_monitor',return_value=record):
            with self.assertRaises(RuntimeError):
                switch.wait_for_monitor_release(10)


if __name__ == '__main__':
    unittest.main()
