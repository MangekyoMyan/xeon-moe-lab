#!/usr/bin/env python3
"""Fixture-only tests. Never stop a real service or touch production config."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import analyze_histogram as hist
import switch_cpu as switch

class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.engine=self.root/'engine';self.engine.mkdir()
        self.study=self.root/'study';self.study.mkdir()
        self.config=self.engine/'server-config.json'
        self.baseline=b'{"environment":{"LD_LIBRARY_PATH":"/original"},"sentinel":73}\n'
        self.config.write_bytes(self.baseline)
        (self.study/'baseline-server-config.json').write_bytes(self.baseline)
        lib=self.study/'cpu.so';lib.write_bytes(b'test-library')
        profile={'library':str(lib),'directory':str(self.study),'sha256':switch.sha(lib.read_bytes())}
        (self.study/'RUNTIME.json').write_text(json.dumps({'baseline_config_sha256':switch.sha(self.baseline),
          'profiles':{name:profile for name in ['hybrid','pair','stock-hist']}}))
        for key,value in [('R',self.study),('LAB',self.root),('ENGINE',self.engine),('CONFIG',self.config)]:
            self.stack.enter_context(patch.object(switch,key,value))
        self.stack.enter_context(patch.object(switch.os,'geteuid',return_value=0))
        self.stack.enter_context(patch.object(switch.os,'chown'))
        self.run=self.stack.enter_context(patch.object(switch.subprocess,'run',side_effect=self.fake_run))
        self.spawn=self.stack.enter_context(patch.object(switch.subprocess,'Popen'))
        self.release=self.stack.enter_context(patch.object(switch,'wait_for_gpu_release'))
        self.output=self.stack.enter_context(redirect_stdout(StringIO()))
        self.start_error=False
    def fake_run(self,cmd,**kwargs):
        if cmd[:2]==['systemctl','is-active']:return subprocess.CompletedProcess(cmd,1)
        if cmd[0]=='python3' and self.start_error:raise subprocess.CalledProcessError(1,cmd)
        return subprocess.CompletedProcess(cmd,0)
    def invoke(self,*args):
        with patch('sys.argv',['switch_cpu.py',*args]):switch.main()
    def test_dry_run_no_service_or_config_change(self):
        self.invoke('hybrid');self.run.assert_not_called();self.spawn.assert_not_called()
        self.assertEqual(self.config.read_bytes(),self.baseline)
    def test_unknown_config_refused(self):
        self.config.write_bytes(b'{}')
        with self.assertRaises(SystemExit):self.invoke('hybrid','--apply')
        self.run.assert_not_called();self.assertEqual(self.config.read_bytes(),b'{}')
    def test_library_change_refused(self):
        (self.study/'cpu.so').write_bytes(b'changed')
        with self.assertRaises(SystemExit):self.invoke('pair','--apply')
        self.run.assert_not_called()
    def test_failed_start_restores_exact_previous_bytes(self):
        self.start_error=True
        with self.assertRaises(subprocess.CalledProcessError):self.invoke('hybrid','--apply')
        self.assertEqual(self.config.read_bytes(),self.baseline)
        self.assertFalse((self.study/'profile-history.json').exists())
    def test_switch_then_restore(self):
        self.invoke('hybrid','--apply')
        self.assertEqual(json.loads(self.config.read_bytes())['cpu_profile']['name'],'hybrid')
        self.assertEqual(json.loads(self.config.read_bytes())['sentinel'],73)
        self.invoke('stock','--apply');self.assertEqual(self.config.read_bytes(),self.baseline)
        self.assertEqual(len(json.loads((self.study/'profile-history.json').read_text())),2)
    def test_hist_capture_paths_and_readiness_worker(self):
        self.invoke('stock-hist','--apply')
        config=json.loads(self.config.read_bytes())
        path=Path(config['environment']['GGML_MOE_HIST_FILE'])
        self.assertTrue(path.parent.is_dir());self.assertFalse(path.exists())
        self.spawn.assert_called_once()
    def test_other_api_not_stopped(self):
        self.run.side_effect=lambda cmd,**kw:subprocess.CompletedProcess(cmd,0)
        with self.assertRaises(SystemExit):self.invoke('hybrid','--apply')
        self.assertEqual(self.run.call_count,1);self.assertEqual(self.config.read_bytes(),self.baseline)
    def test_no_gpu_release_preserves_config(self):
        self.release.side_effect=SystemExit('not released')
        with self.assertRaises(SystemExit):self.invoke('hybrid','--apply')
        self.assertEqual(self.config.read_bytes(),self.baseline)
        self.assertFalse(any(c.args[0][0]=='python3' for c in self.run.call_args_list))

class HistogramTests(unittest.TestCase):
    def test_readiness_zero_experts_truncation_and_partial_line(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(hist.analyze(root)['status'],'waiting_for_ready')
            (root/'READY.json').write_text('{"ready":true,"monotonic_s":10}')
            row={'monotonic_s':11,'tensor':'blk.0.ffn_up_exps.weight','weight_type':39,
                 'batch_tokens':2,'experts':4,'active_experts':2,'assigned_rows':6,
                 'hist_0_1_2_3_4to7_8to15_16to31_32plus':[2,0,0,2,0,0,0,0],
                 'mx421_plan_eligible':True,'planned_rows_4_2_1':[0,4,2]}
            warm=dict(row,monotonic_s=9)
            (root/'histogram.jsonl').write_text(json.dumps(warm)+'\n'+json.dumps(row)+'\n{"truncated":true}\n{"partial":')
            result=hist.analyze(root);s=result['groups']['batch>=2/MXFP4']
            self.assertEqual(result['warmup_operations_excluded'],1)
            self.assertEqual(result['partial_lines_ignored'],1);self.assertTrue(result['truncated'])
            self.assertEqual(s['rows_per_expert_including_zero'],1.5)
            self.assertEqual(s['rows_per_active_expert'],3)
            self.assertAlmostEqual(s['planned_percent_4_2_1'][1],200/3)

if __name__=='__main__':unittest.main()
