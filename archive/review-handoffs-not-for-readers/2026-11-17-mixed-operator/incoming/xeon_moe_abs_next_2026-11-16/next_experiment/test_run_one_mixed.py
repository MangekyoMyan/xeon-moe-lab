#!/usr/bin/env python3
"""Synthetic launcher tests: Popen is replaced; no executable or model is run."""
import argparse
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mixed_launcher', HERE/'run_one_mixed.py')
launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
CASE_DIR = None


def fake_outputs(folder, index, spec, cpu, base, success=True):
    rotation=(index//2)%3
    rows=[{'kind':'mixed_meta','schema':1,'process_index':index,'library_label':launcher.LABELS[index],
           'commit':launcher.COMMIT,'tensor':spec['tensor_name'],'routes':spec['routes'],
           'inputs_sha256':launcher.INPUT_SHA,'plan_sha256':launcher.PLAN_SHA,'hashes_verified_by_harness':False,
           'threads':12,'blocks':16,'repeats':32,'ggml_numa':False,'clock':'CLOCK_MONOTONIC',
           'cpu_provider':cpu,'base_provider':base,'cache_provider':cpu,
           'mode_provider':cpu if launcher.LABELS[index]=='E' else '',
           'case_order':list(launcher.CASES[rotation:]+launcher.CASES[:rotation])}]
    for name in launcher.CASES:
        (folder/('expected-'+name+'.bin')).write_bytes(bytes(launcher.EXPECTED_BYTES))
        rows.append({'kind':'mixed_case','case':name,'selected_ids':[] if name=='C0' else spec['routes'] if name=='C6' else spec['cache_ids'],
                     'cache_bytes':spec['cache_cases'][name]['cache_payload_bytes'],'pins':list(range(12)),'bit_mismatches':0})
        rows.extend({'kind':'mixed_chunk','case':name,'bit_mismatches':0} for _ in range(64))
    rows.append({'kind':'mixed_complete','passed':success,'error':None if success else 'synthetic failure',
                 'chunks':192,'timed_graph_calls':6144,'outputs_bytes':3*launcher.EXPECTED_BYTES})
    (folder/'raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (folder/'maps.txt').write_text('SYNTHETIC: no library was mapped\n')


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.cpu={}
        for label in ('F','E'):
            p=self.folder/('cpu-'+label)/'fake-cpu.so';p.parent.mkdir();p.write_bytes(('FAKE_'+label).encode());self.cpu[label]=p
        self.base=self.folder/'fake-base.so';self.base.write_bytes(b'FAKE_BASE')
        self.harness=self.folder/'fake-harness';self.harness.write_bytes(b'NEVER_EXECUTED');self.harness.chmod(0o700)
        self.numactl=self.folder/'fake-numactl';self.numactl.write_bytes(b'NEVER_EXECUTED');self.numactl.chmod(0o700)
        self.shard=self.folder/'fake-shard';self.shard.write_bytes(b'SYNTHETIC_NOT_A_MODEL')
        self.run_root=self.folder/'run'
        self.case_spec=json.loads((CASE_DIR/'mixed_case.json').read_text())
        self.children=[];self.behavior='success'
        actual_record=launcher.file_record
        def record(path):
            result=actual_record(path)
            for label,p in self.cpu.items():
                if path==p:result['sha256']=launcher.CPU_SHA[label]
            return result
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(launcher,'file_record',record))
        self.stack.enter_context(patch.object(launcher,'host_preflight',lambda:(self.numactl,{'synthetic':True})))
        owner=self
        class FakeChild:
            def __init__(self,argv,**kwargs):
                self.pid=987654;self.returncode=None;self.terminated=False;self.killed=False;self.first_wait=True
                owner.children.append(self);self.argv=argv;self.kwargs=kwargs
                folder=Path(argv[7]);index=int(argv[8]);fake_outputs(folder,index,owner.case_spec,argv[9],argv[10],owner.behavior!='failed_complete')
                kwargs['stdout'].write(b'{"passed":true,"scope":"SYNTHETIC"}\n')
            def poll(self):return self.returncode
            def wait(self,timeout=None):
                if self.first_wait:
                    self.first_wait=False
                    if owner.behavior=='timeout':raise subprocess.TimeoutExpired(self.argv,timeout)
                    if owner.behavior=='signal':raise InterruptedError('synthetic SIGTERM')
                if self.returncode is None:self.returncode=0
                return self.returncode
            def terminate(self):self.terminated=True;self.returncode=-15
            def kill(self):self.killed=True;self.returncode=-9
        self.stack.enter_context(patch.object(launcher.subprocess,'Popen',FakeChild))

    def args(self,index,label=None):
        return argparse.Namespace(harness=self.harness,cpu_library=self.cpu[label or launcher.LABELS[index]],base_library=self.base,
                                  shard=self.shard,case_dir=CASE_DIR,run_root=self.run_root,library_dir=[],process_index=index)
    def invoke(self,index,label=None):
        with redirect_stdout(io.StringIO()):return launcher.run_one(self.args(index,label))
    def status(self,index):return json.loads((self.run_root/('process-'+str(index))/'STATUS.json').read_text())

    def test_all_eight_sequential_fake_processes(self):
        for index in range(8):
            self.assertEqual(self.invoke(index),0)
            row=self.status(index);self.assertTrue(row['passed']);self.assertEqual(row['error'],None)
            self.assertEqual(set(row['files']),set(launcher.REQUIRED_FILES))
            self.assertEqual(row['library_label'],launcher.LABELS[index])
        self.assertEqual(len(self.children),8)
        for child in self.children:
            self.assertFalse(child.kwargs['start_new_session'])
            self.assertEqual(child.argv[1:3],['--physcpubind=0-23','--interleave=0,1'])

    def test_wrong_cpu_label_rejected_before_spawn(self):
        self.assertEqual(self.invoke(0,'E'),1);self.assertFalse(self.children)
        self.assertIn('CPU hash',self.status(0)['error'])

    def test_out_of_order_rejected_before_spawn(self):
        self.assertEqual(self.invoke(1),1);self.assertFalse(self.children)

    def test_existing_process_cannot_be_retried(self):
        self.assertEqual(self.invoke(0),0)
        before=(self.run_root/'process-0/STATUS.json').read_bytes()
        with self.assertRaises(FileExistsError):self.invoke(0)
        self.assertEqual(before,(self.run_root/'process-0/STATUS.json').read_bytes())

    def test_base_identity_drift_rejected(self):
        self.assertEqual(self.invoke(0),0);self.base.write_bytes(b'CHANGED_BASE')
        self.assertEqual(self.invoke(1),1);self.assertEqual(len(self.children),1)
        self.assertIn('base_library_sha256',self.status(1)['error'])

    def test_previous_output_tamper_rejected(self):
        self.assertEqual(self.invoke(0),0)
        (self.run_root/'process-0/expected-Cplan.bin').write_bytes(b'CORRUPTED')
        self.assertEqual(self.invoke(1),1);self.assertEqual(len(self.children),1)

    def test_previous_status_must_be_completed(self):
        self.assertEqual(self.invoke(0),0)
        row=self.status(0);row['state']='starting'
        (self.run_root/'process-0/STATUS.json').write_text(json.dumps(row))
        self.assertEqual(self.invoke(1),1);self.assertEqual(len(self.children),1)

    def test_failed_completion_preserved_and_blocks_next(self):
        self.behavior='failed_complete';self.assertEqual(self.invoke(0),1)
        row=self.status(0);self.assertFalse(row['passed']);self.assertEqual(row['exit_code'],0)
        self.assertEqual(set(row['files']),set(launcher.REQUIRED_FILES))
        self.behavior='success';self.assertEqual(self.invoke(1),1);self.assertEqual(len(self.children),1)

    def test_timeout_terminates_only_child_and_keeps_raw(self):
        self.behavior='timeout';self.assertEqual(self.invoke(0),1)
        self.assertTrue(self.children[0].terminated);self.assertEqual(self.status(0)['exit_code'],-15)
        self.assertIn('TimeoutExpired',self.status(0)['error'])
        self.assertTrue((self.run_root/'process-0/raw.jsonl').is_file())

    def test_signal_path_terminates_child_and_writes_status(self):
        self.behavior='signal';self.assertEqual(self.invoke(0),1)
        self.assertTrue(self.children[0].terminated);self.assertIn('InterruptedError',self.status(0)['error'])

    def test_environment_removes_loader_and_ggml_overrides(self):
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'UNTRUSTED','LD_PRELOAD':'bad.so','GGML_CPU_TILED_MM':'0','OMP_THREAD_LIMIT':'1'}):
            env,receipt=launcher.child_environment(self.cpu['F'],self.base,[])
        self.assertNotIn('LD_PRELOAD',env);self.assertNotIn('GGML_CPU_TILED_MM',env)
        self.assertEqual(env['OMP_THREAD_LIMIT'],'24');self.assertNotIn('UNTRUSTED',env['LD_LIBRARY_PATH'])
        self.assertTrue(receipt['inherits_outer_process_group'])


class CaseAndPreflightTests(unittest.TestCase):
    def test_real_generated_case_bytes_read_only(self):
        spec,paths,records=launcher.verify_case(CASE_DIR)
        self.assertEqual(records['inputs.bin']['bytes'],163840)
        self.assertEqual(len(spec['cache_ids']),39)

    def test_input_and_plan_tampering_rejected(self):
        for filename in ('inputs.bin','source-plan.txt','mixed_case_data.h'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as folder:
                case=Path(folder)/'case';shutil.copytree(CASE_DIR,case)
                with (case/filename).open('ab') as f:f.write(b'X')
                with self.assertRaises(RuntimeError):launcher.verify_case(case)

    def test_cpu_list_parser_rejects_invalid_ranges(self):
        for value in ('','2-1','1,,3','-1','0-99999','a'):
            with self.subTest(value=value),self.assertRaises(RuntimeError):launcher.parse_cpu_list(value)

    def test_preflight_with_synthetic_sysfs_and_meminfo(self):
        with tempfile.TemporaryDirectory() as folder:
            executable=Path(folder)/'fake-numactl';executable.write_bytes(b'NOT_EXECUTED');executable.chmod(0o700)
            mapping={'/sys/devices/system/cpu/online':'0-23','/sys/devices/system/node/online':'0-1',
                     '/proc/self/status':'Mems_allowed_list:\t0-1\n',
                     '/proc/meminfo':'MemAvailable: '+str((launcher.RESERVE_BYTES+launcher.CACHE_BUDGET_BYTES+1024**2)//1024)+' kB\n'}
            for cpu in range(24):
                root='/sys/devices/system/cpu/cpu'+str(cpu)+'/topology/'
                mapping[root+'physical_package_id']=str((cpu//6)%2);mapping[root+'core_id']=str(cpu%6)
            original=Path.read_text
            def read(path,*args,**kwargs):return mapping[str(path)] if str(path) in mapping else original(path,*args,**kwargs)
            with patch.object(Path,'read_text',read),patch.object(launcher.os,'sched_getaffinity',lambda _:set(range(24))),patch.object(launcher.shutil,'which',lambda _:str(executable)):
                command,host=launcher.host_preflight();self.assertEqual(command,executable);self.assertEqual(host['physical_cores'],12)
                mapping['/proc/meminfo']='MemAvailable: 1024 kB\n'
                with self.assertRaises(RuntimeError):launcher.host_preflight()
                mapping['/proc/meminfo']='MemAvailable: '+str(512*1024**2)+' kB\n'
                mapping['/sys/devices/system/cpu/cpu1/topology/core_id']='0'
                with self.assertRaises(RuntimeError):launcher.host_preflight()


def main():
    global CASE_DIR
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case-dir',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args();CASE_DIR=args.case_dir.resolve()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    receipt={'scope':'Synthetic/fake-Popen launcher tests only; no model, numactl, harness ELF, library, API, or GPU was executed.',
             'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
             'launcher_sha256':launcher.file_record(HERE/'run_one_mixed.py')['sha256'],
             'test_sha256':launcher.file_record(Path(__file__))['sha256']}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
