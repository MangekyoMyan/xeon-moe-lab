import contextlib
import copy
import importlib.machinery
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import gpu_guard
from test_gpu_guard import Clock, healthy, initial

R=Path(__file__).resolve().parent
class Loader:
    def create_module(self,spec):return None
    def exec_module(self,module):pass

def load_supervisor():
    module=types.ModuleType('test_supervisor_target');module.__file__=str(R/'supervise.py')
    spec=importlib.machinery.ModuleSpec('hybrid_watch',Loader())
    with mock.patch('importlib.util.spec_from_file_location',return_value=spec):
        exec(compile((R/'supervise.py').read_text(),str(R/'supervise.py'),'exec'),module.__dict__)
    return module

class Child:
    def __init__(self):self.pid=98765;self.returncode=None
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode
class Future:
    def done(self):return True
    def result(self):return {'sel':'','sensors':''}
class Pool:
    def __init__(self,*args,**kwargs):pass
    def submit(self,*args,**kwargs):return Future()
    def shutdown(self,**kwargs):pass

class IntegrationTests(unittest.TestCase):
    def run_supervisor(self,mode):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);engine=root/'engine';engine.mkdir();lock=root/'results/glm-air-q8-20261020/.experiment.lock';lock.parent.mkdir(parents=True);lock.touch()
            (engine/'server-config.json').write_text(json.dumps({'command':['do-not-run-any-model'],'environment':{}}))
            module=load_supervisor();module.ROOT=engine;module.LAB=root;child=Child();clock=Clock();gpu_calls=0;events=[];signals=[]
            def gpu():
                nonlocal gpu_calls
                gpu_calls+=1
                if gpu_calls==1:return initial() # strict startup preflight
                if mode=='fault':return dict(initial(),ecc_double=1)
                if mode=='fault_cancel':module.stop_requested=True;return dict(initial(),gpu_C=70)
                if mode=='pending_bmc_fault' and gpu_calls==2:return initial()
                if mode=='timeout_success' and gpu_calls>=3:return initial()
                if mode=='timeout_cancel':module.stop_requested=True
                if mode=='child_exits':child.returncode=7
                clock.t+=5
                raise __import__('subprocess').TimeoutExpired(['nvidia-smi'],5)
            def host():
                if mode=='host_fault' and gpu_calls>=2:raise RuntimeError('Host ECC/throttle counter changed')
                return {'cpu':28},1024**3
            watch=module.watch
            watch.gpu_sample=gpu;watch.counters=lambda:{'ECC':0};watch.fan_duty_sample=lambda:{'mode':'00','raw':[102]*4}
            watch.bmc_sample=lambda:{'sel':'','sensors':''};watch.environment=lambda:{};watch.nic_sample=lambda:{}
            def ctor(*args,**kwargs):
                return gpu_guard.GpuProbe(*args,**kwargs,snapshot=lambda:dict(healthy(clock),boot_id=args[3]),monotonic=clock.mono,wall=clock.wall,sleep=clock.sleep)
            class TestPool(Pool):
                def submit(self,*args,**kwargs):
                    future=Future()
                    if mode=='pending_bmc_fault':future.result=lambda:{'sel':'ECC fault','sensors':''}
                    return future
            def sleep(seconds):
                clock.sleep(seconds)
                if seconds==2 and mode!='pending_bmc_fault':child.returncode=0 # exit only after a complete healthy iteration
            def killpg(pid,signal):signals.append(int(signal));child.returncode=-int(signal)
            with mock.patch.object(module,'check_host',side_effect=lambda _:host()),mock.patch.object(module,'GpuProbe',side_effect=ctor),\
                 mock.patch.object(module.os,'geteuid',return_value=0),mock.patch.object(module.signal,'signal'),\
                 mock.patch.object(module.subprocess,'check_output',return_value=''),mock.patch.object(module.subprocess,'Popen',return_value=child),\
                 mock.patch.object(module.concurrent.futures,'ThreadPoolExecutor',TestPool),mock.patch.object(module.time,'sleep',side_effect=sleep),\
                 mock.patch.object(module.time,'monotonic',side_effect=clock.mono),mock.patch.object(module.os,'killpg',side_effect=killpg),\
                 contextlib.redirect_stderr(io.StringIO()):
                code=module.main()
            record=json.loads((engine/'health.json').read_text());marker=(engine/'STOP_REQUESTED').exists()
            return code,record,marker,signals,gpu_calls
    def test_timeout_recovers_without_stopping_child(self):
        code,record,marker,signals,calls=self.run_supervisor('timeout_success')
        self.assertEqual(code,0);self.assertEqual(record['status'],'exited');self.assertFalse(marker);self.assertEqual(signals,[])
        self.assertEqual(record['gpu_monitor']['recovered_queries'],1);self.assertEqual(calls,3)
    def test_persistent_timeout_latches_and_terminates(self):
        code,record,marker,signals,calls=self.run_supervisor('timeout_all')
        self.assertEqual(code,1);self.assertEqual(record['status'],'stopped_on_error');self.assertTrue(marker)
        self.assertEqual(signals,[15]);self.assertEqual(record['child_returncode'],-15);self.assertEqual(calls,4)
    def test_actual_fault_immediately_latches(self):
        code,record,marker,signals,calls=self.run_supervisor('fault')
        self.assertEqual(code,1);self.assertTrue(marker);self.assertEqual(calls,2);self.assertEqual(signals,[15])
        self.assertIn('counter changed',record['error'])
    def test_host_fault_during_retry_latches(self):
        code,record,marker,signals,calls=self.run_supervisor('host_fault')
        self.assertEqual(code,1);self.assertTrue(marker);self.assertEqual(calls,2);self.assertEqual(signals,[15])
    def test_user_stop_during_timeout_does_not_latch(self):
        code,record,marker,signals,calls=self.run_supervisor('timeout_cancel')
        self.assertEqual(code,0);self.assertEqual(record['status'],'stopped_by_request');self.assertFalse(marker)
        self.assertEqual(signals,[15]);self.assertEqual(calls,2)
    def test_pending_bmc_fault_stops_before_another_retry(self):
        code,record,marker,signals,calls=self.run_supervisor('pending_bmc_fault')
        self.assertEqual(code,1);self.assertTrue(marker);self.assertEqual(calls,3);self.assertEqual(signals,[15])
        self.assertIn('BMC reports an ECC event',record['error'])
    def test_observed_fault_not_hidden_by_user_stop(self):
        code,record,marker,signals,calls=self.run_supervisor('fault_cancel')
        self.assertEqual(code,1);self.assertTrue(marker);self.assertIn('thermal cutoff',record['error'])
    def test_child_exit_not_mislabeled_as_guard_error(self):
        code,record,marker,signals,calls=self.run_supervisor('child_exits')
        self.assertEqual(code,7);self.assertEqual(record['status'],'exited');self.assertFalse(marker);self.assertEqual(signals,[])

if __name__=='__main__':unittest.main(verbosity=2)
