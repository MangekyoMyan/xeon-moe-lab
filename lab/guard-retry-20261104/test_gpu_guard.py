import copy
import json
import subprocess
import unittest
from pathlib import Path
import gpu_guard as g

class Clock:
    def __init__(self):self.t=0.;self.stopped=False
    def mono(self):return self.t
    def wall(self):return 1000.+self.t
    def sleep(self,s):self.t+=s

def initial():return {'gpu_C':29.,'hbm_C':28.,'limit_W':150.,'ecc':'Enabled','ecc_single':0,'ecc_double':0,'replays':0}
def healthy(clock):
    return {'unix':clock.wall(),'boot_id':'boot','errors':{},'active_alerts':{},
     'gpu':{'gpu_C':29.,'hbm_C':28.,'power_limit_W':150.,'ecc_mode':'Enabled','ecc_single':0,'ecc_double':0,'replays':0,'used_MiB':0},
     'cpu':{'CPU':28.},'host_counters':{'ECC':0},
     'bmc':{'observed_unix':clock.wall(),'peripheral_C':30.,'pch_C':40.,'dimms_C':{str(i):30. for i in range(16)},
      'fan_mode':'00','fan_percent':[40.]*4,'fan_rpm':{name:700. for name in ['FAN1','FAN3','FAN5','FANB']},'bad_sensor_status':{}}}

class ProbeTests(unittest.TestCase):
    def run_case(self,actions,mutate=None,host=None,sleep=None,emit=None):
        self.clock=Clock();self.calls=0;self.events=[];self.host_checks=0
        def sample():
            action=actions[self.calls];self.calls+=1
            if action=='timeout':
                self.clock.t+=5
                raise subprocess.TimeoutExpired(['nvidia-smi'],5)
            if callable(action):return action(self.clock)
            if isinstance(action,Exception):raise action
            return copy.deepcopy(action)
        def snapshot():
            snap=healthy(self.clock)
            if mutate:mutate(snap)
            return snap
        def check_host():
            self.host_checks+=1
            if host:host()
        def event(e):
            self.events.append(e)
            if emit:emit(e,self.clock)
        def nap(s):
            self.clock.sleep(s)
            if sleep:sleep(self.clock)
        self.probe=g.GpuProbe(sample,initial(),{'ECC':0},'boot',check_host,
          cancelled=lambda:self.clock.stopped,emit=event,snapshot=snapshot,
          monotonic=self.clock.mono,wall=self.clock.wall,sleep=nap)
        return self.probe.read()
    def test_success_without_retry(self):
        self.assertEqual(self.run_case([initial()]),initial());self.assertEqual(self.calls,1);self.assertEqual(self.events,[])
    def test_timeout_recovers(self):
        self.run_case(['timeout',initial()]);self.assertEqual(self.calls,2);self.assertEqual(self.probe.stats['recovered_queries'],1)
        self.assertEqual([e['kind'] for e in self.events],['timeout','retry','recovered']);self.assertGreaterEqual(self.host_checks,3)
    def test_two_timeouts_recover(self):
        self.run_case(['timeout','timeout',initial()]);self.assertEqual(self.calls,3);self.assertLessEqual(self.clock.t,20)
    def test_three_timeouts_stop(self):
        with self.assertRaisesRegex(RuntimeError,'bounded retries'):self.run_case(['timeout']*3)
        self.assertEqual(self.calls,3);self.assertLessEqual(self.clock.t,20)
    def test_exit_error_not_retried(self):
        with self.assertRaises(subprocess.CalledProcessError):self.run_case([subprocess.CalledProcessError(1,['nvidia-smi'])])
        self.assertEqual(self.calls,1)
    def test_parse_error_not_retried(self):
        with self.assertRaises(ValueError):self.run_case([ValueError('bad XML')])
        self.assertEqual(self.calls,1)
    def test_malformed_direct_gpu(self):
        for field,value in [('gpu_C',None),('hbm_C',float('nan')),('limit_W',float('inf')),('ecc_single',True)]:
            with self.subTest(field=field):
                v=initial();v[field]=value
                with self.assertRaises(RuntimeError):self.run_case([v])
                self.assertEqual(self.calls,1)
    def test_actual_gpu_fault_is_not_retried(self):
        for field,value in [('gpu_C',60),('hbm_C',65),('limit_W',200),('ecc','Disabled'),('ecc_single',1),('ecc_double',1),('replays',1)]:
            for prefix in [[],['timeout']]:
                with self.subTest(field=field,prefix=prefix):
                    v=initial();v[field]=value
                    with self.assertRaises(RuntimeError):self.run_case(prefix+[v])
                    self.assertEqual(self.calls,len(prefix)+1)
    def test_bad_independent_snapshot_blocks_retry(self):
        mutations=[lambda s:s.update(unix=0),lambda s:s.update(unix=9999),lambda s:s.update(boot_id='old'),
          lambda s:s.update(errors={'gpu':'timeout'}),lambda s:s.pop('errors'),lambda s:s.update(active_alerts={'gpu_hot':{}}),
          lambda s:s.pop('gpu'),lambda s:s['gpu'].update(gpu_C=60),lambda s:s['gpu'].update(hbm_C=65),
          lambda s:s['gpu'].update(ecc_single=1),lambda s:s['gpu'].update(ecc_double=1),lambda s:s['gpu'].update(replays=1),
          lambda s:s['gpu'].update(power_limit_W=200),lambda s:s['gpu'].update(gpu_C=float('nan')),
          lambda s:s.update(cpu={}),lambda s:s['cpu'].update(CPU=60),lambda s:s['cpu'].update(CPU=None),
          lambda s:s.update(host_counters={'ECC':1}),lambda s:s.update(host_counters={}),
          lambda s:s['bmc'].update(observed_unix=0),lambda s:s['bmc'].update(observed_unix=9999),
          lambda s:s['bmc'].update(pch_C=55),lambda s:s['bmc'].update(peripheral_C=45),
          lambda s:s['bmc']['dimms_C'].update({'0':70}),lambda s:s['bmc']['dimms_C'].pop('0'),
          lambda s:s['bmc'].update(fan_mode='01'),lambda s:s['bmc'].update(fan_percent=[100]*4),
          lambda s:s['bmc'].update(fan_percent=[]),lambda s:s['bmc']['fan_rpm'].update(FANB=0),
          lambda s:s['bmc']['fan_rpm'].pop('FANB'),
          lambda s:s['bmc'].update(bad_sensor_status={'FAN1':'bad'})]
        for i,mutate in enumerate(mutations):
            with self.subTest(case=i):
                with self.assertRaises((RuntimeError,KeyError)):self.run_case(['timeout',initial()],mutate=mutate)
                self.assertEqual(self.calls,1)
    def test_host_fault_blocks_retry(self):
        def fault():raise RuntimeError('CPU thermal cutoff')
        with self.assertRaisesRegex(RuntimeError,'CPU thermal'):self.run_case(['timeout',initial()],host=fault)
        self.assertEqual(self.calls,1)
    def test_cancel_during_query(self):
        def cancel(clock):
            clock.stopped=True;raise subprocess.TimeoutExpired(['nvidia-smi'],5)
        with self.assertRaises(g.ProbeCancelled):self.run_case([cancel])
        self.assertEqual(self.events,[])
    def test_cancel_during_gap(self):
        with self.assertRaises(g.ProbeCancelled):self.run_case(['timeout',initial()],sleep=lambda c:setattr(c,'stopped',True))
        self.assertEqual(self.calls,1)
    def test_monitor_changes_during_gap(self):
        def mutate(s):
            if self.clock.t>=6:s['gpu']['ecc_double']=1
        with self.assertRaisesRegex(RuntimeError,'counter changed'):self.run_case(['timeout',initial()],mutate=mutate)
        self.assertEqual(self.calls,1)
    def test_slow_host_check_exhausts_budget(self):
        def slow():self.clock.t+=6
        with self.assertRaisesRegex(RuntimeError,'budget'):self.run_case(['timeout',initial()],host=slow)
        self.assertEqual(self.calls,1)
    def test_late_query_success_rejected(self):
        def late(clock):clock.t+=21;return initial()
        with self.assertRaisesRegex(RuntimeError,'budget'):self.run_case([late])
    def test_late_recovery_validation_rejected(self):
        def slow_final_check():
            if self.host_checks==3:self.clock.t+=21
        with self.assertRaisesRegex(RuntimeError,'budget'):self.run_case(['timeout',initial()],host=slow_final_check)
    def test_observed_fault_wins_over_cancel(self):
        def fault(clock):clock.stopped=True;return dict(initial(),gpu_C=70)
        with self.assertRaisesRegex(RuntimeError,'thermal cutoff'):self.run_case([fault])
    def test_non_timeout_error_wins_over_cancel(self):
        def fault(clock):
            clock.stopped=True;raise ValueError('invalid GPU data')
        with self.assertRaisesRegex(ValueError,'invalid GPU data'):self.run_case([fault])

if __name__=='__main__':unittest.main(verbosity=2)
