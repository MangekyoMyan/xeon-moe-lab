import copy
import json
from pathlib import Path
import tempfile
import unittest
from monitor import Detector, atomic, parse_bmc
from notify import cycle

C = json.loads((Path(__file__).parent / 'config.json').read_text())

def healthy(t=100):
    return {'unix': t, 'boot_id': 'boot-one', 'errors': {}, 'cpu': {'cpu0': 30.0},
            'gpu': {'gpu_C': 30, 'hbm_C': 31, 'ecc_mode': 'Enabled', 'ecc_single': 0, 'ecc_double': 0, 'replays': 0},
            'host_counters': {'edac:ce': 0},
            'bmc': {'dimms_C': {f'DIMM{i}': 30 for i in range(16)}, 'fan_rpm': {'FAN1': 2300},
                    'fan_percent': [40, 50], 'fan_mode': '00', 'pch_C': 43, 'peripheral_C': 31}}

class DetectorTests(unittest.TestCase):
    def test_quiet_healthy(self):
        d = Detector(C)
        for t in range(100, 200, 10):
            self.assertEqual(d.update(healthy(t)), ([], []))

    def test_threshold_persistent_and_recovery(self):
        d = Detector(C)
        s = healthy(); s['cpu']['cpu0'] = 60
        self.assertEqual(len(d.update(s)[0]), 1)
        d = Detector(C, json.loads(json.dumps(d.s)))
        s['unix'] += 10
        self.assertFalse(d.update(s)[0])
        s['cpu']['cpu0'] = 59
        for i in range(4):
            self.assertEqual(d.update(s), ([], []))
        for t in range(120, 150, 10):
            result = d.update(healthy(t))
        self.assertEqual(result[1], ['cpu_C'])
        self.assertFalse(d.s['alarms']['cpu_C']['active'])

    def test_flap_cooldown(self):
        d = Detector(C)
        s = healthy(); s['cpu']['cpu0'] = 61
        d.update(s)
        for t in [110, 120, 130]: d.update(healthy(t))
        s['unix'] = 140
        self.assertFalse(d.update(s)[0])
        s['unix'] = 1901
        self.assertEqual(len(d.update(s)[0]), 1)

    def test_dimm_hbm_group(self):
        d = Detector(C); s = healthy()
        s['bmc']['dimms_C']['DIMM3'] = 70
        s['gpu']['hbm_C'] = 65
        events, _ = d.update(s)
        self.assertEqual({v['key'] for v in events}, {'dimm:DIMM3', 'hbm_C'})

    def test_missing_sensor_no_false_normal(self):
        d = Detector(C); s = healthy()
        del s['bmc']['dimms_C']['DIMM0']
        self.assertFalse(d.update(s)[0]); self.assertFalse(d.update(s)[0])
        self.assertEqual(d.update(s)[0][0]['key'], 'sensor:dimm_temperature')

    def test_gpu_missing_and_fan_limit(self):
        d = Detector(C); s = healthy(); s['gpu']['hbm_C'] = None
        for _ in range(2):self.assertFalse(d.update(s)[0])
        self.assertEqual(d.update(s)[0][0]['key'], 'sensor:gpu_temperature')
        s = healthy(200); s['bmc']['fan_percent'] = [70]
        self.assertFalse(d.update(s)[0])
        s['bmc']['fan_percent'] = [70.1]
        self.assertEqual(d.update(s)[0][0]['key'], 'fan_percent')

    def test_counter_restart_and_rate_limit(self):
        d = Detector(C); d.update(healthy())
        s = healthy(110); s['host_counters']['edac:ce'] = 1
        self.assertEqual(d.update(s)[0][0]['delta'], 1)
        d = Detector(C, json.loads(json.dumps(d.s)))
        self.assertFalse(d.update(s)[0])
        s['unix'] = 120; s['host_counters']['edac:ce'] = 2
        self.assertFalse(d.update(s)[0])
        s['unix'] = 1911; s['host_counters']['edac:ce'] = 3
        self.assertEqual(len(d.update(s)[0]), 1)

    def test_api_guard_captures_brief_overheat(self):
        d = Detector(C); s = healthy()
        s['api_guard'] = {'boot_id': 'old', 'status': 'stopped_on_error', 'error': 'GPU cutoff'}
        self.assertFalse(d.update(s)[0])
        s['api_guard']['boot_id'] = 'boot-one'
        self.assertEqual(d.update(s)[0][0]['key'], 'api_guard_stopped')

    def test_bmc_na_and_temperatures(self):
        rows = parse_bmc('P1-DIMMA1 TEMP | 31.000 | degrees C | ok\nFAN2 | na | | na')
        self.assertEqual(rows['P1-DIMMA1 TEMP']['value'], 31)
        self.assertIsNone(rows['FAN2']['value'])
    def test_additional_api_guard(self):
        d=Detector(C);s=healthy()
        s['additional_api_guards']={'new-api':{'boot_id':'boot-one','status':'stopped_on_error','error':'thermal cutoff'}}
        self.assertEqual(d.update(s)[0][0]['key'],'api_guard:new-api')

class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'monitor'; self.out = Path(self.tmp.name) / 'notify'
        (self.root/'outbox').mkdir(parents=True); self.out.mkdir()
        atomic(self.root/'latest.json', healthy(100))
        self.state = {'acked': {}, 'next_attempt': 0}
        self.calls = []
    def event(self, name):
        atomic(self.root/'outbox'/f'{name}.json', {'id': name, 'unix':100, 'kind':'hardware_alert',
            'items':[{'description':'GPU温度','value':60,'limit':60}]})
    def sender(self, c, message):
        self.calls.append(message)
        return {'accepted':True,'queue_id':'test-receipt'}
    def test_healthy_never_calls_codex(self):
        self.assertFalse(cycle(C,self.root,self.out,self.state,self.sender,100))
        self.assertFalse(self.calls)
    def test_offline_retry_ack_restart(self):
        self.event('a')
        offline=lambda *_:{'accepted':False,'reason':'offline'}
        self.assertFalse(cycle(C,self.root,self.out,self.state,offline,100))
        self.assertFalse(self.state['acked'])
        self.assertFalse(cycle(C,self.root,self.out,self.state,self.sender,200))
        atomic(self.root/'latest.json', healthy(401))
        self.assertTrue(cycle(C,self.root,self.out,self.state,self.sender,401))
        state=json.loads((self.out/'state.json').read_text())
        self.assertFalse(cycle(C,self.root,self.out,state,self.sender,402))
        self.assertEqual(len(self.calls),1)
    def test_backlog_is_one_notification(self):
        for i in range(30):self.event(str(i))
        self.assertTrue(cycle(C,self.root,self.out,self.state,self.sender,100))
        self.assertEqual(len(self.calls),1);self.assertEqual(len(self.state['acked']),30)
    def test_stale_monitor_once(self):
        self.assertFalse(cycle(C,self.root,self.out,self.state,self.sender,100))
        self.assertTrue(cycle(C,self.root,self.out,self.state,self.sender,221))
        self.assertFalse(cycle(C,self.root,self.out,self.state,self.sender,600))
        self.assertEqual(len(self.calls),1)
    def test_startup_old_snapshot_has_grace(self):
        self.assertFalse(cycle(C,self.root,self.out,self.state,self.sender,1000))
        atomic(self.root/'latest.json', healthy(1001))
        self.assertFalse(cycle(C,self.root,self.out,self.state,self.sender,1002))

if __name__=='__main__':unittest.main()
