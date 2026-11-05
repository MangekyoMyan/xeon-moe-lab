import errno
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import xeon_imc as imc
import xeon_dashboard as dashboard


class ImcTests(unittest.TestCase):
    def test_sysfs_definitions_and_socket_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pmus=root/'pmus';cpus=root/'cpus'
            for cpu,package in [(0,0),(6,1)]:
                p=cpus/f'cpu{cpu}/topology';p.mkdir(parents=True)
                (p/'physical_package_id').write_text(str(package))
            for channel in range(4):
                p=pmus/f'uncore_imc_{channel}'
                (p/'format').mkdir(parents=True);(p/'events').mkdir()
                (p/'type').write_text(str(12+channel));(p/'cpumask').write_text('0,6')
                (p/'format/event').write_text('config:0-7');(p/'format/umask').write_text('config:8-15')
                for name,mask in [('read','0x03'),('write','0x0c')]:
                    e=p/'events'/('cas_count_'+name)
                    e.write_text('event=0x04,umask='+mask)
                    e.with_name(e.name+'.unit').write_text('MiB')
                    e.with_name(e.name+'.scale').write_text('6.103515625e-5')
            with patch.object(imc,'PMU_ROOT',pmus),patch.object(imc,'CPU_ROOT',cpus):
                events=imc.discover()
            self.assertEqual(len(events),16)
            self.assertEqual({e['socket'] for e in events},{0,1})
            self.assertEqual({e['bytes_per_count'] for e in events},{64})
            self.assertEqual({e['config'] for e in events if e['direction']=='read'},{0x304})
            self.assertEqual({e['config'] for e in events if e['direction']=='write'},{0xc04})

    def test_multiplexing_uses_interval_times_not_lifetime_ratio(self):
        # Lifetime ratios differ from this interval's 50 percent scheduling.
        a=(100,10000,9000,2);b=(300,12000,10000,4)
        value=imc.interval_rate(a,b,64)
        self.assertEqual(value['bytes_per_second'],12800)
        self.assertEqual(value['running_fraction'],.5)

    def test_unavailable_or_reset_counter_is_not_zero_bandwidth(self):
        a=(100,1000,1000,2)
        for b in [(200,2000,1000,3),(90,2000,2000,3),(200,2000,3000,3)]:
            with self.subTest(record=b),self.assertRaises(ValueError):imc.interval_rate(a,b,64)

    def test_per_event_correction_precedes_socket_sum(self):
        rows=[]
        for socket,channel,direction,rate,cov in [(0,0,'read',100,.5),(0,1,'read',300,1),(0,0,'write',20,1),(1,0,'read',200,1),(1,0,'write',40,1)]:
            rows.append(({'socket':socket,'pmu':str(channel),'direction':direction},
                         {'bytes_per_second':rate,'running_fraction':cov,'interval_seconds':2}))
        result=imc.aggregate(rows)
        self.assertEqual(result['read_bytes_per_second'],600)
        self.assertEqual(result['write_bytes_per_second'],60)
        self.assertEqual(result['sockets']['0']['read_bytes_per_second'],400)
        self.assertEqual(result['running_fraction_min'],.5)
        self.assertTrue(result['multiplexed'])

    def test_failed_open_closes_already_opened_descriptors(self):
        with patch.object(imc,'discover',return_value=[{},{}]), \
             patch.object(imc,'open_counter',side_effect=[42,OSError(errno.EPERM,'not allowed')]), \
             patch.object(imc.os,'close') as close:
            reader=imc.ImcReader();value=reader.sample();reader.close()
            self.assertEqual(value['status'],'unavailable');close.assert_called_once_with(42)

    def test_disabled_does_not_open_counters(self):
        with patch.object(imc,'open_counter') as open_counter:
            self.assertEqual(imc.ImcReader(False).sample()['status'],'disabled')
            open_counter.assert_not_called()

    def test_cpu_ranges(self):
        self.assertEqual(imc.cpu_list('0-2,6,6'),[0,1,2,6])
        with self.assertRaises(ValueError):imc.cpu_list('6-2')

    def test_gpu_window_uses_recent_valid_existing_samples(self):
        samples=[{'unix':t,'gpu':{'gpu_util':v}} for t,v in [(60,100),(80,0),(90,90),(101,80),(99,float('nan'))]]
        result=dashboard.recent_gpu_window({'recent_samples':samples},100)
        self.assertEqual(result['sample_count'],2)
        self.assertEqual(result['mean_percent'],45)
        self.assertEqual(result['max_percent'],90)
        self.assertIsNone(dashboard.recent_gpu_window({'recent_samples':samples},200))

    def test_renderer_labels_system_scope_and_multiplexing(self):
        s={'unix':100,'watch':{},'process':None,'health':{},'machine':{},'api':{},'performance':{},'download':None,
           'model':'test','profile':'expert固定: GPU 2層 / CPU 38層',
           'memory_bandwidth':{'status':'ok','read_bytes_per_second':2e9,'write_bytes_per_second':1e9,
               'sockets':{'0':{'read_bytes_per_second':2e9,'write_bytes_per_second':1e9}},
               'multiplexed':True,'running_fraction_min':.5,'interval_seconds':2}}
        text='\n'.join(dashboard.render(s))
        self.assertIn('合計 3.00 GB/s（全体実測）',text)
        self.assertIn('多重化',text)
        self.assertIn('expertの動的入替なし',text)


if __name__=='__main__':unittest.main()
