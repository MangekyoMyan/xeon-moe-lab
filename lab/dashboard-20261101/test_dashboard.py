import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

LAB=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(LAB))
import xeon_dashboard as d
from xeon_download_progress import write_progress


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.item={'name':'part/model.gguf','size':100,'sha256':'a'*64}
        self.manifest={'repository':'test/model','revision':'r1','files':[self.item]}
        self.path=self.root/'manifest.json';self.path.write_text(json.dumps(self.manifest))
        (self.root/'part').mkdir()
    def tearDown(self):self.tmp.cleanup()
    def snapshot(self,rates=None,now=1):return d.download_snapshot(self.path,self.root,rates or d.Rates(),now)

    def test_partial_full_size_is_not_verified(self):
        (self.root/'part/model.gguf.part').write_bytes(b'x'*100)
        s=self.snapshot();self.assertEqual(s['percent'],100)
        self.assertFalse(s['verified_record']);self.assertIn('検証未確認',s['state'])

    def test_rename_does_not_double_count_or_invent_network_speed(self):
        p=self.root/'part/model.gguf.part';p.write_bytes(b'x'*100)
        rates=d.Rates();self.snapshot(rates,1)
        p.rename(self.root/'part/model.gguf')
        s=self.snapshot(rates,3)
        self.assertEqual(s['done'],100);self.assertEqual(s['bytes_per_second'],0)
        (self.root/'download-verified.json').write_text(json.dumps(self.manifest))
        self.assertTrue(self.snapshot()['verified_record'])

    def test_resume_rate_counts_only_new_bytes(self):
        p=self.root/'part/model.gguf.part';p.write_bytes(b'x'*40)
        rates=d.Rates();self.assertIsNone(self.snapshot(rates,1)['bytes_per_second'])
        p.write_bytes(b'x'*60);s=self.snapshot(rates,3)
        self.assertEqual(s['bytes_per_second'],10);self.assertEqual(s['eta_seconds'],4)

    def test_sparse_partial_has_no_percentage_claim(self):
        self.item['size']=1024*1024
        self.path.write_text(json.dumps(self.manifest))
        with (self.root/'part/model.gguf.part').open('wb') as f:f.truncate(self.item['size'])
        self.assertIsNone(self.snapshot()['percent'])

    def test_manifest_cannot_escape_download_directory(self):
        self.item['name']='../secret';self.path.write_text(json.dumps(self.manifest))
        self.assertIsNone(self.snapshot())

    def test_aria2_reservation_is_not_claimed_as_downloaded(self):
        (self.root/'part/model.gguf').write_bytes(b'x'*100)
        (self.root/'part/model.gguf.aria2').write_bytes(b'reserved')
        self.assertIsNone(self.snapshot()['percent'])

    def test_sha_progress_is_distinct_from_download_completion(self):
        (self.root/'part/model.gguf.part').write_bytes(b'x'*100)
        self.assertTrue(write_progress(self.root,'test/model','r1','verifying',self.item,40))
        s=self.snapshot();self.assertTrue(s['live'])
        self.assertEqual(s['percent'],100);self.assertEqual(s['verification_percent'],40)
        self.assertFalse(s['verified_record'])

    def test_api_only_reads_metadata_and_drops_private_fields(self):
        key=self.root/'key';key.write_text('SENSITIVE_KEY')
        cfg=self.root/'config';cfg.write_text(json.dumps({'host':'127.0.0.1','port':8000,'api_key_file':str(key)}))
        calls=[]
        class Response(io.BytesIO):
            status=200
        class Opener:
            def open(self,req,timeout):
                calls.append((req.get_method(),req.full_url))
                value={'status':'ok'} if req.full_url.endswith('/health') else [{'id':0,'id_task':1,'is_processing':True,'prompt':'PRIVATE_PROMPT','generated':'PRIVATE_ANSWER','next_token':[{'n_decoded':3}]}]
                return Response(json.dumps(value).encode())
        with patch.object(d.urllib.request,'build_opener',return_value=Opener()):s=d.ApiReader(cfg).poll()
        self.assertEqual([x[0] for x in calls],['GET','GET'])
        self.assertEqual([x[1].rsplit('/',1)[-1] for x in calls],['health','slots'])
        text=json.dumps(s);self.assertNotIn('PRIVATE',text);self.assertNotIn('SENSITIVE_KEY',text)
        self.assertEqual(s['slots'][0]['generated'],3)

    def test_log_parser_keeps_numbers_not_text_and_resets_on_new_worker(self):
        tail=d.LogTail(self.root/'unused')
        tail.parse('98.15.124.551 I slot print_timing: id 0 | task 17 | prompt processing, n_tokens = 128, progress = 0.25, t = 14.1 s / 9.04 tokens per second PRIVATE_PROMPT',1000,10000)
        tail.parse('99.57.034.227 I slot print_timing: id 0 | task 17 | eval time = 439.37 ms / 3 tokens (219.68 ms per token, 4.55 tokens per second)',1000,10000)
        self.assertEqual(tail.values['progress']['percent'],25)
        self.assertEqual(tail.values['last_tg']['tps'],4.55)
        self.assertNotIn('PRIVATE',json.dumps(tail.values))
        tail.parse('0.01.000.000 I loading model',9000,10000)
        self.assertEqual(tail.values,{})

    def test_rate_reset_and_terminal_control_characters(self):
        r=d.Rates();self.assertIsNone(r.sample('x',100,1))
        self.assertEqual(r.sample('x',120,3),10)
        self.assertIsNone(r.sample('x',1,4))
        self.assertNotIn('\x1b',d.clip('bad\x1b[2J\ntext',80))

    def test_progress_write_failure_does_not_abort_download(self):
        with patch('xeon_download_progress.tempfile.NamedTemporaryFile',side_effect=PermissionError):
            self.assertFalse(write_progress(self.root,'a','b','starting'))

    def test_process_identity_survives_restricted_io_file(self):
        original=Path.read_text
        def read(path,*args,**kwargs):
            if path==Path(f'/proc/{os.getpid()}/io'):raise PermissionError
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',read):s=d.proc_snapshot(os.getpid())
        self.assertIsNotNone(s);self.assertGreater(s['start_ticks'],0)
        self.assertIsNone(s['read_bytes'])

    def test_unavailable_capacity_is_not_zero(self):
        self.assertEqual(d.fmt(d.scaled(None,2**30),'GiB'),'--')

    def test_original_downloaders_still_check_hashes(self):
        for name in ['download_glm_air_q8','download_deepseek41_orca_mix256']:
            spec=importlib.util.spec_from_file_location(name,LAB/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            p=self.root/(name+'.gguf');p.write_bytes(b'fixture')
            item={'name':p.name,'size':7,'sha256':hashlib.sha256(b'fixture').hexdigest()}
            module.DEST=self.root;module.PROGRESS_OWNED=True
            with patch('sys.stdout',new=io.StringIO()):module.check_sha256(p,item)
            item['sha256']='0'*64
            with patch('sys.stdout',new=io.StringIO()),self.assertRaises(RuntimeError):module.check_sha256(p,item)


if __name__=='__main__':unittest.main()
