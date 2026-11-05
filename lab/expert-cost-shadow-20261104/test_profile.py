"""Exercise the config transaction on isolated temporary files, never the API config."""
from pathlib import Path
from unittest import mock
import contextlib,io,json,os,tempfile,unittest
import observer_profile as m

@unittest.skipUnless(os.geteuid()==0,'root-owned private state fixture requires sudo')
class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='shadow-profile-test-');self.root=Path(self.temp.name)
        self.original=b'{"environment":{"LD_LIBRARY_PATH":"/example/runtime"},"host":"127.0.0.1","port":12345}\n'
        config=self.root/'server-config.json';config.write_bytes(self.original);private=self.root/'private'
        self.patches=[mock.patch.object(m,n,v) for n,v in [('R',self.root),('CONFIG',config),('PRIVATE',private),('STATE',private/'profile.json'),('BASELINE_SHA',m.sha(self.original))]]
        for p in self.patches:p.start()
        library=self.root/'library.source';library.write_bytes(b'test library');self.library_sha=m.sha(library.read_bytes())
        self.build=mock.patch.object(m,'verified_build',return_value={'sha256':self.library_sha,'library':str(library)});self.build.start()
        self.idle=mock.patch.object(m,'inactive');self.inactive=self.idle.start()
    def tearDown(self):
        self.idle.stop();self.build.stop()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def run_action(self,action,apply=True):
        with mock.patch('sys.argv',['observer_profile.py',action]+(['--apply'] if apply else [])),contextlib.redirect_stdout(io.StringIO()):m.main()
    def test_dry_run_and_exact_roundtrip(self):
        self.run_action('enable',False);self.assertEqual(m.CONFIG.read_bytes(),self.original);self.assertFalse(m.STATE.exists());self.inactive.assert_not_called()
        self.run_action('enable');cfg=json.loads(m.CONFIG.read_text());self.assertEqual(cfg['environment']['LD_LIBRARY_PATH'],str(m.PRIVATE/('runtime-'+self.library_sha))+':/example/runtime')
        self.assertTrue(m.STATE.exists());self.assertEqual(m.STATE.stat().st_mode&0o777,0o600)
        self.run_action('restore');self.assertEqual(m.CONFIG.read_bytes(),self.original);self.assertFalse(m.STATE.exists());self.assertEqual(self.inactive.call_count,2)
    def test_active_api_prevents_config_write(self):
        self.inactive.side_effect=RuntimeError('active');
        with self.assertRaises(RuntimeError):self.run_action('enable')
        self.assertEqual(m.CONFIG.read_bytes(),self.original);self.assertFalse(m.STATE.exists())
    def test_restore_preserves_later_user_changes(self):
        self.run_action('enable');changed=m.CONFIG.read_bytes()+b' ';m.CONFIG.write_bytes(changed)
        with self.assertRaises(RuntimeError):self.run_action('restore')
        self.assertEqual(m.CONFIG.read_bytes(),changed);self.assertTrue(m.STATE.exists())
    def test_forged_original_rejected(self):
        self.run_action('enable');state=json.loads(m.STATE.read_text());state['original_utf8']='{}';state['original_sha256']=m.sha(b'{}');m.STATE.write_text(json.dumps(state));current=m.CONFIG.read_bytes()
        with self.assertRaises(RuntimeError):self.run_action('restore')
        self.assertEqual(m.CONFIG.read_bytes(),current)
    def test_state_symlink_rejected(self):
        self.run_action('enable');target=m.PRIVATE/'saved';m.STATE.rename(target);m.STATE.symlink_to(target)
        with self.assertRaises(RuntimeError):self.run_action('restore')
if __name__=='__main__':unittest.main()
