import copy
import unittest
from recovery import verify_incident, reviewed_health

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.reference={'boot_id':'boot','started_unix':123,'finished_unix':456,'status':'stopped_on_error',
         'error':"TimeoutExpired(['nvidia-smi', '-i', 'GPU-test', '-q', '-x'], 5)",'child_returncode':-15,'pid':100}
        self.marker=(self.reference['error']+'\n').encode()
    def test_exact_incident(self):verify_incident(dict(self.reference),self.marker,self.reference,'boot')
    def test_unrelated_new_fault_refused(self):
        for key,value in [('pid',200),('finished_unix',457),('status','running'),('error','GPU thermal cutoff'),('child_returncode',-9)]:
            with self.subTest(key=key):
                h=dict(self.reference);h[key]=value
                with self.assertRaises(RuntimeError):verify_incident(h,self.marker,self.reference,'boot')
    def test_marker_and_boot_mismatch(self):
        with self.assertRaises(RuntimeError):verify_incident(self.reference,b'other\n',self.reference,'boot')
        with self.assertRaises(RuntimeError):verify_incident(self.reference,self.marker,self.reference,'rebooted')
    def test_non_timeout_reference_refused(self):
        h=dict(self.reference,error='GPU thermal cutoff')
        with self.assertRaises(RuntimeError):verify_incident(h,b'GPU thermal cutoff\n',h,'boot')
    def test_review_preserves_evidence_and_inactive_status(self):
        h=reviewed_health(self.reference,'backup',999)
        self.assertEqual(h['status'],'stop_reviewed');self.assertEqual(h['reviewed_error'],self.reference['error']);self.assertNotIn('error',h)
        self.assertEqual(self.reference['status'],'stopped_on_error');self.assertEqual(h['finished_unix'],456)

if __name__=='__main__':unittest.main(verbosity=2)
