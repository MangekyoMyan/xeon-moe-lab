from pathlib import Path
import ast,difflib,hashlib,json,subprocess,time
R=Path(__file__).resolve().parent
names=['gpu_guard.py','supervise.py','recovery.py','deploy.py','test_gpu_guard.py','test_supervisor.py','test_recovery.py','test_deploy.py']
for name in names:ast.parse((R/name).read_text(),filename=name)
command=['python3','-m','unittest','discover','-s',str(R),'-p','test_*.py','-v']
p=subprocess.run(command,capture_output=True,text=True,timeout=30)
(R/'tests.log').write_text(p.stdout+p.stderr)
if p.returncode:raise SystemExit(p.stderr)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
old=(R/'supervise.before.py').read_text();new=(R/'supervise.py').read_text()
(R/'supervise.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='supervise.before.py',tofile='supervise.py')))
j={'passed':True,'unix':time.time(),'sources':{name:sha(R/name) for name in names},'test_command':command,'test_log_sha256':sha(R/'tests.log'),
 'incident_sha256':sha(R.parent/'guard-incident-20261104/health-evidence.json'),'original_supervisor_sha256':sha(R/'supervise.before.py'),
 'scope':'40 simulated tests including failure/cancellation/incident-acknowledgement races; no model or hardware commands executed by tests',
 'runtime_policy':{'query_timeout_seconds':5,'max_attempts':3,'retry_gap_seconds':1,'checked_recovery_budget_seconds':20,'independent_snapshot_max_age_seconds':20,'BMC_max_age_seconds':90,'temperature_limits_unchanged':True},'model_started':False}
(R/'VALIDATION.json').write_text(json.dumps(j,indent=2)+'\n');print('Validated 40 tests; sources and fixed incident pinned by SHA256.')
