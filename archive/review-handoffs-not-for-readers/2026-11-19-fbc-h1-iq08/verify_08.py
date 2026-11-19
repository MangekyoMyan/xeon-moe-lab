from pathlib import Path
import hashlib,json,math,array
R=Path(__file__).resolve().parent;D=R/'cpu08'
REFERENCE_CAPTURE_ROOT=None  # Path-only relocation for offline public replay.
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(run):
    j=read(run/'RUN.json');spec=read(run/'SPEC.json');h=read(run/'health.json');assert j['passed'] and j['spec']==spec and h['status']=='exited' and h['child_returncode']==0
    assert [(b['cell'],b['threads']) for b in j['blocks']]==[(c,12) for c in spec['cells']]
    capfile=Path(REFERENCE_CAPTURE_ROOT)/'CAPTURE.json' if REFERENCE_CAPTURE_ROOT else Path(spec['capture_json']);cap=read(capfile)
    for b in j['blocks']:
        raw=(run/b['output_file']).read_bytes();assert raw==(capfile.parent/cap['cells'][b['cell']]['down_file']).read_bytes();a=array.array('f');a.frombytes(raw);assert len(a)==30720 and all(math.isfinite(v) for v in a)
        assert sorted(b['worker_receipt_after_timing']['CPU'])==list(range(12)) and b['planner_work_size']>0
        assert len(b['samples'])==spec['samples'] and all(s['calls']==spec['loops'] and s['end_ns']>s['begin_ns'] for s in b['samples'])
        if spec['stage']=='quality':
            path=b['path_check'];hot=spec['mode']=='candidate' and b['cell']!=6
            assert path['pack_blocks']==(94 if hot else 0)
            expected=[0,0,0] if not hot else [23040,25600,0] if b['cell']==1 else [0,27648,30720]
            assert [x['dot_rows_nrc1'] for x in path['types']]==expected
            if hot:
                assert len(b['direct_min128_checks'])==3
                for check in b['direct_min128_checks']:assert len(check['modes'])==4 and all(x['equal'] for x in check['modes'])
        else:assert 'path_check' not in b and 'direct_min128_checks' not in b
    return j
def gate():
    state=read(D/'STATE.json');runs=[Path(state['runs']['quality-'+m]['path']) for m in ['control','candidate']];js=[verify(p) for p in runs]
    for a,b in zip(js[0]['blocks'],js[1]['blocks']):assert a['weights']==b['weights'] and (runs[0]/a['output_file']).read_bytes()==(runs[1]/b['output_file']).read_bytes()
    out=D/'QUALITY_GATE.json';assert not out.exists();out.write_text(json.dumps(dict(passed=True,cells=[1,6,5],threads=12,paths_checked=True,min128_direct_checks=24,runs=[dict(path=str(p),manifest_sha256=sha(p/'RESULTS_SHA256.json')) for p in runs]),indent=2)+'\n')
