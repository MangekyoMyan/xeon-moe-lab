from pathlib import Path
import array,hashlib,json,math,re
R=Path(__file__).resolve().parent;F=R/'fbc'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_run(path):
    j=read(path/'RUN.json');s=read(path/'SPEC.json');h=read(path/'health.json');assert h['status']=='exited' and h['child_returncode']==0 and j['passed']
    assert j['spec']==s and j['has_bank']==(s['mode']!='F') and j['uses_custom']==(s['mode']=='C')
    assert j['cache_allocated_bytes']==127401984000
    expected=read(path/'PROVIDER_EXPECTED.json');assert j['providers']==expected['providers'] and j['active_backend_providers']==expected['active_backend_providers']
    cases={c['case_id']:c for c in read(F/'CASES.json')['cases']};layers=set(read(F/'DESIGN.json')['selected_layers'])
    def delta(a,b,n):
        if s['mode']=='F':assert a is None and b is None;return
        assert a['ready'] and b['ready'] and a['native_bytes']==b['native_bytes']==475453440
        for l in range(40):
            want=n if s['mode']=='C' and l in layers else 0
            assert b['calls'][l]-a['calls'][l]==want
            assert [v-u for u,v in zip(a['executed_H'][l],b['executed_H'][l])]==[want,0,0,0,0,0,0]
        if s['mode']=='B':assert a['custom_nodes_built']==b['custom_nodes_built']==0 and not any(b['calls'])
    delta(j['preparation']['before'],j['preparation']['after'],4)
    assert [b['case_id'] for b in j['blocks']]==s['cases']
    for b in j['blocks']:
        c=cases[b['case_id']];assert b['prefix_ids']==c['prefix_ids'] and b['input_tokens']==c['used_T1_ids'] and b['next_targets']==c['next_targets_prefix_and64T1']
        assert b['positions']==list(range(c['prefix_count'],c['prefix_count']+64))
        assert len(b['start_ns'])==len(b['output_ready_ns'])==64 and b['status']==[0]*64
        assert all(e>t for t,e in zip(b['start_ns'],b['output_ready_ns']))
        delta(b['before_prefix'],b['before_T1'],0);delta(b['before_T1'],b['after'],64)
        if s['stage']=='quality':
            raw=(path/b['file']).read_bytes();assert len(raw)==65*129280*4;v=array.array('f');v.frombytes(raw);assert all(math.isfinite(x) for x in v)
            assert (path/(b['case_id']+'-graph-pre.json')).is_file() and (path/(b['case_id']+'-graph-post.json')).is_file()
        else:
            raw=(path/b['last_file']).read_bytes();q=read(F/'QUALITY_GATE.json')['case_hashes'][b['case_id']]['last_row_sha256'];assert hashlib.sha256(raw).hexdigest()==q
    return j
def gate():
    state=read(F/'STATE.json');paths={a:Path(state['runs']['quality-'+a]['path']) for a in 'FBC'}
    js={a:verify_run(p) for a,p in paths.items()};results={};graphs={}
    for c in read(F/'CASES.json')['cases']:
        name=c['case_id'];data={a:(p/f'quality-{name}-{a}.f32').read_bytes() for a,p in paths.items()};assert data['F']==data['B']==data['C'],name+' quality mismatch'
        pre={a:read(p/(name+'-graph-pre.json')) for a,p in paths.items()};assert pre['F']==pre['B'],name+' original F/B DAG mismatch'
        post={a:read(p/(name+'-graph-post.json')) for a,p in paths.items()}
        blocks={}
        for a,p in paths.items():
            log=(p/'server.stderr.log').read_text();begin=f'FBC_TRACE_BEGIN case={name} arm={a}';end=f'FBC_TRACE_END case={name} arm={a}'
            assert log.count(begin)==log.count(end)==1;selected=log.split(begin,1)[1].split(end,1)[0]
            splits=re.findall(r'## SPLIT #(\d+): (\S+) # (\d+) inputs([^\n]*)',selected)
            assert splits,'missing actual scheduler debug split ledger'
            blocks[a]=dict(scheduled_split_headers=[dict(index=int(i),backend=b,inputs=int(n),printed_inputs=x.strip()) for i,b,n,x in splits],post_allocation_splits=post[a]['scheduler_splits'],trace_file=name+'-graph-post.json')
        results[name]=dict(all_65_FBC_byte_equal=True,sha256=hashlib.sha256(data['F']).hexdigest(),last_row_sha256=hashlib.sha256(data['F'][-129280*4:]).hexdigest())
        graphs[name]=dict(original_F_B_equal=True,original_node_counts={a:len(pre[a]['nodes']) for a in 'FBC'},placement=blocks,scope='Pre-original DAG;post-allocation/optimization metadata. Debugsplit listing before CUDAoptimizer and rounded/truncated labels; not hardware transfer byte/time measurement.')
    out=F/'QUALITY_GATE.json';assert not out.exists();out.write_text(json.dumps(dict(passed=True,case_hashes=results,graphs=graphs,runs={a:dict(path=str(p),manifest_sha256=sha(p/'RESULTS_SHA256.json')) for a,p in paths.items()}),indent=2)+'\n');print('FBC quality/DAG/placement gate passed',flush=True)
if __name__=='__main__':gate()
