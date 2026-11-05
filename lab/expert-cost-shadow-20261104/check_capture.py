from pathlib import Path
import json,os,subprocess,tempfile,stat
R=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='expert-shadow-check-',dir=R) as name:
    p=Path(name);exe=R/'test_capture';base=os.environ.copy();base.pop('XEON_SHADOW_LOG',None)
    off=subprocess.run([str(exe)],env=base,capture_output=True,text=True,check=True,timeout=10)
    env=dict(base,XEON_SHADOW_LOG=str(p/'normal.jsonl'))
    on=subprocess.run([str(exe)],env=env,capture_output=True,text=True,check=True,timeout=10)
    assert off.stdout==on.stdout
    rows=[json.loads(x) for x in (p/'normal.jsonl').read_text().splitlines()];events=[x for x in rows if x['kind']=='route_observation'];footer=rows[-1]
    assert len(events)==6 and footer['final'] and footer['written']==6
    assert footer['attempted']==footer['accepted']+footer['enqueue_dropped']
    assert footer['accepted']==footer['written']+footer['write_dropped']
    assert events[0]['route_epoch']==events[1]['route_epoch']
    assert events[0]['shared_fetch_wait_ms']==.01 and events[1]['shared_fetch_wait_ms']==0
    assert events[2]['route_epoch']!=events[1]['route_epoch']
    assert events[3]['histogram'][6]==1 and events[3]['histogram'][7]==0
    assert events[4]['histogram'][0]==1 and events[4]['histogram'][6]==0
    assert events[4]['route_epoch']!=events[3]['route_epoch'] and not events[4]['route_refreshed']
    assert events[5]['histogram'][7]==1 and events[5]['histogram'][0]==0 and events[5]['route_epoch']!=events[4]['route_epoch']
    assert footer['skipped'][4]==1 and footer['skipped'][0]==1
    assert stat.S_IMODE((p/'normal.jsonl').stat().st_mode)==0o600
    victim=p/'victim';victim.write_text('untouched');link=p/'link';link.symlink_to(victim)
    for path in [link,victim,p/'missing-dir/log']:
        bad=dict(base,XEON_SHADOW_LOG=str(path));result=subprocess.run([str(exe)],env=bad,capture_output=True,text=True,check=True,timeout=5)
        assert result.stdout==off.stdout
    assert victim.read_text()=='untouched'
    fifo=p/'fifo';os.mkfifo(fifo);subprocess.run([str(exe)],env=dict(base,XEON_SHADOW_LOG=str(fifo)),capture_output=True,text=True,check=True,timeout=5)
    subprocess.run([str(exe),'cap'],env=dict(base,XEON_SHADOW_LOG=str(p/'capped.jsonl')),capture_output=True,text=True,check=True,timeout=10)
    cap=[json.loads(x) for x in (p/'capped.jsonl').read_text().splitlines()];end=cap[-1]
    assert end['final'] and end['written']==2048 and (p/'capped.jsonl').stat().st_size<=8*1024**2
    assert end['attempted']==end['accepted']+end['enqueue_dropped'] and end['accepted']==end['written']+end['write_dropped']
    result={'normal_observations':len(events),'strided_ids':True,'refresh_reuse_and_invalid_recovery':True,'excluded_refresh_invalidation':True,'cross_thread_refresh_invalidation':True,'loss_invariants':True,
      'existing_file_symlink_fifo_and_bad_path_fail_open':True,'file_permissions':'0600','record_cap':2048,'no_assignment_changes':all(not e['assignment_changed'] for e in events)}
    (R/'CAPTURE_FIXTURE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
