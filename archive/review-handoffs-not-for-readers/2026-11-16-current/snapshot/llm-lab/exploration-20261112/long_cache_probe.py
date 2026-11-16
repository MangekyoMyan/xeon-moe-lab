"""Bounded native-token continuation smoke; synthetic data, no service changes."""
from pathlib import Path
import argparse,fcntl,hashlib,http.client,importlib.util,json,os,socket,threading,time
R=Path(__file__).resolve().parent;CONFIG=R.parent/'deepseek41-bringup-20261101/server-config.json'
spec=importlib.util.spec_from_file_location('probe_helpers',R.parent/'postcompact-20261103/live_probe.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--context',type=int,choices=[16384,65536],default=16384);args=parser.parse_args()
    assert args.label.replace('-','').isalnum()
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();config=json.loads(raw)
    key=Path(config['api_key_file']).read_text().strip()
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    active=[None];active_lock=threading.Lock();failures=[];request_events=[]
    def interrupt():
        with active_lock:
            sock=active[0]
        if sock:
            try:sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
    def query(route,body=None,deadline=15,cleanup_read=False):
        assert not cleanup_read or (body is None and route=='/slots')
        if failures and not cleanup_read:raise RuntimeError(failures[0])
        connection=http.client.HTTPConnection(config['host'],config['port'],timeout=min(10,deadline))
        expired=threading.Event();held_socket=[None];response=None
        event={'route':route,'deadline_seconds':deadline,'cleanup_read':cleanup_read,'started':time.monotonic()}
        request_events.append(event)
        def expire():
            expired.set()
            sock=held_socket[0]
            if sock:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
        timer=threading.Timer(deadline,expire);timer.daemon=True
        started=time.monotonic()
        timer.start()
        try:
            if (failures and not cleanup_read) or expired.is_set():raise RuntimeError('Request cancelled before connect')
            connection.connect();held_socket[0]=connection.sock;held_socket[0].settimeout(deadline)
            with active_lock:active[0]=held_socket[0]
            if (failures and not cleanup_read) or expired.is_set():raise RuntimeError('Request cancelled before send')
            connection.request('GET' if body is None else 'POST',route,body=None if body is None else json.dumps(body).encode(),headers=headers)
            response=connection.getresponse();data=response.read()
            if expired.is_set() or time.monotonic()-started>deadline:raise TimeoutError('Absolute request deadline exceeded')
            if failures and not cleanup_read:raise RuntimeError(failures[0])
            if response.status!=200:raise RuntimeError(f'API HTTP status {response.status}')
            value=json.loads(data)
            if isinstance(value,dict) and value.get('error'):raise RuntimeError('API error response')
            return value
        except BaseException as error:
            event['exception']=repr(error)
            if expired.is_set():raise TimeoutError('Absolute request deadline exceeded') from error
            if failures and not cleanup_read:raise RuntimeError(failures[0]) from error
            raise
        finally:
            timer.cancel();timer.join(timeout=1)
            if response:response.close()
            connection.close()
            with active_lock:active[0]=None
            event.update(elapsed_seconds=time.monotonic()-started,deadline_expired=expired.is_set(),hardware_failures=list(failures))
    assert query('/health')['status']=='ok'
    assert not any(s.get('is_processing') for s in query('/slots'))
    pid=helpers.api_pid();initial=helpers.watch()
    folder=R/'long-cache'/f'{time.strftime("%Y%m%d-%H%M%S")}-{args.label}';folder.mkdir(parents=True,mode=0o700)
    owner=R.stat()
    if os.geteuid()==0:os.chown(folder,owner.st_uid,owner.st_gid)
    record={'label':args.label,'context':args.context,'pid':pid,'config_sha256':hashlib.sha256(raw).hexdigest(),'initial_watch':initial,'samples':[],'requests':[],'request_events':request_events,'error':None,'passed':False,
            'libraries':sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines() if 'libggml' in line or 'libllama' in line})}
    def save():
        output=folder/'RESULT.json';output.write_text(json.dumps(record,indent=2)+'\n');output.chmod(0o600)
        if os.geteuid()==0:os.chown(output,owner.st_uid,owner.st_gid)
    def tokens(text,special=False):return query('/tokenize',{'content':text,'add_special':special})['tokens']
    try:
        prefix=tokens('Synthetic benchmark background notes:',True)
        tail=tokens('\nComplete this increasing list of positive integers, using commas: 1, 2, 3, 4, 5,')
        filler=tokens(' The system stores numbers in memory and processes a sequence of tasks.')
        fill=args.context-len(prefix)-len(tail);prompt=prefix+(filler*((fill+len(filler)-1)//len(filler)))[:fill]+tail
        assert len(prompt)==args.context
        record['initial_prompt_tokens']=prompt
    except BaseException as error:
        record['error']=repr(error);save();raise
    stop=threading.Event();phase=['idle']
    def observe():
        while not stop.wait(2):
            try:
                state=helpers.watch()
                if state['host_counters']!=initial['host_counters']:raise RuntimeError('Hardware counter change')
                record['samples'].append({'time':time.monotonic(),'phase':phase[0],'watch':state})
            except Exception as error:
                failures.append(str(error));interrupt();return
    observer=threading.Thread(target=observe,daemon=True);observer.start()
    try:
        common={'id_slot':0,'n_predict':16,'temperature':0,'seed':42,'ignore_eos':True,'return_tokens':True,'stream':False,'n_cache_reuse':0}
        for phase_name,deadline in [('cold',600 if args.context==16384 else 900),('continuation',60)]:
            assert CONFIG.read_bytes()==raw and helpers.api_pid()==pid
            phase[0]=phase_name;started=time.monotonic()
            response=query('/completion',{**common,'prompt':prompt,'cache_prompt':phase_name=='continuation'},deadline)
            row={'phase':phase_name,'prompt_length':len(prompt),'wall_seconds':time.monotonic()-started,'response':response}
            record['requests'].append(row)
            assert response['id_slot']==0 and not response.get('truncated',False)
            assert response['stop_type']=='limit' and response['timings']['predicted_n']==16 and len(response['tokens'])==16
            assert response['tokens_evaluated']==len(prompt)
            expected_cache=0 if phase_name=='cold' else len(prompt)-1
            expected_new=len(prompt) if phase_name=='cold' else 1
            assert response['timings']['cache_n']==expected_cache and response['timings']['prompt_n']==expected_new
            assert response['tokens_cached']==len(prompt)+15
            prompt=prompt+response['tokens']
            print(json.dumps({'phase':phase_name,'wall_seconds':row['wall_seconds'],'timings':response['timings']}),flush=True)
        record['final_watch']=helpers.watch();assert record['final_watch']['host_counters']==initial['host_counters']
        assert CONFIG.read_bytes()==raw and helpers.api_pid()==pid
        assert not any(s.get('is_processing') for s in query('/slots'))
        stop.set();observer.join(timeout=3)
        if failures:raise RuntimeError(failures[0])
        record['passed']=True
    except BaseException as error:
        record['error']=repr(error);stop.set();observer.join(timeout=3)
        record['cleanup_slot_checks']=[]
        until=time.monotonic()+10
        while time.monotonic()<until:
            try:
                slots=query('/slots',deadline=min(3,max(.1,until-time.monotonic())),cleanup_read=True)
                busy=any(s.get('is_processing') for s in slots);record['cleanup_slot_checks'].append({'busy':busy})
                if not busy:break
            except Exception as cleanup_error:
                record['cleanup_slot_checks'].append({'error':repr(cleanup_error)});break
            time.sleep(.5)
        raise
    finally:
        stop.set();observer.join(timeout=3)
        record['hardware_failures']=failures;save()
    print('Saved '+str(folder),flush=True)
if __name__=='__main__':main()
