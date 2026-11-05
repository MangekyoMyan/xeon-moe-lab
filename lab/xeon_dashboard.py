#!/usr/bin/env python3
"""Read-only live terminal dashboard. Never starts, stops or generates model requests."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo
from xeon_imc import ImcReader

LAB = Path(__file__).resolve().parent
DEFAULT_ENGINE = LAB/'deepseek41-bringup-20261101'
WATCH = Path('/var/lib/xeon-watch/latest.json')


def read_json(path, limit=512*1024):
    try:
        with Path(path).open('rb') as f:
            raw = f.read(limit+1)
        if len(raw)>limit:
            return None
        value=json.loads(raw)
        return value if isinstance(value,dict) else None
    except (OSError,ValueError):
        return None


def number(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def clean(value):
    return ''.join(c for c in str(value) if c.isprintable() and c not in '\x1b\r\n')


def clip(text, width):
    result='';used=0
    for c in clean(text):
        n=0 if unicodedata.combining(c) else (2 if unicodedata.east_asian_width(c) in 'WF' else 1)
        if used+n>width-1:
            return result+'…'
        result+=c;used+=n
    return result


def fmt(value, unit='', digits=1):
    return f'{value:.{digits}f}{unit}' if number(value) else '--'


def scaled(value, divisor):
    return value/divisor if number(value) else None


def age(stamp, now):
    return max(0,now-stamp) if number(stamp) else None


def seconds(value):
    if not number(value):return '--'
    value=max(0,int(value))
    return f'{value//3600}h{value%3600//60:02d}m' if value>=3600 else f'{value//60}m{value%60:02d}s'


def bar(percent, width=18):
    if not number(percent):return '['+'?'*width+']'
    n=round(width*max(0,min(100,percent))/100)
    return '['+'#'*n+'.'*(width-n)+']'


class Rates:
    def __init__(self):self.previous={}
    def sample(self,key,value,now):
        old=self.previous.get(key)
        self.previous[key]=(value,now)
        if not old or value<old[0] or now<=old[1]:return None
        return (value-old[0])/(now-old[1])


def proc_snapshot(pid):
    if not isinstance(pid,int) or pid<=0:return None
    try:
        root=Path('/proc')/str(pid)
        fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        status={}
        try:
            for line in (root/'status').read_text().splitlines():
                if line.startswith(('VmRSS:','VmLck:','VmSwap:')):status[line.split(':')[0]]=int(line.split()[1])*1024
        except OSError:pass
        try:io={k:int(v) for k,v in (line.split(':') for line in (root/'io').read_text().splitlines())}
        except OSError:io={}
        return dict(pid=pid,start_ticks=int(fields[19]),cpu_ticks=int(fields[11])+int(fields[12]),
                    read_bytes=io.get('read_bytes'),write_bytes=io.get('write_bytes'),**status)
    except (OSError,ValueError,IndexError):return None


def machine_snapshot(rates,now):
    result={}
    try:
        mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:'))}
        result.update(ram_total=mem['MemTotal'],ram_available=mem['MemAvailable'],swap_used=mem['SwapTotal']-mem['SwapFree'])
        cpu=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        total=rates.sample('cpu-total',sum(cpu),now);idle=rates.sample('cpu-idle',cpu[3]+cpu[4],now)
        result['cpu_percent']=100*(1-idle/total) if total and idle is not None else None
        rx=tx=0
        for line in Path('/proc/net/dev').read_text().splitlines()[2:]:
            name,values=line.split(':');name=name.strip()
            if name=='lo' or not (Path('/sys/class/net')/name/'device').exists():continue
            values=values.split();rx+=int(values[0]);tx+=int(values[8])
        result['net_rx']=rates.sample('net-rx',rx,now);result['net_tx']=rates.sample('net-tx',tx,now)
        disks={p.name for p in Path('/sys/block').iterdir() if not p.name.startswith(('loop','ram','dm-','md'))}
        read=write=0
        for line in Path('/proc/diskstats').read_text().splitlines():
            v=line.split()
            if v[2] in disks:read+=int(v[5])*512;write+=int(v[9])*512
        result['disk_read']=rates.sample('disk-r',read,now);result['disk_write']=rates.sample('disk-w',write,now)
        disk=shutil.disk_usage(LAB)
        result.update(disk_free=disk.free,disk_total=disk.total)
    except (OSError,ValueError,KeyError,IndexError):result['unavailable']=True
    return result


class LogTail:
    """Retain numerical performance fields only; never expose log text or prompts."""
    def __init__(self,path):
        self.path=Path(path);self.identity=None;self.pos=0;self.pending=b'';self.values={};self.last_elapsed=-1
    def parse(self,line,started,now):
        prefix=re.match(r'^(\d+)\.(\d+)\.(\d+)\.(\d+)\s',line)
        stamp=None
        if prefix:
            a,b,c,d=map(int,prefix.groups());elapsed=a*60+b+c/1000+d/1e6
            if elapsed<self.last_elapsed-1:self.values={}
            self.last_elapsed=elapsed
            stamp=started+elapsed if number(started) else None
            if stamp and stamp>now+10:return
        task=re.search(r'\btask\s+(\d+)\s*\|',line)
        task=int(task.group(1)) if task else None
        m=re.search(r'prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([\d.]+).*?/\s*([\d.]+) tokens per second',line)
        if m:self.values['progress']=dict(tokens=int(m[1]),percent=float(m[2])*100,tps=float(m[3]),task=task,unix=stamp)
        m=re.search(r'n_gen\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+) t/s,\s*tg_3s\s*=\s*([\d.]+)',line)
        if m:self.values['generation']=dict(tokens=int(m[1]),tps=float(m[3]),average=float(m[2]),task=task,unix=stamp)
        m=re.search(r'(prompt eval time|\beval time)\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens.*?,\s*([\d.]+) tokens per second',line)
        if m:self.values['last_pp' if m[1].startswith('prompt') else 'last_tg']=dict(ms=float(m[2]),tokens=int(m[3]),tps=float(m[4]),task=task,unix=stamp)
    def update(self,identity,started,now):
        try:
            st=self.path.stat();new=(identity,st.st_dev,st.st_ino)
            if new!=self.identity or st.st_size<self.pos:
                self.identity=new;self.values={};self.last_elapsed=-1;self.pending=b'';self.pos=max(0,st.st_size-256*1024)
            if number(started) and st.st_mtime<started:return dict(self.values)
            with self.path.open('rb') as f:
                f.seek(self.pos);data=f.read(256*1024);self.pos=f.tell()
            lines=(self.pending+data).split(b'\n');self.pending=lines.pop()[-8192:]
            for line in lines:self.parse(line.decode('utf-8',errors='replace'),started,now)
            return dict(self.values)
        except OSError:return dict(self.values)


def normalize_api(health,status,slots,now):
    if not isinstance(health,dict):health={}
    result={'unix':now,'http_status':status,'state':'ready' if status==200 and health.get('status')=='ok' else ('loading' if status==503 else 'unavailable')}
    p=health.get('progress')
    if isinstance(p,dict) and number(p.get('value')) and 0<=p['value']<=1:result['loading_percent']=p['value']*100
    result['slots']=[]
    for s in slots if isinstance(slots,list) else []:
        if not isinstance(s,dict):continue
        nt=s.get('next_token')
        next_token=nt[0] if isinstance(nt,list) and nt and isinstance(nt[0],dict) else {}
        result['slots'].append({k:s.get(k) for k in ['id','id_task','is_processing','n_ctx','n_prompt_tokens_cache','n_prompt_tokens_processed']})
        result['slots'][-1]['generated']=next_token.get('n_decoded')
    return result


class ApiReader:
    def __init__(self,config_path,enabled=True):
        self.config_path=config_path;self.enabled=enabled;self.stop=threading.Event();self.lock=threading.Lock();self.value={};self.thread=None
    def poll(self):
        c=read_json(self.config_path)
        if not c:return {'state':'unavailable','unix':time.time(),'error':'設定を読めません'}
        try:
            key=Path(c['api_key_file']).read_text().strip()
            http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def get(path):
                req=urllib.request.Request(f"http://{c['host']}:{c['port']}"+path,headers={'Authorization':'Bearer '+key})
                try:
                    with http.open(req,timeout=1.5) as f:return f.status,json.loads(f.read(512*1024))
                except urllib.error.HTTPError as e:
                    try:body=json.loads(e.read(64*1024))
                    except ValueError:body={}
                    return e.code,body
            status,health=get('/health')
            slots=[];error=None
            if status==200:
                try:
                    code,slots=get('/slots')
                    if code!=200:slots=[];error=f'スロット情報 HTTP {code}'
                except (OSError,ValueError):error='スロット情報待ち / 処理中のため次回確認'
            result=normalize_api(health,status,slots,time.time())
            if error:result['error']=error
            return result
        except (OSError,ValueError,KeyError):return {'state':'unavailable','unix':time.time(),'error':'接続待ち / 読み取り不可'}
    def start(self):
        if not self.enabled:return
        def loop():
            while not self.stop.is_set():
                value=self.poll()
                with self.lock:self.value=value
                self.stop.wait(15 if value.get('error') else 5)
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def snapshot(self):
        with self.lock:return dict(self.value)


def download_snapshot(manifest_path,directory,rates,now):
    manifest=read_json(manifest_path)
    if not manifest or not isinstance(manifest.get('files'),list):return None
    base=Path(directory).resolve();total=done=finals=0;current=None;invalid=False;signatures=[]
    for item in manifest['files']:
        if not isinstance(item,dict) or not isinstance(item.get('name'),str) or not isinstance(item.get('size'),int) or item['size']<=0:return None
        path=(base/item['name']).resolve()
        if not path.is_relative_to(base):return None
        if path.with_name(path.name+'.aria2').exists():invalid=True
        expected=item['size'];total+=expected;signatures.append((item['name'],expected,item.get('sha256')))
        try:
            size=path.stat().st_size
            if size==expected:done+=expected;finals+=1;continue
            invalid=True;current=path.name
        except FileNotFoundError:pass
        except OSError:invalid=True
        partial=path.with_name(path.name+'.part')
        try:
            st=partial.stat();size=st.st_size
            if size>expected or st.st_blocks*512+4096<size:invalid=True
            else:done+=size
            current=partial.name
        except FileNotFoundError:pass
        except OSError:invalid=True
    if not total:return None
    rate=rates.sample(('download',str(base),tuple(signatures)),done,now)
    verification=read_json(base/'download-verified.json') or {}
    verified_files=[(f.get('name'),f.get('size'),f.get('sha256')) for f in verification.get('files',[]) if isinstance(f,dict)]
    verified=finals==len(signatures) and verified_files==signatures and verification.get('revision')==manifest.get('revision')
    progress=read_json(base/'download-progress.json') or {}
    live=False
    if progress.get('repository')==manifest.get('repository') and progress.get('revision')==manifest.get('revision'):
        proc=proc_snapshot(progress.get('pid'))
        live=bool(proc and str(proc['start_ticks'])==str(progress.get('start_ticks')) and progress.get('boot_id')==read_boot_id())
    else:progress={}
    state='検証記録あり' if verified else ('転送100%・検証未確認' if done==total else '未開始/変化待ち')
    if rate and rate>0:state='転送中'
    if live:state={'starting':'開始準備','downloading':'転送中','verifying':'SHA256検証中','file_verified':'ファイル検証済み','complete':'完了','error':'エラー','interrupted':'中断'}.get(progress.get('phase'),'実行中')
    elif progress.get('phase') in ['error','interrupted']:state='エラー/中断（途中データ保持）'
    elif progress.get('phase') in ['downloading','verifying','starting'] and not verified:state='前回の途中データ / 現在の実行は未確認'
    if invalid:state='容量不一致/予約ファイル等・割合未確定'
    vp=None
    if live and progress.get('phase')=='verifying' and number(progress.get('total_bytes')) and progress['total_bytes']>0:
        vp=100*progress.get('done_bytes',0)/progress['total_bytes']
    return {'repository':manifest.get('repository',base.name),'directory':str(base),'total':total,'done':done,
            'percent':None if invalid else 100*done/total,'bytes_per_second':rate,'eta_seconds':(total-done)/rate if rate and rate>0 and done<total else None,
            'files_done':finals,'files_total':len(signatures),'state':state,'file':progress.get('file') if live else current,
            'verification_percent':vp,'verified_record':verified,'live':live}


def read_boot_id():
    try:return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:return None


def recent_gpu_window(health,now):
    samples=[]
    for sample in health.get('recent_samples',[]):
        stamp=sample.get('unix');util=sample.get('gpu',{}).get('gpu_util')
        if number(stamp) and 0<=now-stamp<=30 and number(util) and 0<=util<=100:
            samples.append((stamp,util))
    if not samples:return None
    return {'mean_percent':sum(v for _,v in samples)/len(samples),'max_percent':max(v for _,v in samples),
            'sample_count':len(samples),'last_age_seconds':now-max(t for t,_ in samples),'window_seconds':30}


class Dashboard:
    def __init__(self,engine,manifest,destination,api=True,imc=True):
        self.engine=Path(engine);self.manifest=manifest;self.destination=destination
        self.rates=Rates();self.log=LogTail(self.engine/'server.stderr.log');self.api=ApiReader(self.engine/'server-config.json',api)
        self.last_slot=None;self.last_rate=None
        self.imc=ImcReader(imc)
    def snapshot(self):
        now=time.time();mono=time.monotonic();config=read_json(self.engine/'server-config.json') or {}
        health=read_json(self.engine/'health.json') or {};raw_watch=read_json(WATCH) or {}
        watch={k:raw_watch[k] for k in ['unix','cpu','gpu','bmc'] if k in raw_watch}
        watch['errors']={k:True for k in raw_watch.get('errors',{})}
        watch['active_alerts']={k:True for k in raw_watch.get('active_alerts',{})}
        if health.get('boot_id')!=read_boot_id():health={}
        proc=proc_snapshot(health.get('pid'))
        if proc:
            identity=(proc['pid'],proc['start_ticks'])
            cpu=self.rates.sample(('process-cpu',identity),proc['cpu_ticks'],mono)
            proc['cpu_cores']=cpu/os.sysconf('SC_CLK_TCK') if cpu is not None else None
            proc['read_per_second']=self.rates.sample(('process-read',identity),proc['read_bytes'],mono) if number(proc['read_bytes']) else None
        perf=self.log.update((health.get('pid'),health.get('started_unix')),health.get('started_unix'),now)
        api=self.api.snapshot();active=[s for s in api.get('slots',[]) if s.get('is_processing')]
        slot=active[0] if active else None
        if slot and number(slot.get('generated')):
            ident=(health.get('pid'),slot.get('id'),slot.get('id_task'))
            old=self.last_slot
            if not old or old[0]!=ident:self.last_rate=None
            if old and old[0]==ident and api.get('unix',0)>old[2] and slot['generated']>=old[1]:
                self.last_rate=(slot['generated']-old[1])/(api['unix']-old[2])
            if not old or api.get('unix',0)>old[2]:self.last_slot=(ident,slot['generated'],api.get('unix',now))
            api['observed_tps']=self.last_rate
        else:self.last_slot=None;self.last_rate=None
        return {'unix':now,'model':config.get('model_alias'),'context':config.get('context'),
                'profile':'expert固定: GPU 2層 / CPU 38層' if config.get('n_cpu_moe')==38 else ('CPU expert' if config.get('cpu_moe') else 'カスタム'),
                'expert_cache_mib':config.get('moe_cache_mib',0),
                'memory_bandwidth':self.imc.sample(),'gpu_window':recent_gpu_window(health,now),
                'watch':watch,'health':{k:health.get(k) for k in ['status','pid','started_unix','updated_unix']},
                'process':proc,'machine':machine_snapshot(self.rates,mono),'performance':perf,'api':api,
                'download':download_snapshot(self.manifest,self.destination,self.rates,mono)}


def render(s):
    now=s['unix'];w=s['watch'];gpu=w.get('gpu',{});bmc=w.get('bmc',{});m=s['machine'];api=s['api'];p=s['process'] or {};perf=s['performance'];d=s['download']
    fresh=age(w.get('unix'),now);ba=age(bmc.get('observed_unix'),now)
    stale=fresh is None or fresh>45
    cpu=[v for v in w.get('cpu',{}).values() if number(v)];dimm=[v for v in bmc.get('dimms_C',{}).values() if number(v)]
    lines=['XEON LIVE  '+datetime.datetime.fromtimestamp(now,ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST')+'   Ctrl+Cで表示だけ終了',
           '表示専用 | センサーは既存監視（CPU/GPU約10秒、DIMM/ファン約30秒）を再利用',
           '',f"温度  CPU最高 {fmt(max(cpu) if cpu else None,'C')}  GPU {fmt(gpu.get('gpu_C'),'C')}  HBM {fmt(gpu.get('hbm_C'),'C')}  DIMM最高 {fmt(max(dimm) if dimm else None,'C')}",
           f"      PCH {fmt(bmc.get('pch_C'),'C')}  周辺 {fmt(bmc.get('peripheral_C'),'C')}  センサー {seconds(fresh)}前 / BMC {seconds(ba)}前"+('  [古い/取得不可]' if stale else ''),
           'ファン '+(' / '.join(fmt(v,'%') for v in bmc.get('fan_percent',[])) or '--')+'  '+('  '.join(clean(k)+':'+fmt(v,'rpm',0) for k,v in bmc.get('fan_rpm',{}).items()) or ''),
           f"GPU   使用率 {fmt(gpu.get('util_percent'),'%')}  電力 {fmt(gpu.get('power_W'),'W')} / {fmt(gpu.get('power_limit_W'),'W',0)}  VRAM使用 {fmt(gpu.get('used_MiB',0)/1024 if number(gpu.get('used_MiB')) else None,'GiB',2)}  ECC {clean(gpu.get('ecc_mode','--'))}",
           f"CPU   使用率 {fmt(m.get('cpu_percent'),'%')}  モデルCPU使用 {fmt(p.get('cpu_cores'),'コア相当',2)}",
           f"RAM   利用可能 {fmt(scaled(m.get('ram_available'),2**30),'GiB')} / {fmt(scaled(m.get('ram_total'),2**30),'GiB')}  モデルRSS {fmt(scaled(p.get('VmRSS'),2**30),'GiB')}  固定 {fmt(scaled(p.get('VmLck'),2**30),'GiB')}",
           f"HDD   空き {fmt(scaled(m.get('disk_free'),2**30),'GiB')}  読出 {fmt(m.get('disk_read')/2**20 if number(m.get('disk_read')) else None,'MiB/s')}  書込 {fmt(m.get('disk_write')/2**20 if number(m.get('disk_write')) else None,'MiB/s')}",
           f"通信  受信 {fmt(m.get('net_rx')/2**20 if number(m.get('net_rx')) else None,'MiB/s')}  送信 {fmt(m.get('net_tx')/2**20 if number(m.get('net_tx')) else None,'MiB/s')}", '']
    bandwidth=s.get('memory_bandwidth',{})
    if bandwidth.get('status')=='ok':
        read=bandwidth['read_bytes_per_second'];write=bandwidth['write_bytes_per_second']
        detail=' / '.join('S'+sid+' R'+fmt(values['read_bytes_per_second']/1e9,'',1)+' W'+fmt(values['write_bytes_per_second']/1e9,'',1)
                        for sid,values in sorted(bandwidth['sockets'].items()))
        extra=f" [多重化・稼働率 {100*bandwidth['running_fraction_min']:.0f}%]" if bandwidth['multiplexed'] else ''
        lines[9:9]=[f"DRAM  読込 {read/1e9:.2f} / 書込 {write/1e9:.2f} / 合計 {(read+write)/1e9:.2f} GB/s（全体実測）",
                    '      '+detail+' GB/s  窓'+fmt(bandwidth['interval_seconds'],'秒')+extra]
    else:
        reason={'warming':'初回の差分待ち','disabled':'--no-imcで無効'}.get(bandwidth.get('status'),'取得不可（sudo権限/PMU競合等）')
        lines[9:9]=['DRAM  -- GB/s  '+reason]
    window=s.get('gpu_window')
    if window:
        lines.insert(7,f"GPU窓 30秒採取値: 平均 {window['mean_percent']:.1f}% / 最大 {window['max_percent']:.1f}% ({window['sample_count']}点、最新{window['last_age_seconds']:.0f}秒前)")
    states={'ready':'起動済み','loading':'読込/準備中','unavailable':'停止/接続待ち'}
    status=states.get(api.get('state'),'確認待ち' if api else 'API情報なし')
    aa=age(api.get('unix'),now)
    if aa is not None and aa>20:status+='（情報が古い）'
    expert_cache=s.get('expert_cache_mib',0)
    cache_text=f'expert cache {expert_cache}MiB' if expert_cache else 'expertの動的入替なし'
    lines += [f"モデル {clean(s.get('model') or '--')}  context {s.get('context','--')}",
              f"配置   {clean(s['profile'])} / {cache_text}（設定）",
              f"API    {status}  PID {s['health'].get('pid') or '--'}  起動経過 {seconds(age(s['health'].get('started_unix'),now))}  API情報 {seconds(aa)}前"]
    if number(api.get('loading_percent')):lines.append('読込   '+bar(api['loading_percent'])+' '+fmt(api['loading_percent'],'%'))
    active=[x for x in api.get('slots',[]) if x.get('is_processing')]
    if active:
        slot=active[0];task=slot.get('id_task');pp=perf.get('progress',{});tg=perf.get('generation',{})
        lines.append(f"処理中 task {task}  生成済 {slot.get('generated','--')} token  再利用 {slot.get('n_prompt_tokens_cache','--')} token")
        if pp.get('task')==task and not slot.get('generated'):
            lines.append('prefill '+bar(pp.get('percent'))+' '+fmt(pp.get('percent'),'%')+'  '+fmt(pp.get('tps'),' tok/s')+'（入力の処理状況）')
        else:
            speed=tg.get('tps') if tg.get('task')==task else api.get('observed_tps')
            lines.append('生成   '+fmt(speed,' tok/s',2)+'（最近の計測）')
    else:lines.append('処理   '+('待機中' if api.get('state')=='ready' and 'error' not in api else '状態確認待ち'))
    pp=perf.get('last_pp',{});tg=perf.get('last_tg',{})
    lines.append('前回完了  入力 '+fmt(pp.get('tps'),' tok/s',2)+' / 生成 '+fmt(tg.get('tps'),' tok/s',2)+'  '+seconds(age(tg.get('unix'),now))+'前')
    if api.get('error'):lines.append('API情報 '+clean(api['error']))
    lines.append('')
    if d:
        lines += ['Download '+clean(d['repository']),
                  bar(d['percent'])+' '+fmt(d['percent'],'%')+f"  {d['done']/2**30:.2f}/{d['total']/2**30:.2f}GiB  完了ファイル {d['files_done']}/{d['files_total']}",
                  f"{d['state']}  転送 {fmt(d['bytes_per_second']/2**20 if number(d['bytes_per_second']) else None,'MiB/s')}  残り目安 {seconds(d['eta_seconds'])}"]
        if d.get('file'):lines.append('対象 '+clean(d['file']))
        if number(d.get('verification_percent')):lines.append('SHA256 '+bar(d['verification_percent'])+' '+fmt(d['verification_percent'],'%'))
    else:lines.append('Download 進捗情報なし（対応manifestを指定できます）')
    alerts=list(w.get('active_alerts',{}));errors=list(w.get('errors',{}))
    lines += ['', 'アラーム: '+(', '.join(map(clean,alerts)) or 'なし')+'   センサー取得失敗: '+(', '.join(map(clean,errors)) or 'なし')]
    if 'unix' not in w:lines.append('監視記録を読めません。sudoで起動するか常駐監視の状態を確認してください。')
    return lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true',help='display one snapshot')
    parser.add_argument('--json',action='store_true',help='one sanitized JSON snapshot')
    parser.add_argument('--interval',type=float,default=2)
    parser.add_argument('--duration',type=float,help='stop this display after N seconds')
    parser.add_argument('--no-api',action='store_true',help='skip HTTP API; use local files/counters')
    parser.add_argument('--no-imc',action='store_true',help='do not open memory-controller counters')
    parser.add_argument('--engine-dir',type=Path,default=DEFAULT_ENGINE)
    parser.add_argument('--download-manifest',type=Path,default=LAB/'deepseek41-orca-mix256-download-manifest.json')
    parser.add_argument('--download-dir',type=Path,default=LAB/'models/orcarouter--DeepSeek-V4.1-Flash-1017-abliterated-GGUF')
    args=parser.parse_args()
    if not 1<=args.interval<=60:parser.error('--interval must be1..60')
    if args.duration is not None and args.duration<=0:parser.error('--duration must be positive')
    one=args.once or args.json
    d=Dashboard(args.engine_dir,args.download_manifest,args.download_dir,not args.no_api,not args.no_imc)
    stop=threading.Event()
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.set())
    tty=sys.stdout.isatty() and not one
    try:
        if one and not args.no_api:d.api.value=d.api.poll()
        else:d.api.start()
        if one and not args.no_imc:
            d.imc.sample()
            stop.wait(1)
        if tty:sys.stdout.write('\x1b[?1049h\x1b[?25l');sys.stdout.flush()
        start=time.monotonic()
        while not stop.is_set():
            snapshot=d.snapshot()
            if args.json:print(json.dumps(snapshot,ensure_ascii=False,indent=2));break
            lines=render(snapshot)
            if tty:
                size=shutil.get_terminal_size((110,36));lines=[clip(x,max(8,size.columns)) for x in lines]
                if len(lines)>size.lines-1:
                    alert=next((line for line in lines if line.startswith('アラーム:')),'')
                    lines=lines[:max(1,size.lines-3)]+[alert,'表示を広げると残りの項目が見えます。Ctrl+Cで終了']
                sys.stdout.write('\x1b[H\x1b[J'+'\n'.join(lines)+'\n');sys.stdout.flush()
            else:print('\n'.join(lines),flush=True)
            if one:break
            left=None if args.duration is None else args.duration-(time.monotonic()-start)
            if left is not None and left<=0:break
            stop.wait(args.interval if left is None else min(args.interval,left))
    finally:
        d.api.stop.set()
        d.imc.close()
        if tty:sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()


if __name__=='__main__':
    main()
