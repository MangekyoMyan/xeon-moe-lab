"""System-wide DRAM traffic through Linux perf IMC counters; no MSR/sysctl writes."""
import ctypes
import math
import os
from pathlib import Path
import platform
import struct
import time

PMU_ROOT = Path('/sys/bus/event_source/devices')
CPU_ROOT = Path('/sys/devices/system/cpu')


def cpu_list(text):
    cpus=set()
    for part in text.strip().split(','):
        if not part:continue
        values=part.split('-')
        start=int(values[0]);end=int(values[-1])
        if start<0 or end<start or end-start>4096:raise ValueError('invalid PMU CPU list')
        cpus.update(range(start,end+1))
    return sorted(cpus)


def event_definition(pmu, name):
    if (pmu/'format/event').read_text().strip()!='config:0-7' or (pmu/'format/umask').read_text().strip()!='config:8-15':
        raise ValueError('unsupported IMC event layout')
    terms=dict(term.split('=') for term in (pmu/'events'/name).read_text().strip().split(','))
    if set(terms)!={'event','umask'}:raise ValueError('unsupported IMC event terms')
    event=int(terms['event'],0);umask=int(terms['umask'],0)
    if not 0<=event<=255 or not 0<=umask<=255:raise ValueError('invalid event value')
    unit=(pmu/'events'/(name+'.unit')).read_text().strip()
    scale=float((pmu/'events'/(name+'.scale')).read_text())
    if unit!='MiB' or not math.isfinite(scale) or scale<=0:raise ValueError('unsupported IMC counter unit')
    return event | (umask<<8), scale*1024**2


def discover():
    definitions=[]
    for pmu in sorted(PMU_ROOT.glob('uncore_imc_[0-9]*')):
        kind=int((pmu/'type').read_text())
        seen=set()
        for cpu in cpu_list((pmu/'cpumask').read_text()):
            package=int((CPU_ROOT/f'cpu{cpu}/topology/physical_package_id').read_text())
            if package in seen:raise ValueError('ambiguous IMC representatives for one socket')
            seen.add(package)
            for direction in ['read','write']:
                config,scale=event_definition(pmu,'cas_count_'+direction)
                definitions.append(dict(pmu=pmu.name,cpu=cpu,socket=package,direction=direction,
                                        type=kind,config=config,bytes_per_count=scale))
    if not definitions:raise ValueError('no supported uncore IMC counters')
    return definitions


def open_counter(event):
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise ValueError('this collector supports x86_64 Linux')
    attr=bytearray(128)
    struct.pack_into('=IIQ',attr,0,event['type'],len(attr),event['config'])
    # Counting only, automatically enabled. Include enabled/running times for multiplex correction.
    struct.pack_into('=Q',attr,32,3)
    raw=(ctypes.c_char*len(attr)).from_buffer(attr)
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    fd=libc.syscall(ctypes.c_long(298),ctypes.byref(raw),ctypes.c_int(-1),ctypes.c_int(event['cpu']),
                    ctypes.c_int(-1),ctypes.c_ulong(8))  # PERF_FLAG_FD_CLOEXEC
    if fd<0:
        err=ctypes.get_errno();raise OSError(err,os.strerror(err))
    return fd


def read_counter(fd):
    raw=os.read(fd,24)
    if len(raw)!=24:raise OSError('short IMC counter read')
    return (*struct.unpack('=QQQ',raw),time.monotonic())


def interval_rate(previous,current,bytes_per_count):
    count,enabled,running,stamp=current
    dc=count-previous[0];de=enabled-previous[1];dr=running-previous[2];dt=stamp-previous[3]
    if dc<0 or de<=0 or dr<=0 or dr>de or dt<=0:
        raise ValueError('IMC counter reset or not scheduled')
    return {'bytes_per_second':dc*(de/dr)*bytes_per_count/dt,
            'running_fraction':dr/de,'interval_seconds':dt}


def aggregate(rows):
    sockets={};coverage=1.0
    for event,rate in rows:
        socket=sockets.setdefault(str(event['socket']),{'read_bytes_per_second':0.0,'write_bytes_per_second':0.0,'channels':{}})
        direction=event['direction'];value=rate['bytes_per_second']
        socket[direction+'_bytes_per_second']+=value
        socket['channels'].setdefault(event['pmu'],{})[direction+'_bytes_per_second']=value
        coverage=min(coverage,rate['running_fraction'])
    return {'sockets':sockets,'read_bytes_per_second':sum(s['read_bytes_per_second'] for s in sockets.values()),
            'write_bytes_per_second':sum(s['write_bytes_per_second'] for s in sockets.values()),
            'running_fraction_min':coverage,'multiplexed':coverage<.99,
            'interval_seconds':sum(rate['interval_seconds'] for _,rate in rows)/len(rows)}


class ImcReader:
    def __init__(self, enabled=True):
        self.fds=[];self.previous=[];self.events=[];self.error=None;self.enabled=enabled
        if not enabled:return
        try:
            self.events=discover()
            for event in self.events:self.fds.append(open_counter(event))
            self.previous=[read_counter(fd) for fd in self.fds]
        except (OSError,ValueError) as error:
            self.error=f'{type(error).__name__}: {error}'
            self.close()

    def sample(self):
        base={'unix':time.time(),'scope':'system-wide DRAM controller traffic','unit':'bytes/s',
              'event_count':len(self.events)}
        if not self.enabled:return dict(base,status='disabled')
        if self.error or not self.fds:return dict(base,status='unavailable',error=self.error)
        try:
            current=[read_counter(fd) for fd in self.fds]
            previous=self.previous;self.previous=current
            if any(b[3]-a[3]<.2 for a,b in zip(previous,current)):
                return dict(base,status='warming')
            rows=[(event,interval_rate(a,b,event['bytes_per_count'])) for event,a,b in zip(self.events,previous,current)]
            return dict(base,status='ok',**aggregate(rows))
        except ValueError as error:
            return dict(base,status='unavailable',error=str(error))
        except OSError as error:
            self.error=f'{type(error).__name__}: {error}'
            self.close()
            return dict(base,status='unavailable',error=self.error)

    def close(self):
        for fd in self.fds:
            try:os.close(fd)
            except OSError:pass
        self.fds=[]
