#!/usr/bin/env python3
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;LAB=ROOT.parent
sys.path.insert(0,str(LAB/'board-study-20261024'))
from hardware_run import ipmi

def read():return [int(ipmi('raw','0x30','0x90','0x5a','0x03',hex(reg),'0x01').strip(),16) for reg in range(0x10,0x14)]
def set_raw(zone,value):
 assert 0<=zone<4 and 1<=value<=178
 return ipmi('raw','0x30','0x91','0x5a','0x03',hex(0x10+zone),hex(value))
assert ipmi('raw','0x30','0x45','0x00').strip()=='00'
initial=read();assert all(0<v<=178 for v in initial)
report={'initial_raw':initial,'initial_sensors':ipmi('sensor','list'),'test_raw':127,'samples':[]}
try:
 for zone in range(4):set_raw(zone,127)
 start=time.monotonic()
 for delay in [0,2,5,10,20]:
  time.sleep(max(0,start+delay-time.monotonic()))
  row={'elapsed_s':time.monotonic()-start,'raw':read(),'mode':ipmi('raw','0x30','0x45','0x00').strip(),'sensors':ipmi('sensor','list')};report['samples'].append(row)
  assert row['mode']=='00' and max(row['raw'])<=178
  print({'elapsed_s':row['elapsed_s'],'raw':row['raw']},flush=True)
finally:
 for zone,value in enumerate(initial):set_raw(zone,value)
 report['restored_raw']=read();report['restored_mode']=ipmi('raw','0x30','0x45','0x00').strip()
 p=ROOT/'fan-control-probe.json';p.write_text(json.dumps(report,indent=2)+'\n');o=LAB.stat();os.chown(p,o.st_uid,o.st_gid)
