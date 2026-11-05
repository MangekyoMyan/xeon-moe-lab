#!/usr/bin/env python3
import csv
import argparse
import json
import re
import statistics
from pathlib import Path

OUT=Path(__file__).resolve().parent.parent/'results/server-study-20261023'

def core_metrics(path):
    entries={}
    for row in csv.reader(path.read_text().splitlines()):
        if not row or row[0].startswith('#'):continue
        if len(row)<5:continue
        entries[row[2]]={'count':float(row[0]),'unit':row[1],'runtime_ns':float(row[3]),'running_pct':float(row[4])}
    assert entries and all(v['running_pct']>=99.9 for v in entries.values()),entries
    v=lambda k:entries[k]['count']
    return {'IPC':v('instructions:u')/v('cycles:u'),
            'average_active_MHz':v('cycles:u')/(v('task-clock')*1000),
            'L2_pending_stall_cycle_fraction':v('cycle_activity.stalls_l2_pending:u')/v('cycles:u'),
            'cycles':v('cycles:u'),'instructions':v('instructions:u'),
            'minor_faults':v('minor-faults'),'major_faults':v('major-faults'),
            'CPU_migrations':v('cpu-migrations'),'context_switches':v('context-switches'),
            'all_events_running_99_9pct':True,'entries':entries}

def imc_metrics(path):
    entries=[]
    pattern=re.compile(r'^CPU(\d+),([0-9.]+),,uncore_imc_(\d+)/event=0x04,umask=0x(03|0c)/,([0-9.]+),([0-9.]+)')
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):continue
        m=pattern.match(line);assert m,line
        cpu,count,channel,mask,runtime,pct=m.groups()
        assert float(pct)>=99.9
        entries.append({'counter_CPU':int(cpu),'node':0 if int(cpu)==0 else 1 if int(cpu)==6 else -1,
                        'channel':int(channel),'direction':'read' if mask=='03' else 'write',
                        'transactions':float(count),'bytes':float(count)*64,
                        'enabled_s':float(runtime)/1e9,'GB_s':float(count)*64/float(runtime),'running_pct':float(pct)})
    assert len(entries)==16
    totals={direction:sum(x['GB_s'] for x in entries if x['direction']==direction) for direction in ['read','write']}
    return {'read_GBs':totals['read'],'write_GBs':totals['write'],'entries':entries}

def io_values(text):
    return {k:int(v.strip()) for line in (text or '').splitlines() for k,v in [line.split(':',1)]}

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args();OUT=args.output.resolve()
    assert json.loads((OUT/'complete.json').read_text())['complete']
    rows=[];runs=[];excluded=[]
    for folder in sorted((OUT/'measurements').iterdir()):
        if not (re.search(r'_r[012]$',folder.name) or folder.name=='read_both_SMT_sustained'):
            excluded.append(folder.name);continue
        r=json.loads((folder/'run.json').read_text());assert r['valid']
        core=core_metrics(folder/'core.csv');imc=imc_metrics(folder/'imc.csv')
        before,after=io_values(r.get('process_io_before')),io_values(r.get('process_io_after'))
        disk_read=after.get('read_bytes',0)-before.get('read_bytes',0)
        assert disk_read==0
        assert all(x==0 for x in r['thermal_throttle_delta'].values())
        assert all(x==0 for d in r['ECC_delta'].values() for x in d.values())
        row={'label':folder.name,'case':re.sub(r'_r[012]$','',folder.name),'mode':r['mode'],'threads':r['threads'],
             'logical_GBs':r['result']['logical_GBs'],'dependent_load_ns':r['result']['dependent_load_ns'],
             'IMC_read_GBs':imc['read_GBs'],'IMC_write_GBs':imc['write_GBs'],
             'active_MHz':core['average_active_MHz'],'IPC':core['IPC'],
             'L2_pending_stall_fraction':core['L2_pending_stall_cycle_fraction'],
             'RAPL_package_W':sum(v['watts'] for v in r['power']['domains'].values() if v['name'].startswith('package-')),
             'RAPL_DRAM_W':sum(v['watts'] for v in r['power']['domains'].values() if v['name']=='dram'),
             'RAPL_package_plus_DRAM_W':r['power']['package_plus_dram_W'],
             'max_CPU_C':r['temperature_max_C'],'disk_read_bytes':disk_read,
             'minor_faults_observation_window':core['minor_faults'],'major_faults':core['major_faults']}
        assert row['major_faults']==0
        rows.append(row);runs.append({'label':folder.name,'core':core,'imc':imc})
    assert len(rows)==55
    grouped={}
    keys=['logical_GBs','dependent_load_ns','IMC_read_GBs','IMC_write_GBs','active_MHz','IPC','L2_pending_stall_fraction','RAPL_package_W','RAPL_DRAM_W','RAPL_package_plus_DRAM_W','max_CPU_C']
    for case in sorted({r['case'] for r in rows}):
        subset=[r for r in rows if r['case']==case]
        assert len(subset)==(1 if case.endswith('sustained') else 3)
        grouped[case]={'runs':len(subset),'mode':subset[0]['mode'],'threads':subset[0]['threads'],
                       **{k:{'median':statistics.median(r[k] for r in subset),'min':min(r[k] for r in subset),'max':max(r[k] for r in subset)} for k in keys}}
    def metric(case,key='logical_GBs'):return grouped[case][key]['median']
    summary={'user_date_JST':'2026-10-23','scope':'Current128GB host real measurements. No GPU or GPU simulation.',
             'cases':grouped,'measurement_count':len(rows),'excluded_from_main':excluded,
             'read_remote_vs_local_ratio_node0':metric('read_n0_remote')/metric('read_n0_local'),
             'read_remote_vs_local_ratio_node1':metric('read_n1_remote')/metric('read_n1_local'),
             'latency_remote_vs_local_ratio_node0':metric('latency_n0_remote','dependent_load_ns')/metric('latency_n0_local','dependent_load_ns'),
             'latency_remote_vs_local_ratio_node1':metric('latency_n1_remote','dependent_load_ns')/metric('latency_n1_local','dependent_load_ns'),
             'copy_remote_destination_vs_local_ratio':metric('copy_n0_to_remote1')/metric('copy_n0_local1'),
             'read_SMT_vs_physical_ratio':metric('read_both_SMT')/metric('read_both_physical'),
             'max_observed_CPU_C':max(r['max_CPU_C'] for r in rows),
             'ECC_new_errors':0,'thermal_throttle_new_events':0,'major_faults':0,'model_disk_reads_during_RAM_loops':0,
             'all_core_and_IMC_counters_unmultiplexed':True,
             'caveats':['RAPL numbers exclude PSU losses,motherboard,HDD,fans; not whole-system power.','4KiB pointer-chase latency includes cache/TLB overhead.','Copy throughput is payload GB/s; total memory traffic is higher.','Counter/RAPL windows include short handshake bookends; C++throughput is timed-loop-only.','Some minor faults occur in the observer window; no claim of zero minor faults.','Different workload shapes may have different bottlenecks; these are not LLM tokens/s.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'counter-details.json').write_text(json.dumps(runs,indent=2)+'\n')
    with (OUT/'measurements.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    for case,v in grouped.items():
        print(case,'GB/s',round(v['logical_GBs']['median'],3),'ns',round(v['dependent_load_ns']['median'],2),
              'MHz',round(v['active_MHz']['median']),'RAPL',round(v['RAPL_package_plus_DRAM_W']['median'],1))
if __name__=='__main__':main()
