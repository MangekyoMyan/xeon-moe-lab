#!/usr/bin/env python3
"""Read-only host inventory. Excludes serial numbers and networking identities."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

LAB=Path(__file__).resolve().parent.parent
def read(path):
    try:return Path(path).read_text().strip()
    except (OSError,UnicodeError):return None
def command(args,timeout=20):
    if not shutil.which(args[0]):return {'available':False,'command':args}
    try:
        r=subprocess.run(args,text=True,capture_output=True,timeout=timeout)
        return {'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except subprocess.TimeoutExpired:return {'command':args,'timeout':timeout}
def dmidecode(kind):
    r=command(['dmidecode','-t',str(kind)])
    lines=[line for line in r.get('stdout','').splitlines() if not any(s in line.lower() for s in ['serial number:','asset tag:','uuid:'])]
    r['stdout']='\n'.join(lines)+'\n';return r
def telemetry():
    t={'temperatures':{},'throttle_counts':{},'powercap':{},'frequency':{}}
    for hw in Path('/sys/class/hwmon').glob('hwmon*'):
        name=read(hw/'name')
        for p in hw.glob('temp*_input'):
            label=read(p.with_name(p.name.replace('_input','_label')))
            t['temperatures'][f'{hw.name}:{name}:{label or p.stem}']={'input_mC':read(p),'max_mC':read(p.with_name(p.name.replace('_input','_max'))),'crit_mC':read(p.with_name(p.name.replace('_input','_crit')))}
    for p in Path('/sys/devices/system/cpu').glob('cpu*/thermal_throttle/*throttle_count'):t['throttle_counts'][str(p)]=read(p)
    for p in Path('/sys/class/powercap').glob('intel-rapl:*'):
        t['powercap'][p.name]={x.name:read(x) for x in p.iterdir() if x.name in ['name','energy_uj','max_energy_range_uj','enabled'] or x.name.startswith('constraint_')}
    for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*'):
        t['frequency'][p.name]={n:read(p/n) for n in ['affected_cpus','related_cpus','scaling_driver','scaling_governor','scaling_min_freq','scaling_max_freq','scaling_cur_freq','cpuinfo_min_freq','cpuinfo_max_freq','cpuinfo_transition_latency','energy_performance_preference']}
    return t
def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    data={'captured_UTC':datetime.now(timezone.utc).isoformat(),'user_date_JST':'2026-10-23','boot_id':read('/proc/sys/kernel/random/boot_id'),'uid':os.getuid(),'no_settings_changed':True}
    cmds={'uname':['uname','-r'],'cpu':['lscpu','-J'],'cpu_map':['lscpu','-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ'],
          'numa':['numactl','--hardware'],'pci':['lspci','-Dnn'],'pci_tree':['lspci','-Dtv'],'memory':['free','-b'],
          'disks':['lsblk','-J','-b','-o','NAME,TYPE,SIZE,ROTA,MODEL,TRAN,MOUNTPOINTS'],
          'perf':['perf','--version'],'turbostat':['turbostat','--version'],'cpu_power':['cpupower','frequency-info'],'modules':['lsmod'],
          'packages':['dpkg-query','-W','-f='+chr(36)+'{Package} '+chr(36)+'{Version}\n','nvidia-*','libnvidia*','cuda-*','libcuda*','linux-image-*','linux-headers-*','gcc','g++','cmake','numactl','libnuma*','rasdaemon','ipmitool'],
          'processes':['ps','-eo','pid,comm,pcpu,pmem,stat','--sort=-pcpu'],
          'kernel_hardware_log':['journalctl','-k','-b','--no-pager','-g','EDAC|MCE|Hardware Error|AER|NVRM|nouveau|DMAR|IOMMU']}
    data['commands']={}
    for name,cmd in cmds.items():
        r=command(cmd)
        if name=='processes' and 'stdout' in r:r['stdout']='\n'.join(r['stdout'].splitlines()[:20])
        data['commands'][name]=r
    data['dmi']={str(k):dmidecode(k) for k in [0,2,9,16,17]}
    data['sys']={n:read(n) for n in ['/proc/cmdline','/proc/meminfo','/proc/sys/kernel/numa_balancing','/sys/kernel/mm/transparent_hugepage/enabled','/sys/kernel/mm/transparent_hugepage/defrag','/proc/sys/kernel/perf_event_paranoid','/proc/sys/kernel/nmi_watchdog','/sys/devices/system/cpu/intel_pstate/no_turbo','/sys/devices/system/cpu/intel_pstate/min_perf_pct','/sys/devices/system/cpu/intel_pstate/max_perf_pct','/sys/devices/system/cpu/intel_pstate/status','/proc/sys/vm/swappiness','/proc/sys/vm/overcommit_memory']}
    data['boot_efi']=Path('/sys/firmware/efi').exists();data['telemetry']=telemetry()
    data['nodes']={p.name:{k:read(p/k) for k in ['cpulist','distance','meminfo','numastat']} for p in Path('/sys/devices/system/node').glob('node[0-9]*')}
    data['edac']={}
    for mc in Path('/sys/devices/system/edac/mc').glob('mc[0-9]*'):
        m={k:read(mc/k) for k in ['mc_name','size_mb','ce_count','ue_count','ce_noinfo_count','ue_noinfo_count','seconds_since_reset','sdram_scrub_rate']}
        m['dimms']={p.name:{k:read(p/k) for k in ['dimm_label','dimm_location','dimm_mem_type','dimm_dev_type','dimm_edac_mode','size','dimm_ce_count','dimm_ue_count']} for p in mc.glob('dimm*')}
        data['edac'][mc.name]=m
    data['pci_devices']={}
    for dev in sorted(Path('/sys/bus/pci/devices').iterdir()):
        item={k:read(dev/k) for k in ['vendor','device','class','numa_node','local_cpulist','physical_slot','max_link_speed','current_link_speed','max_link_width','current_link_width','enable','resource']}
        item['driver']=(dev/'driver').resolve().name if (dev/'driver').exists() else None
        item['iommu_group']=(dev/'iommu_group').resolve().name if (dev/'iommu_group').exists() else None
        item['sysfs_path']=str(dev.resolve());data['pci_devices'][dev.name]=item
        if item['class'] and (item['class'].startswith('0x0604') or item['class'].startswith('0x03')):item['verbose']=command(['lspci','-Dvv','-s',dev.name])
    data['pci_slots']={p.name:{x.name:read(x) for x in p.iterdir() if x.is_file()} for p in Path('/sys/bus/pci/slots').iterdir()}
    data['pmus']={p.name:{'type':read(p/'type'),'cpumask':read(p/'cpumask'),'events':{x.name:read(x) for x in (p/'events').glob('*')},'formats':{x.name:read(x) for x in (p/'format').glob('*')}} for p in Path('/sys/bus/event_source/devices').iterdir() if p.name.startswith(('uncore_imc','uncore_qpi','power','msr'))}
    data['runtime_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LAB/'build-avx-prefetch/bin/libggml-cpu.so',LAB/'review-glm-q8-columns4-20261020/build/bin/libggml-cpu.so']}
    (out/'inventory.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'output':str(out),'boot_id':data['boot_id'],'pci_slots':data['pci_slots'],'edac_controllers':list(data['edac']),'power_domains':list(data['telemetry']['powercap']),'pmus':list(data['pmus']),'GPU_devices':{k:{a:v[a] for a in ['numa_node','driver','max_link_speed','current_link_speed','current_link_width','physical_slot']} for k,v in data['pci_devices'].items() if (v['class'] or '').startswith('0x03')}},indent=2))
if __name__=='__main__':main()
