#!/usr/bin/env python3
"""Read-only GPU, NIC, sensor, kernel and package inventory after installation."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name

def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

def command(args):
    executable = shutil.which(args[0])
    if not executable:
        return {'available': False}
    p = subprocess.run([executable, *args[1:]], capture_output=True, text=True, timeout=40)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'boot_id': read('/proc/sys/kernel/random/boot_id'), 'uptime': read('/proc/uptime'),
              'kernel': os.uname().release, 'pci': [], 'interfaces': [], 'sensors': []}
    for dev in sorted(Path('/sys/bus/pci/devices').iterdir()):
        vendor, device = read(dev / 'vendor'), read(dev / 'device')
        if vendor != '0x10de' and (vendor, device) != ('0x8086', '0x1528'):
            continue
        fields = ['vendor', 'device', 'class', 'subsystem_vendor', 'subsystem_device', 'numa_node', 'local_cpulist',
                  'current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width', 'enable',
                  'aer_dev_correctable', 'aer_dev_nonfatal', 'aer_dev_fatal']
        info = {'bdf': dev.name, **{f: read(dev / f) for f in fields},
                'driver': (dev / 'driver').resolve().name if (dev / 'driver').exists() else None,
                'lspci': command(['lspci', '-Dnnk', '-s', dev.name]),
                'link_detail': command(['lspci', '-Dvv', '-s', dev.name])}
        report['pci'].append(info)
    for interface in sorted(Path('/sys/class/net').iterdir()):
        dev = (interface / 'device').resolve()
        if (read(dev / 'vendor'), read(dev / 'device')) != ('0x8086', '0x1528'):
            continue
        row = {'name': interface.name, 'bdf': dev.name,
               **{f: read(interface / f) for f in ['operstate', 'carrier', 'speed', 'duplex', 'mtu']},
               'ethtool': command(['ethtool', interface.name]),
               'driver': command(['ethtool', '-i', interface.name]),
               'statistics': command(['ethtool', '-S', interface.name]),
               'statistics_sysfs': {p.name: read(p) for p in (interface / 'statistics').glob('*')}}
        report['interfaces'].append(row)
    for folder in sorted(Path('/sys/class/hwmon').glob('*')):
        values = {p.name: read(p) for p in folder.iterdir() if p.is_file() and p.name.startswith(('temp', 'fan'))}
        report['sensors'].append({'path': str(folder.resolve()), 'name': read(folder / 'name'), 'values': values})
    report['nvidia_smi'] = command(['nvidia-smi', '-q', '-x'])
    report['nvidia_proc'] = read('/proc/driver/nvidia/version')
    report['kernel_log'] = command(['journalctl', '-k', '-b', '--no-pager', '-o', 'short-monotonic'])
    report['modules'] = command(['lsmod'])
    report['dkms'] = command(['dkms', 'status'])
    report['packages'] = command(['dpkg-query', '-W', '-f=${Package}\t${Version}\t${db:Status-Status}\n',
                                  '*nvidia*', '*cuda*', 'linux-headers-*', 'dkms'])
    report['thermal_zones'] = [{**{k: read(p / k) for k in ['type', 'temp']}, 'path': str(p)}
                               for p in Path('/sys/class/thermal').glob('thermal_zone*')]
    destination = OUT / 'inventory-before.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    owner = LAB.stat()
    os.chown(destination, owner.st_uid, owner.st_gid)
    print(json.dumps({'boot_id': report['boot_id'], 'uptime': report['uptime'],
                      'pci': [{k: d[k] for k in ['bdf', 'vendor', 'device', 'driver', 'numa_node', 'current_link_speed', 'current_link_width']} for d in report['pci']],
                      'interfaces': [{k: i[k] for k in ['name', 'bdf', 'operstate', 'speed']} for i in report['interfaces']],
                      'sensor_names': [s['name'] for s in report['sensors']],
                      'nvidia_smi_available': report['nvidia_smi'].get('available', True)}, indent=2), flush=True)

if __name__ == '__main__':
    main()
