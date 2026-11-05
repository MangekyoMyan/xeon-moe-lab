#!/usr/bin/env python3
"""Record the existing BMC fan mode and select the full-speed commissioning mode."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
sys.path.insert(0, str(LAB / 'board-study-20261024'))
from hardware_run import ipmi

before = ipmi('raw', '0x30', '0x45', '0x00').strip()
assert before in ['00', '01', '02', '04'], before
report = {'before_mode': before, 'before_sensors': ipmi('sensor', 'list'), 'target_mode': '01',
          'reason': 'GPU fan uses motherboard add-on-card header; LAN fan airflow is reduced. User requested thermal care.',
          'restore_command': ['raw', '0x30', '0x45', '0x01', '0x' + before]}
destination = OUT / 'fan-mode-change.json'
assert not destination.exists()
destination.write_text(json.dumps(report, indent=2) + '\n')
if before != '01':
    report['set_response'] = ipmi('raw', '0x30', '0x45', '0x01', '0x01')
time.sleep(10)
report['after_mode'] = ipmi('raw', '0x30', '0x45', '0x00').strip()
report['after_sensors'] = ipmi('sensor', 'list')
destination.write_text(json.dumps(report, indent=2) + '\n')
owner = LAB.stat()
os.chown(destination, owner.st_uid, owner.st_gid)
assert report['after_mode'] == '01'
print('Fan mode', before, '->', report['after_mode'])
print('\n'.join(line for line in report['after_sensors'].splitlines() if line.startswith('FAN') or line.startswith(('System Temp', 'Peripheral Temp', 'PCH Temp'))))
