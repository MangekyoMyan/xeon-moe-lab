#!/usr/bin/env python3
"""Resolve a pinned, compute-only R580 driver without changing installed packages."""
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results' / ROOT.name
STAGE = ROOT / 'apt-stage'
STAGE.mkdir(exist_ok=True)
for rel in ['lists/partial', 'cache/archives/partial']:
    (STAGE / rel).mkdir(parents=True, exist_ok=True)
key_dir = Path('/tmp/llm-gpu-apt-key-20261025')
key_dir.mkdir(mode=0o755, exist_ok=True)
key = key_dir / 'cuda-archive-keyring.gpg'
shutil.copyfile(ROOT / 'references/debian12-keyring-extracted/usr/share/keyrings/cuda-archive-keyring.gpg', key)
key.chmod(0o644)
sources = STAGE / 'sources.list'
sources.write_text('''deb https://deb.debian.org/debian trixie main contrib non-free non-free-firmware
deb https://security.debian.org/debian-security trixie-security main contrib non-free non-free-firmware
deb https://deb.debian.org/debian trixie-updates main contrib non-free non-free-firmware
''' + f'deb [signed-by={key}] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /\n')
preferences = STAGE / 'preferences'
preferences.write_text('''Package: *
Pin: version 580.*
Pin-Priority: 700

Package: *
Pin: origin developer.download.nvidia.com
Pin-Priority: -1
''')
options = ['-o', 'Dir::Etc::sourcelist=' + str(sources), '-o', 'Dir::Etc::sourceparts=-',
           '-o', 'Dir::Etc::preferences=' + str(preferences), '-o', 'Dir::Etc::preferencesparts=-',
           '-o', 'Dir::State::lists=' + str(STAGE / 'lists'), '-o', 'Dir::Cache=' + str(STAGE / 'cache')]
packages = ['linux-headers-' + os.uname().release, 'dkms', 'ethtool', 'nvidia-kernel-dkms=580.196.07-1',
            'nvidia-smi=580.196.07-1', 'libnvidia-ml1=580.196.07-1', 'libcuda1=580.196.07-1']
commands = [['apt-get', *options, 'update'], ['apt-get', *options, '-s', '--no-install-recommends', 'install', *packages]]
record = {'options': options, 'packages': packages, 'commands': commands,
          'scope': 'NVIDIA-only Debian12 repository; Debian operating-system packages remain Debian13. Other NVIDIA versions pinned out. No installation in this script.'}
(OUT / 'apt-stage-plan.json').write_text(json.dumps(record, indent=2) + '\n')
for name, command in zip(['apt-stage-update', 'apt-stage-simulate'], commands):
    print(name, flush=True)
    with (OUT / (name + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
print('Resolution complete; no installed packages changed', flush=True)
