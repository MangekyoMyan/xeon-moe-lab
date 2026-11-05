#!/usr/bin/env python3
"""Download official package metadata and upstream NIC source for inspection."""
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'references'
DEST.mkdir(exist_ok=True)
urls = {
    'nvidia-debian13-Packages.gz': 'https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/Packages.gz',
    'cuda-keyring_1.1-1_all.deb': 'https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb',
    'supportedchips.html': 'https://download.nvidia.com/XFree86/Linux-x86_64/580.196.07/README/supportedchips.html',
    'ixgbe_sysfs.c': 'https://raw.githubusercontent.com/torvalds/linux/v6.12/drivers/net/ethernet/intel/ixgbe/ixgbe_sysfs.c',
    'ixgbe_x540.c': 'https://raw.githubusercontent.com/torvalds/linux/v6.12/drivers/net/ethernet/intel/ixgbe/ixgbe_x540.c',
}
manifest = []
for name, url in urls.items():
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
        (DEST / name).write_bytes(data)
        manifest.append({'name': name, 'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        print(name, len(data), flush=True)
    except Exception as exc:
        manifest.append({'name': name, 'url': url, 'error': str(exc)})
        print(name, str(exc), flush=True)
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
path = DEST / 'nvidia-debian13-Packages.gz'
if path.exists():
    text = gzip.decompress(path.read_bytes()).decode()
    records = []
    for block in text.split('\n\n'):
        fields = {}
        for line in block.splitlines():
            if ': ' in line and not line.startswith(' '):
                key, value = line.split(': ', 1)
                fields[key] = value
        if fields.get('Version', '').startswith('580.') or fields.get('Package') == 'nvidia-driver-pinning-580':
            records.append(fields)
    (DEST / 'r580-packages.json').write_text(json.dumps(records, indent=2) + '\n')
    print('R580 package records', len(records), flush=True)
