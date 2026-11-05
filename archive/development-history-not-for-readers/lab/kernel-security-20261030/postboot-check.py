#!/usr/bin/python3
"""One boot's bounded validation; deliberately does not promote the trial kernel."""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

BASE = Path('/home/codex/projects/llm-lab/kernel-security-20261030')
STATE = Path('/var/lib/kernel-security-20261030')
R29 = Path('/home/codex/projects/llm-lab/moe-engine-20261029')
EXPECTED = '6.12.113+deb13-amd64'
GPU = 'GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3'


def main():
    STATE.mkdir(mode=0o755, exist_ok=True)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    result = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  boot_id=boot, kernel=os.uname().release, checks=[], passed=False,
                  rollback_default_preserved=True)

    def run(args, timeout=30):
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        result['checks'].append(dict(command=args, returncode=p.returncode, output=p.stdout))
        print(p.stdout, flush=True)
        if p.returncode:
            raise RuntimeError(f'{args[0]} exited {p.returncode}')
        return p.stdout.strip()

    try:
        assert os.uname().release == EXPECTED, 'Trial kernel did not boot; leave rollback intact'
        version = run(['dpkg-query', '-W', '-f=${Version}', 'linux-image-' + EXPECTED])
        run(['dpkg', '--compare-versions', version, 'ge', '6.12.113-1'])
        assert '580.196.07' == Path('/sys/module/nvidia/version').read_text().strip()
        assert f'nvidia/580.196.07, {EXPECTED}, x86_64: installed' in run(['dkms', 'status'])
        gpu = run(['nvidia-smi', '-i', GPU, '--query-gpu=uuid,driver_version,power.limit,ecc.mode.current',
                   '--format=csv,noheader,nounits']).split(', ')
        assert gpu == [GPU, '580.196.07', '150.00', 'Enabled'], gpu
        assert run(['systemctl', 'is-active', 'ssh']) == 'active'
        # network-online.target can precede carrier on the second, static-IP NIC.
        # Preserve the check; allow bounded time for physical link negotiation.
        start = time.monotonic()
        while True:
            addresses = json.loads(run(['ip', '-json', 'address', 'show', 'enp4s0f1']))
            ready = bool(addresses) and addresses[0]['operstate'] == 'UP' and any(
                a.get('local') == '10.10.10.2' for a in addresses[0]['addr_info'])
            if ready:
                break
            if time.monotonic() - start >= 120:
                raise RuntimeError('10.10.10.2 physical link not ready within 120 seconds')
            time.sleep(2)
        result['network_link_wait_seconds'] = time.monotonic() - start
        with socket.create_connection(('10.10.10.2', 22), timeout=5) as connection:
            banner = connection.recv(256).decode(errors='replace').strip()
            assert banner.startswith('SSH-2.0-'), banner
            result['local_ssh_banner'] = banner
        result['external_ssh_reconnect_verified'] = False
        result['gpu_tests'] = {}
        for variant in ['baseline', 'warp32']:
            label = f'kernel-security-postboot-{boot[:8]}-{os.getpid()}-{variant}'
            run(['/usr/bin/python3', str(R29 / 'watch.py'), '--label', label,
                 '--timeout', '120', '--power-limit', '150', '--',
                 str(R29 / 'experiments/kernel-agent' / ('check-' + variant))], timeout=150)
            output = BASE.parent / 'results' / R29.name / label
            check = json.loads((output / 'stdout.txt').read_text())
            assert check['cases'] == 508 and check['bitexact_CPU_AVX'] is True, check
            record = json.loads((output / 'run.json').read_text())
            assert record['returncode'] == 0 and 'error' not in record, record.get('error')
            result['gpu_tests'][variant] = dict(result=check, evidence=str(output))
        result['kernel_journal'] = run(['journalctl', '-k', '-b', '--no-pager'], timeout=30)
        severe = [line for line in result['kernel_journal'].splitlines()
                  if any(s in line for s in ['NVRM: Xid', 'BUG:', 'Oops:', 'Kernel panic'])]
        assert not severe, severe
        run(['nvidia-smi', '-q'])
        run(['ip', '-brief', 'address'])
        result['passed'] = True
        result['next_step'] = 'Verify external reconnection, review result, then remove temporary GRUB default guard and run update-grub.'
    except BaseException:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data = json.dumps(result, indent=2) + '\n'
        temporary = STATE / 'postboot-result.json.tmp'
        temporary.write_text(data)
        temporary.replace(STATE / 'postboot-result.json')
        (BASE / 'postboot-result.json').write_text(data)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
