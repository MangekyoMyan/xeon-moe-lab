#!/usr/bin/env python3
"""Read-only DRAM RAPL energy-to-power display. No daemon or power-limit writes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--count', type=int, default=0, help='0: continuous')
    args = parser.parse_args()
    if not 0.2 <= args.interval <= 60 or args.count < 0:
        parser.error('interval must be 0.2..60 seconds; count must be >= 0')

    root = Path('/sys/class/powercap')
    zones = []
    for path in sorted(root.glob('intel-rapl:*')):
        if (path / 'name').read_text().strip() != 'dram':
            continue
        parent = root / path.name.rsplit(':', 1)[0]
        label = (parent / 'name').read_text().strip()
        zones.append((path, label, int((path / 'max_energy_range_uj').read_text())))
    if not zones:
        raise SystemExit('DRAM RAPLカウンターが見つかりません。')

    def sample():
        values = []
        for path, _, _ in zones:
            before = time.monotonic()
            energy = int((path / 'energy_uj').read_text())
            values.append((energy, (before + time.monotonic()) / 2))
        return values

    previous = sample()
    cumulative_joules = [0.0] * len(zones)
    cumulative_seconds = [0.0] * len(zones)
    peak = 0.0
    count = 0
    while args.count == 0 or count < args.count:
        time.sleep(args.interval)
        current = sample()
        watts = []
        for i, ((old, t0), (new, t1), (_, _, wrap)) in enumerate(zip(previous, current, zones)):
            delta = new - old
            if delta < 0:
                delta += wrap
            joules = delta / 1_000_000
            elapsed = t1 - t0
            watts.append(joules / elapsed)
            cumulative_joules[i] += joules
            cumulative_seconds[i] += elapsed
        total = sum(watts)
        peak = max(peak, total)
        average = sum(j / s for j, s in zip(cumulative_joules, cumulative_seconds))
        if sys.stdout.isatty():
            print('\033[2J\033[H', end='')
        print(datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S JST'), 'メモリ電力 / DRAM RAPL', flush=True)
        for (_, label, _), power in zip(zones, watts):
            print(f'  {label}側のメモリ: {power:7.2f} W')
        print(f'  メモリ合計:         {total:7.2f} W  （直近約{args.interval:g}秒の平均）')
        print(f'  表示開始からの平均: {average:7.2f} W / 最大: {peak:.2f} W')
        print('RAPLのDRAM領域の参考値。DIMMごとの値や、電源入力での実測値ではありません。')
        print('終了: Ctrl+C', flush=True)
        previous = current
        count += 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n終了しました。')
    except PermissionError:
        raise SystemExit('カウンター読取りにはsudoが必要です。sudo python3 ' + str(Path(__file__).resolve()))
    except OSError as error:
        raise SystemExit('カウンターを読み取れません: ' + str(error))
