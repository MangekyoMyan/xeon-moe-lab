#!/usr/bin/env python3
"""One-shot local status; no LLM and no model loading."""
import datetime
import json
from pathlib import Path
import time

root = Path('/var/lib/xeon-watch')
try:
    s = json.loads((root / 'latest.json').read_text())
except FileNotFoundError:
    raise SystemExit('監視記録はまだありません。')
age = time.time() - s['unix']
gpu = s.get('gpu', {})
bmc = s.get('bmc', {})
dimms = [v for v in bmc.get('dimms_C', {}).values() if v is not None]
print(f'最終取得: {age:.0f}秒前' + (' [STALE: 監視停止の可能性]' if age > 60 else ''))
print('CPU最高:', max(s.get('cpu', {}).values(), default=None), '℃ / GPU:', gpu.get('gpu_C'), '℃ / HBM:', gpu.get('hbm_C'), '℃')
print('DIMM:', f'{min(dimms):.0f}〜{max(dimms):.0f}℃ ({len(dimms)}枚)' if dimms else '取得不可')
print('PCH:', bmc.get('pch_C'), '℃ / 周辺部:', bmc.get('peripheral_C'), '℃')
print('ファン出力:', [round(x, 1) for x in bmc.get('fan_percent', [])], '%')
print('継続アラーム:', ','.join(s.get('active_alerts', {})) or 'なし')
print('取得失敗:', s.get('errors') or 'なし')
print('上限:', s['monitor']['limits'])
print('記録:', root)
