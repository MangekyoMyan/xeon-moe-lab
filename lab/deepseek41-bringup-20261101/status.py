#!/usr/bin/env python3
"""Read readiness once, without generating tokens or waiting for model loading."""
import json
from pathlib import Path
import subprocess
import urllib.request
import urllib.error
R=Path(__file__).resolve().parent
c=json.loads((R/'server-config.json').read_text())
print(subprocess.run(['systemctl','is-active','llm-deepseek41-api.service'],capture_output=True,text=True).stdout.strip())
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    key=Path(c['api_key_file']).read_text().strip()
    req=urllib.request.Request(f"http://{c['host']}:{c['port']}/health",headers={'Authorization':'Bearer '+key})
    with opener.open(req,timeout=5) as res:print(res.status,res.read().decode())
except urllib.error.HTTPError as e:print(e.code,e.read().decode()[:300])
except (OSError,urllib.error.URLError) as e:print(type(e).__name__,str(e))
print(f"URL: http://{c['host']}:{c['port']}/v1\nModel: {c['model_alias']}\nContext: {c['context']}")
