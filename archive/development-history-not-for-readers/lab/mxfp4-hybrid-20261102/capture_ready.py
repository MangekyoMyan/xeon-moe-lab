#!/usr/bin/env python3
"""User-launch helper: write a readiness boundary without model polling in chat."""
from pathlib import Path
import argparse
import http.client
import json
import time

ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--capture',type=Path,required=True);a=ap.parse_args()
config=json.loads(a.config.read_text());key=Path(config['api_key_file']).read_text().strip()
host=config['host'];host='127.0.0.1' if host in ['0.0.0.0','::'] else host
start=time.monotonic();last_error=None
while time.monotonic()-start<1800:
    connection=None
    try:
        connection=http.client.HTTPConnection(host,config['port'],timeout=2)
        connection.request('GET','/health',headers={'Authorization':'Bearer '+key})
        response=connection.getresponse();payload=response.read(65536)
        if response.status==200 and json.loads(payload).get('status')=='ok':
            ready=a.capture/'READY.json';temporary=a.capture/'READY.tmp'
            temporary.write_text(json.dumps({'ready':True,'monotonic_s':time.monotonic(),'unix':time.time(),
              'note':'Histogram rows earlier than this boundary are excluded by the analyzer.'},indent=2)+'\n')
            temporary.replace(ready);break
        last_error='HTTP '+str(response.status)
    except Exception as error:last_error=type(error).__name__
    finally:
        if connection is not None:connection.close()
    time.sleep(2)
else:
    (a.capture/'READY.json').write_text(json.dumps({'ready':False,'timeout_s':1800,'last_error_type':last_error})+'\n')
