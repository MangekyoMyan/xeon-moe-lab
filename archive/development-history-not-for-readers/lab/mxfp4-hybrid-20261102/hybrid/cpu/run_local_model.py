#!/usr/bin/env python3
"""Optional CPU-only server for a model the reviewer already has. No downloads."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser()
ap.add_argument('--model',type=Path,required=True)
ap.add_argument('--profile',choices=['portable','ivybridge','native'],default='portable')
ap.add_argument('--threads',type=int,default=2)
ap.add_argument('--context',type=int,default=4096)
ap.add_argument('--port',type=int,default=18080)
a=ap.parse_args()
if not a.model.is_file():ap.error('model must be an existing local GGUF entry shard')
if not 1<=a.threads<=64 or not 256<=a.context<=131072 or not 1024<=a.port<=65535:ap.error('invalid resource bound')
binary=ROOT/('build-cpu-'+a.profile)/'bin/llama-server'
if not binary.is_file():ap.error('build cpu/build.py --target engine first')
env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_')) and k!='LD_PRELOAD'}
subprocess.run([str(binary),'-m',str(a.model.resolve()),'-ngl','0','--no-kv-offload','--no-op-offload',
    '--fit','off','--offline','--no-repack','-c',str(a.context),'-t',str(a.threads),'-tb',str(a.threads),
    '--host','127.0.0.1','--port',str(a.port),'--parallel','1','--no-cont-batching','--no-context-shift',
    '--cache-prompt','--cache-ram','0','--ctx-checkpoints','0','--jinja','--reasoning','off',
    '--chat-template-kwargs','{"thinking_mode":"chat"}','--no-reasoning-preserve','--no-ui','--no-agent','--no-ui-mcp-proxy'],env=env,check=True)
