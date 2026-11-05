#!/usr/bin/env python3
"""Disable THP only for this process and its children, then execute the command."""
import ctypes,os,sys
lib=ctypes.CDLL(None,use_errno=True)
if lib.prctl(41,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
with open('/proc/self/status') as f:
 assert any(line.strip()=='THP_enabled:\t0' for line in f), 'THP disable did not take effect'
if len(sys.argv)<2:raise SystemExit('Usage: without_thp.py COMMAND [ARG ...]')
os.execvp(sys.argv[1],sys.argv[1:])
