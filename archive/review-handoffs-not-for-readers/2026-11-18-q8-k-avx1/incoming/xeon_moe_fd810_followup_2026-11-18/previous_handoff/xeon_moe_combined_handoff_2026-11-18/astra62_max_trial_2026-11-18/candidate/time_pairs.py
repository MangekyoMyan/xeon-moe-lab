#!/usr/bin/env python3
"""Run independent processes in alternated A/B and B/A order."""
import argparse, csv, os, pathlib, subprocess, sys

cases = {
    # variant, type, K, rows, experts, slots, tokens, activation columns, threads, cache, pattern, loops
    "iq-tiny": ("iq","iq3_xxs",256,17,8,4,1,1,1,1,3,1800),
    "iq-xxs-long": ("iq","iq3_xxs",7168,512,8,4,1,1,1,1,3,9),
    "iq-2s-short": ("iq","iq2_s",1536,768,8,4,1,4,1,1,3,24),
    "iq-3s-long": ("iq","iq3_s",7168,512,8,4,1,1,1,1,3,9),
    "iq-3s-down": ("iq","iq3_s",1536,768,8,4,1,4,1,1,3,24),
    "iq-xxs-3t": ("iq","iq3_xxs",7168,512,8,4,1,1,3,1,3,12),
    "iq-rotating": ("iq","iq3_s",4096,768,32,4,1,1,1,1,5,16),
    "route-many": ("route","f32",8,1,256,8,512,1,1,0,3,70),
    "route-tiny": ("route","f32",8,8,256,8,1,1,1,0,3,1800),
    "route-mxfp4": ("route","mxfp4",256,32,64,4,64,1,1,0,0,36),
}
parser = argparse.ArgumentParser()
parser.add_argument("probe", type=pathlib.Path)
parser.add_argument("libroot", type=pathlib.Path, help="contains lib-baseline, lib-route, lib-iq")
parser.add_argument("--rounds", type=int, default=6)
parser.add_argument("--samples", type=int, default=3)
parser.add_argument("--cpu1", default="2")
parser.add_argument("--cpu3", default="2-4")
parser.add_argument("--cases", nargs="+", choices=list(cases), default=list(cases))
a = parser.parse_args()
writer = csv.writer(sys.stdout)
writer.writerow(["case","round","order","variant","sample","ns_per_op","work_bytes","output_hash","seed"])
for rid in range(a.rounds):
    seed = 67913 + rid
    for name in a.cases:
        candidate, *params = cases[name]
        pair = ["baseline", candidate] if rid%2 == 0 else [candidate, "baseline"]
        for order, variant in enumerate(pair):
            env = dict(os.environ, LD_LIBRARY_PATH=str((a.libroot/("lib-" + variant)).resolve()), OMP_DYNAMIC="false")
            allowed = a.cpu1 if params[7] == 1 else a.cpu3
            command = ["taskset","-c",allowed,str(a.probe.resolve()),"bench",*map(str,params),str(a.samples),str(seed)]
            rows = subprocess.check_output(command, env=env, text=True).splitlines()
            for row in rows:
                sample, ns, size, digest = row.split(",")
                writer.writerow([name,rid,order,variant,sample,ns,size,digest,seed])
            sys.stdout.flush()
