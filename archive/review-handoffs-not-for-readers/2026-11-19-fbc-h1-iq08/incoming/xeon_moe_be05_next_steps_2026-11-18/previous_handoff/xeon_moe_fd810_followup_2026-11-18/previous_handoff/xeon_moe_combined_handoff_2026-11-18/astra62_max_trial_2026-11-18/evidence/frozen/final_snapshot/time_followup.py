#!/usr/bin/env python3
"""Paired verification of the revised route, quantizer and complete experimental build."""
import argparse, csv, os, pathlib, subprocess, sys

# Graph columns: type K M E S T A threads cache pattern loops. Quantizer: K rows loops.
cases = {
 "route2-tiny":("route2","graph",["f32",8,8,256,8,1,1,1,0,3,2400]),
 "route2-many":("route2","graph",["f32",8,1,256,8,512,1,1,0,3,80]),
 "route2-mxfp4":("route2","graph",["mxfp4",256,32,64,4,64,1,1,0,0,40]),
 "q8-256":("q8","quant",[256,16,1400]),
 "q8-7168":("q8","quant",[7168,8,95]),
 "q8-small-graph":("q8","graph",["iq2_s",1536,1,8,8,1,8,1,1,3,500]),
 "q8-wide-graph":("q8","graph",["iq2_s",1536,768,8,4,1,4,1,1,3,30]),
 "shape-iq2s-gate":("iq","graph",["iq2_s",5120,2304,6,6,1,1,1,1,3,4]),
 "shape-xxs-gate":("iq","graph",["iq3_xxs",5120,2304,6,6,1,1,1,1,3,4]),
 "shape-3s-down":("iq","graph",["iq3_s",2304,5120,6,6,1,6,1,1,3,4]),
 "shape-xxs-down-3t":("iq","graph",["iq3_xxs",2304,5120,12,6,1,6,3,1,5,4]),
 "shape-xxs-gate-combo":("combined","graph",["iq3_xxs",5120,2304,6,6,1,1,1,1,3,4]),
 "shape-q8-gate":("q8","graph",["iq3_xxs",5120,2304,6,6,1,1,1,1,3,4]),
 "safe-iq2s-gate":("iqsafe","graph",["iq2_s",5120,2304,6,6,1,1,1,1,3,4]),
 "safe-xxs-gate":("iqsafe","graph",["iq3_xxs",5120,2304,6,6,1,1,1,1,3,4]),
 "safe-3s-down":("iqsafe","graph",["iq3_s",2304,5120,6,6,1,6,1,1,3,4]),
 "safe-xxs-down-3t":("iqsafe","graph",["iq3_xxs",2304,5120,12,6,1,6,3,1,5,4]),
 "safe-combo-gate":("combosafe","graph",["iq3_xxs",5120,2304,6,6,1,1,1,1,3,4]),
 "combo-rotating":("combined","graph",["iq3_s",4096,768,32,4,1,1,1,1,5,16]),
}
p=argparse.ArgumentParser()
p.add_argument("root",type=pathlib.Path,help="contains probes and lib-* directories")
p.add_argument("--rounds",type=int,default=8)
p.add_argument("--samples",type=int,default=5)
p.add_argument("--cpu1",default="2")
p.add_argument("--cpuN",default="2-4")
p.add_argument("--cases", nargs="+",choices=list(cases),default=[name for name in cases if not name.startswith("safe-")])
a=p.parse_args()
w=csv.writer(sys.stdout)
w.writerow(["case","round","order","variant","sample","ns_per_op","work_bytes","output_hash","seed"])
for rid in range(a.rounds):
    seed=90791+rid
    for name in a.cases:
        variant,kind,params=cases[name]
        pair=["baseline",variant] if rid%2==0 else [variant,"baseline"]
        for order,v in enumerate(pair):
            env=dict(os.environ, LD_LIBRARY_PATH=str((a.root/("lib-"+v)).resolve()),OMP_DYNAMIC="false")
            path=a.root/("graph_probe" if kind=="graph" else "q8_probe")
            cpu=a.cpu1 if kind=="quant" or params[7]==1 else a.cpuN
            command=["taskset","-c",cpu,str(path.resolve()),"bench",*map(str,params),str(a.samples),str(seed)]
            out=subprocess.check_output(command,text=True,env=env)
            for row in out.splitlines():
                sample,ns,size,digest=row.split(",")
                w.writerow([name,rid,order,v,sample,ns,size,digest,seed])
            sys.stdout.flush()
