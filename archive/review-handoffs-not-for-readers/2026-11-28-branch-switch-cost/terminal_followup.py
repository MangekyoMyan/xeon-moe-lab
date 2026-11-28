"""Resolve the discrepancy between the first short terminal pairs and the RPC clock."""
import argparse
from study_lib import R,STORE,engine,window,read,write
from terminal_cost import measure

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["plan","run"]);a=p.parse_args()
    if a.action=="plan":
        rows=read(R/"evidence/terminal.json")
        pair={}
        for s in ("short","long"):
            pair[s]=[next(r["to_first_ms"] for r in rows if r["size"]==s and r["pair"]==i and r["version"]=="base")
                    -next(r["to_first_ms"] for r in rows if r["size"]==s and r["pair"]==i and r["version"]=="new") for i in range(2)]
        assert max(pair["short"])<0 and min(pair["long"])>0
        out={"primary_terminal_pairs":pair,"selected":["short"],"additional_pairs":list(range(2,10)),
             "reason":"Both initial short terminal pairs had the opposite direction to the paired RPC result. Add eight preordered base/new pairs on the uninstrumented private backend for this cell, then report all ten regardless of outcome. Long remains an exploratory terminal check of two pairs.",
             "timing":"two real Python CLI processes; begin before spawning checkout; end on first byte of the following text turn; warm, fresh outgoing image"}
        target=R/"TERMINAL_PLAN.json";assert not target.exists();write(target,out)
    else:
        design=read(R/"TERMINAL_PLAN.json")
        assert design["additional_pairs"]==list(range(2,10)) and design["selected"]==["short"]
        run=R/"runs/terminal-extra";assert not run.exists()
        with window(R,"BRANCH-COST-TERMINAL"),engine(R,STORE,run) as binding:
            measure(run,binding,sizes=("short",),pairs=range(2,10),receipt="terminal_extra.json")
        write(run/"DONE.json",{"complete":True,"scope":"8 additional short terminal pairs"})
