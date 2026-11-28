"""Complete a conditional supplement and numerical/failure controls in one maintenance window."""
import argparse
import signal
import sys
import traceback
from trial import R,STORE,window,engine,read,write
from supplement import measure_extra,verify as verify_extra
from validation import VS,candidate,stock,loss_case

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--attempt",type=int,default=1);a=p.parse_args()
    def stop(sig,frame):raise KeyboardInterrupt(sig)
    for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    ext=read(R/"SUPPLEMENT.json")
    index=read(R/"VALIDATION_RUNS.json") if (R/"VALIDATION_RUNS.json").exists() else {}
    with window(R,"ADAPTIVE-CONTROLS"):
        if not (R/"EXTRA.json").exists():
            if ext["order"]:
                run=R/"runs"/f"supplement-{a.attempt}"
                assert not run.exists()
                with engine(R,STORE,run) as binding:
                    try:measure_extra(run,binding,ext)
                    except BaseException:(run/"FAILURE.txt").write_text(traceback.format_exc());traceback.print_exc();raise
                write(run/"DONE.json",{"complete":True})
            write(R/"EXTRA.json",verify_extra(R))
        for stage,fn,store in (("candidate",candidate,VS),("lost",loss_case,VS),("stock",stock,R/"stock-store")):
            if stage in index:continue
            if stage=="lost" and (R/"evidence/uncertain.json").exists():continue
            run=R/"runs"/f"{stage}-{a.attempt}";assert not run.exists()
            with engine(R,store,run,observed=stage!="lost",stock=stage=="stock") as binding:
                try:fn(run,binding)
                except BaseException:(run/"FAILURE.txt").write_text(traceback.format_exc());traceback.print_exc();raise
            write(run/"DONE.json",{"complete":True});index[stage]=run.name;write(R/"VALIDATION_RUNS.json",index)
