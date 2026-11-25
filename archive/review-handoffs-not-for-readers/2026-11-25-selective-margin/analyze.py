"""Compute the selectivity and end-to-end cost of the frozen margin policy."""
import array
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parent; LAB=ROOT.parent; PRIOR=LAB/"worker-balance-20261124"
def read(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
def table(p):
    with p.open() as s:return list(csv.DictReader(s,delimiter="\t"))
def records(run):return [json.loads(s) for s in (run/"measurements.jsonl").read_text().splitlines()]
def median(v):return statistics.median(v)
def q(v,p):return sorted(v)[math.ceil(p*len(v))-1]
def safe(a,b):return a/b if b else None

def confusion(flags,outcomes):
    assert len(flags)==len(outcomes)
    n=len(flags);nf=sum(flags);nc=sum(outcomes);tp=sum(f and c for f,c in zip(flags,outcomes))
    return {"positions":n,"flagged":nf,"changes":nc,"true_positive":tp,"false_positive":nf-tp,
            "misses":nc-tp,"precision":safe(tp,nf),"recall":safe(tp,nc),"trigger_rate":safe(nf,n)}

def local_error():
    design=read(ROOT/"DESIGN.json");by=collections.defaultdict(lambda:[0.,0.,0]);matched=0
    prior=LAB/"volta-residual-20261125";frozen=prior/read(prior/"MEASURED-forward.json")["run"]
    refs={(v["fixture"],v["projection"],int(v["token"]),int(v["slot"]),int(v["expert"]),int(v["row"])):(float(v["full"]),float(v["base"])) for v in table(frozen/"dots.tsv")}
    for v in table(ROOT/"dot_reference.tsv"):
        t=int(v["token"]);gold=float(v["F64"]);actual=float(v["GPU"]);k=(v["case"],t)
        by[k][0]+=(actual-gold)**2;by[k][1]+=gold*gold;by[k][2]+=1
        loc=(v["case"],v["projection"],t,int(v["slot"]),int(v["expert"]),int(v["row"]))
        if loc in refs:
            same,expected=refs[loc];assert abs(same-gold)<1e-8 and actual==expected;matched+=1
    assert matched==len(refs)==4608 and sum(v[2] for v in by.values())==73728
    out={}
    for tag,rows in design["L0_frozen"].items():
        assert len(rows)==96 and all(by[tag,t][2]==384 for t in range(96))
        flags=[v["gap"]<=design["threshold"] for v in rows];v=[by[tag,t][0] for t in range(96)]
        rotations=[sum(x for i,x in enumerate(v) if flags[(i+s)%96])/sum(v) for s in range(96)]
        out[tag]={"N":96,"marked":sum(flags),"selected_fraction":sum(flags)/96,
                  "share_of_baseline_squared_error":rotations[0],"same_budget_rotation37":rotations[37],
                  "circular_shift_median":median(rotations),"shifts_ge_near":sum(x>=rotations[0] for x in rotations),
                  "per_token":[{"token":t,"gap":rows[t]["gap"],"near":flags[t],"SSE":v[t],"reference_energy":by[tag,t][1]} for t in range(96)]}
    return {"reference_dots":73728,"prior_dots_reproduced":matched,"cases":out,"scope":"32 rows per expert projection, all positions. SSE share is the upper bound on removable ordinary squared error in that subset if corrected exactly."}

def router():
    design=read(ROOT/"DESIGN.json");items={};groups={};st=struct.Struct("<iifi")
    for case in ("JA96","PY96"):
        data=(ROOT/f"runs/candidate/q_D_{case}.gaps").read_bytes();assert len(data)==40*96*st.size
        raw=list(st.iter_unpack(data));vals=[]
        for layer in range(40):
            seq=[]
            for t in (64,128):
                a=array.array("i");a.frombytes((PRIOR/f"runs/precision-{t}-{t}"/(case+"-routes")/f"ffn_moe_down-{layer}-ids.bin").read_bytes())
                assert len(a)==576;seq.append(a)
            for pos in range(96):
                six,seven,d,f=raw[layer*96+pos];x=list(seq[0][6*pos:6*pos+6]);y=list(seq[1][6*pos:6*pos+6])
                assert six==x[-1] and seven not in x and d>=0 and math.isfinite(d) and f==(d<=design["threshold"])
                if not layer:
                    v=design["L0_frozen"][case][pos]
                    assert v["sixth"]==six and v["seventh"]==seven and abs(v["gap"]-d)<5e-6
                    assert bool(f)==(v["gap"]<=design["threshold"])
                vals.append((layer,pos,d,set(x)!=set(y)))
        items[case]=vals
    per={str(f):[q([v[2] for v in items["JA96"] if v[0]==l],f) for l in range(40)] for f in (.05,.1,.2)}
    for case,vs in items.items():
        label=[v[3] for v in vs];near=[v[2]<=design["threshold"] for v in vs]
        groups[case]={"absolute_L0_cut_across_all_layers":confusion(near,label),
            "per_layer_JA_calibrated":{f:confusion([v[2]<=cuts[v[0]] for v in vs],label) for f,cuts in per.items()},
            "L0_gap_predicts_next_layer_set_change":confusion(near[:96],label[96:192]),
            "median_gap_changed":median([v[2] for v in vs if v[3]]),"median_gap_stable":median([v[2] for v in vs if not v[3]])}
    return {"calibration":"JA only","cut_by_layer":per,"cases":groups,
            "label":"Top-six set mismatch on the same fixed inputs between existing 64 and 128 arithmetic policies. Sensitivity, not an error label."}

def clocks():
    runs={s:records(ROOT/"runs"/s) for s in ("F0","candidate","F1")}
    eligible=[k for k,v in read(ROOT/"driver/FIXTURES.json")["cases"].items() if 64<=len(v["prefix"])<128]
    by={}
    for n in eligible:
        d={}
        for arm in ("F0","F1","D","S","R","A"):
            es=[v for v in runs.get(arm,runs["candidate"]) if v["case"]==n and (arm in ("F0","F1") or v["arm"]==arm)]
            assert len(es)==(2 if arm in ("D","S") else 1)
            d[arm]=[(v["ready_ns"]-v["start_ns"])/1e6 for v in es]
        d["reference"]=(d["F0"][0]+d["F1"][0])/2
        near={k:next(v for v in runs["candidate"] if v["arm"]==k and v["case"]==n) for k in "DSRA"}
        h=near["D"]["layer0_near_tokens"];N=near["D"]["prefix"]
        for arm,r in near.items():
            assert r["layer0_near_tokens"]==h and r["native_counts"][2:4]==[6*(N if arm=="A" else h)]*2
            assert r["native_counts"][4:]==[0 if arm=="D" else 6*(N if arm=="A" else h)]*2
        by[n]={"ms":d,"prefix":N,"near":h,"corrected_expert_columns_per_projection":6*h}
    aggregate={}
    for g,names in [("all",eligible),("held_chats",[n for n in eligible if n not in ("JA96","PY96")])]:
        a={k:statistics.mean(statistics.mean(by[n]["ms"][k]) for n in names) for k in ("F0","F1","D","S","R","A")}
        a["reference"]=(a["F0"]+a["F1"])/2
        aggregate[g]={"cases":len(names),"mean_ms":a,"D_vs_F_pct":100*(a["D"]/a["reference"]-1),"S_vs_F_pct":100*(a["S"]/a["reference"]-1),
             "S_vs_D_pct":100*(a["S"]/a["D"]-1),"A_vs_F_pct":100*(a["A"]/a["reference"]-1),
             "D_mean_above_both_F_cases":sum(statistics.mean(by[n]["ms"]["D"])>max(by[n]["ms"]["F0"][0],by[n]["ms"]["F1"][0]) for n in names),
             "near_count":sum(by[n]["near"] for n in names),"tokens":sum(by[n]["prefix"] for n in names)}
    return {"cases":by,"aggregate":aggregate,"scope":"Whole-model cold-prefix time to output-ready. F0/F1 fresh bracketing ordinary processes. D/S/R/A share one isolated process; repeated D/S. Native flag and dispatch work included; verification dumps outside."}

def numerics():
    fx=read(ROOT/"driver/FIXTURES.json")["cases"];cand=ROOT/"runs/candidate";f0=ROOT/"runs/F0";dest=ROOT/"risks";dest.mkdir(exist_ok=True)
    labels={"F-S":("F","S"),"F-R":("F","R"),"F-A":("F","A"),"S-A":("S","A"),"R-A":("R","A")}
    parsed={};gate={}
    for tag,(aa,bb) in labels.items():
        sub=dest/tag;sub.mkdir(exist_ok=True);grouped=collections.defaultdict(list)
        for n,c in fx.items():
            if not 64<=len(c["prefix"])<128:continue
            paths=[(f0 if arm=="F" else cand)/f"q_{arm}_{n}.f32" for arm in (aa,bb)]
            targets=sub/(n+".ids");targets.write_text("\n".join(map(str,c["targets"]))+"\n");tsv=sub/(n+".tsv")
            subprocess.run([str(PRIOR/"precision/score_logits"),str(paths[0]),str(paths[1]),str(targets),str(tsv)],check=True)
            rs=table(tsv);assert len(rs)==len(c["targets"])
            for v in rs:v["case"]=n
            grouped["fixed96" if n in ("JA96","PY96") else "held_chats"].extend(rs)
        info={}
        for g,rs in grouped.items():
            flips=[v for v in rs if v["A_top"]!=v["B_top"]]
            info[g]={"rows":len(rs),"byte_exact":sum(int(v["exact"]) for v in rs),"top_changes":len(flips),
                    "mean_TV":statistics.mean(float(v["TV"]) for v in rs),
                    "mean_target_NLL_delta":statistics.mean(float(v["B_target_nll"])-float(v["A_target_nll"]) for v in rs),
                    "reference_top1_A":sum(v["target"]==v["A_top"] for v in rs),"reference_top1_B":sum(v["target"]==v["B_top"] for v in rs),
                    "switched":[[v["case"],int(v["row"]),int(v["A_top"]),int(v["B_top"])] for v in flips]}
            if tag=="F-A":
                gate[g]={str(t):confusion([float(v["A_margin"])<=t for v in rs],[v["A_top"]!=v["B_top"] for v in rs]) for t in (.05,.10,.20,.25,.50,1.)}
        parsed[tag]=info
    sys.path.insert(0,str(PRIOR));from score_chat import score
    source={c["id"]:c for c in read(PRIOR/"CHAT_FIXTURES.json")};tasks={}
    for n,c in source.items():
        tasks[n]={}
        for arm in ("F","D","S","R","A"):
            folder=f0 if arm=="F" else cand
            rows=[x for x in records(folder) if x["case"]==n and x["quality"] and x["arm"]==arm]
            if not rows:continue
            assert len(rows)==1;v=rows[0];text=v["free_text"]
            tasks[n][arm]={"score":score(c,{"text":text,"finish":"stop" if v.get("new_free_eog",True) else "length"})["passes"],
                "same_text_as_reference":text==fx[n]["reference_text"],"text":text}
    return {"compared":parsed,"output_gap_vs_actual_all_residual":gate,"practical":tasks}

def sampler():
    meta=read(ROOT/"SAMPLER_INPUTS.json");ts=table(ROOT/"output_cost.tsv")
    totals=collections.defaultdict(list);rset=collections.defaultdict(dict)
    for v in ts:rset[int(v["rep"]),int(v["row"])][v["method"]]=v
    names=[(n,i) for n in meta["order"] for i in range(meta["rows"][n])]
    assert len(rset)==6*len(names) and len(ts)==18*len(names)
    byfile={n:table(PRIOR/"risk/both"/(n+".tsv" if n.endswith("96") else "hold-"+n+".tsv")) for n in meta["order"]}
    for (rep,i),m in rset.items():
        assert set(m)=={"native","cached1","top2"}
        n,j=names[i];ref=byfile[n][j]
        assert int(m["top2"]["second"])==int(ref["A_second"]) and all(v["id"]==ref["A_top"] for v in m.values())
        assert abs(float(m["top2"]["gap"])-float(ref["A_margin"]))<2e-5 and int(m["top2"]["flag"])==(float(ref["A_margin"])<=.25)
        for k,v in m.items():totals[k].append(int(v["ns"])/1000)
        totals["top2_minus_cached1"].append((int(m["top2"]["ns"])-int(m["cached1"]["ns"]))/1000)
        totals["top2_minus_native"].append((int(m["top2"]["ns"])-int(m["native"]["ns"]))/1000)
    return {"calls_per_method":len(rset),"rows":len(names),"median_us":{k:median(v) for k,v in totals.items()},
            "mean_us":{k:statistics.mean(v) for k,v in totals.items()},
            "scope":"Real 129280-element CPU greedy sampler arrays, same order and data for all three scans, excludes recomputation."}

if __name__=="__main__":
    for name in sys.argv[1:] or ["local","router","time","quality","sampler"]:
        fn={"local":local_error,"router":router,"time":clocks,"quality":numerics,"sampler":sampler}[name]
        save(ROOT/(name.upper()+"_RESULT.json"),fn());print("verified",name,flush=True)
