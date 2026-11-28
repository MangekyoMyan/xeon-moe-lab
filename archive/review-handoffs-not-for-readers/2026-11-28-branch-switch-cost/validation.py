"""Candidate recovery, recorded tool-result edits, and actual input/logit comparison."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import shutil
import sqlite3
import time
from study_lib import (R,OLD,BUDGET,Native,Error,Store,add_asset,asset,blob,digest,encode,engine,follow,image_tokens,
                       install_blob,pack,private,read,rows,rpc,uid,unpack,window,write,append)
from measure import action,server

VS=R/"val-store"
EV=R/"evidence"
LABELS=("tool-parent","tool-A","tool-B")
QUESTION="この枝に記録されている保管箱の登録番号は何ですか。半角の数字だけを返してください。"


def point(store, label):
    con=sqlite3.connect("file:"+str(store/"branches.sqlite3")+"?mode=ro",uri=True)
    con.row_factory=sqlite3.Row
    try:
        v=con.execute("SELECT s.* FROM snapshots s JOIN marks m ON m.sid=s.id WHERE m.name=?",(label,)).fetchone()
        if v is None:raise RuntimeError(label)
        return {"id":v["id"],"parent":v["parent"],"tokens":unpack(v["tokens"]),"pending":unpack(v["pending"]),
                "messages":json.loads(v["messages"]),"sha":v["sha"],"bytes":v["bytes"]}
    finally:
        con.close()


def phase(run,label):
    tr=run/"trace"
    p=tr/"next-phase"
    p.write_text(label+"\n");p.replace(tr/"phase")


def exact_inputs():
    all_cases={k:read(R/"cases"/(k+".json")) for k in ("short","long")}
    points={k:c["points"]["T"] for k,c in all_cases.items()}
    prompts=[{"name":k,"anchor":k,"messages":follow(z["messages"],all_cases[k]["next_request"]),
              "ids":all_cases[k]["next_ids"],"expected":all_cases[k]["answer"],"max":8} for k,z in points.items()]
    anchors=read(OLD/"evidence/anchors.json")
    for name,v in anchors.items():
        src=OLD/"evidence/references"/(name+".slot")
        z=v["snapshot"]
        points[name]={"id":z["id"],"parent":None,"tokens":v["ids"],"pending":[],
                      "messages":v["messages_after"],"sha":z["image_sha256"],"bytes":z["image_bytes"]}
        add_asset(src,points[name])
    for t in read(OLD/"evidence/numeric-inputs.json"):
        prompts.append({"name":t["name"],"anchor":t["anchor"],"messages":read(OLD/"evidence"/("question-"+t["name"]+".json")),
                        "ids":t["ids"],"expected":"619","max":t["max_tokens"]})
    return points,prompts


def seed_validation(native):
    points,prompts=exact_inputs()
    store=Store(VS,native,BUDGET)
    with store.tx():
        for key,z in points.items():
            install_blob(native,z)
            store.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?,?)",
                    (z["id"],None,native.compat,pack(z["tokens"]),pack(z["pending"]),encode(z["messages"]),z["sha"],z["bytes"],time.time()))
            store.db.execute("INSERT INTO blobs VALUES(?,?,1,?)",(z["sha"],z["bytes"],time.time()))
            store.db.execute("INSERT INTO marks VALUES(?,?,1)",("anchor-"+key,z["id"]))
    store.db.close()
    private(VS/"branches.sqlite3",False)
    write(EV/"validation-cases.json",{"points":points,"prompts":prompts})


def invoke(run, label, op, **kw):
    result=action(VS,op,**kw)
    append(run/"logical.jsonl",{"label":label,"op":op,"result":result})
    return result


def expect_error(run,label,code,op,**kw):
    try: action(VS,op,**kw)
    except Error as e:
        assert e.code==code,(label,e.code,code,str(e))
        result={"code":e.code,"message":str(e),"details":e.details}
        append(run/"logical.jsonl",{"label":label,"error":result})
        return result
    raise AssertionError("unexpected success "+label)


def tool_test(run,binding):
    native=Native(binding);seed_validation(native)
    with server("new",False,binding,run,"tool",VS,R/"fault_worker.py"):
        history=[{"role":"user","content":"これは架空の在庫台帳です。以後は会話に記録された照会結果を使って答えてください。今は OK だけ返してください。"}]
        ans=invoke(run,"parent-turn","turn",messages=history,max_tokens=8)
        invoke(run,"parent-mark","mark",name=LABELS[0],pin=True)
        p=point(VS,LABELS[0])
        assert p["messages"]==history+[{"role":"assistant","content":ans["content"]}]
        call_id="fixture_lookup_07"
        msgs=[*p["messages"],{"role":"user","content":"箱K7の保管台帳を照会し、ツール結果の registered_number の値だけを半角の数字で回答してください。"},
              {"role":"assistant","content":"","tool_calls":[{"id":call_id,"type":"function","function":{"name":"lookup_box","arguments":'{"box":"K7"}'}}]},
              {"role":"tool","name":"lookup_box","tool_call_id":call_id,"content":'{"box":"K7","registered_number":731}'}]
        other=copy.deepcopy(msgs);other[-1]["content"]=other[-1]["content"].replace("731","284")
        idsA=native.render(msgs);idsB=native.render(other);n=len(p["tokens"])
        assert idsA[:n]==p["tokens"]==idsB[:n] and idsA!=idsB,(n,idsA[:12],p["tokens"][:12])
        invoke(run,"fork-A","fork",name="A",**{"from":LABELS[0]})
        invoke(run,"enter-A","checkout",name="A")
        phase(run,"tool-original")
        try:a=invoke(run,"tool-A-turn","turn",messages=msgs,max_tokens=8)
        finally:phase(run,"idle")
        invoke(run,"hold-A","mark",name=LABELS[1],pin=True)
        pa=point(VS,LABELS[1])
        invoke(run,"fork-B-from-A","fork",name="B",**{"from":LABELS[1]})
        invoke(run,"start-B-at-A","checkout",name="B")
        phase(run,"tool-edited")
        try:b=invoke(run,"tool-B-edited-turn","turn",messages=other,max_tokens=8)
        finally:phase(run,"idle")
        assert b["restored_prefix"]==p["id"] and b["timings"]["cache_n"]==n and b["timings"]["prompt_n"]==len(idsB)-n
        invoke(run,"hold-B","mark",name=LABELS[2],pin=True)
        pb=point(VS,LABELS[2])
        for label,z in zip(LABELS,(p,pa,pb)):
            assert digest(native.images/blob(z))==z["sha"] and image_tokens(native.images/blob(z),131072)==pack(z["tokens"])
            add_asset(native.images/blob(z),z)
        real=rows(run/"trace/calls.jsonl")
        for label,expected in (("tool-original",idsA),("tool-edited",idsB)):
            got=[(pos,t) for c in real if c["phase"]==label for pos,t in zip(c["pos"],c["input_ids"]) if pos<len(expected)]
            assert got==list(zip(range(n,len(expected)),expected[n:]))
        write(EV/"tool-edit.json",{"parent":p,"A":pa,"B":pb,"A_request":msgs,"B_request":other,"A_ids":idsA,"B_ids":idsB,
                                   "A_reply":a,"B_reply":b,"external_tool_executions":0})
        invoke(run,"before-fault-A","checkout",name="A")
        # A dirty outgoing state is first captured for rollback. Both fault paths use a real
        # documented native error for a nonexistent synthetic image.
        invoke(run,"make-A-dirty","turn",text=QUESTION,max_tokens=8)
        source=rpc(VS,{"op":"ticket"})["ticket"]
        plan=run/"test-fault.json"
        def last_use(sha):
            con=sqlite3.connect("file:"+str(VS/"branches.sqlite3")+"?mode=ro",uri=True)
            try:return con.execute("SELECT used FROM blobs WHERE sha=?",(sha,)).fetchone()[0]
            finally:con.close()
        before=last_use(pb["sha"])
        write(plan,{"mode":"abort-final-tx","count":1})
        v=expect_error(run,"cancel-after-active-and-last-use-writes","operation_failed","checkout",name="B")
        st=rpc(VS,{"op":"status"})
        assert v["details"]["rollback"]["exact_head"] and st["mode"]=="ready" and st["ticket"]["branch"]==source["branch"]
        after=last_use(pb["sha"]);assert before==after
        write(EV/"cancelled-placement.json",{"target":pb["id"],"target_used_before":before,"target_used_after":after,
                                              "retained_source":st["ticket"]["branch"],"confirmed_native_rollback":v["details"]["rollback"]})
        for count in (1,2):
            write(plan,{"mode":"absent-native-image","count":count})
            v=expect_error(run,"native-reject-"+str(count),"native_error" if count==1 else "rollback_failed","checkout",name="B")
            st=rpc(VS,{"op":"status"})
            assert st["ticket"]["branch"]==source["branch"] and st["ticket"]["generation"]==source["generation"]
            if count==1:assert st["mode"]=="ready" and v["details"]["rollback"]["exact_head"]
            else:
                assert st["mode"]=="invalid"
                expect_error(run,"invalid-fence","invalid_slot","turn",text="unused",max_tokens=1)
                assert invoke(run,"recover-A","recover")["exact_head"]
        pre=rpc(VS,{"op":"ticket"})["ticket"]
        write(plan,{"mode":"crash-after-confirmed-restore","count":1})
        expect_error(run,"crash-in-checkout","manager_disconnected","checkout",name="B")
    # This manager deliberately stays fenced; use the original readiness helper which permits
    # inspecting an uncertain slot instead of requiring it to be ready for inference.
    from x9ram_pilot import manager
    with manager(VS,binding,run/"fence.log"):
        st=rpc(VS,{"op":"status"})
        assert st["mode"]=="uncertain" and st["ticket"]==pre
        expect_error(run,"uncertain-fence","native_outcome_unknown","recover")
    write(EV/"fence.json",{"ticket":pre,"mode":st["mode"]})


def after_restart(run,binding):
    native=Native(binding);old=read(EV/"fence.json")["ticket"];tool=read(EV/"tool-edit.json")
    val=read(EV/"validation-cases.json");answers={}
    with server("new",False,binding,run,"restart",VS):
        state=rpc(VS,{"op":"status"});assert state["mode"]=="ready"
        assert all(state["ticket"][k]==old[k] for k in ("session","branch","generation"))
        for name,num in (("A","731"),("B","284")):
            invoke(run,"enter-"+name,"checkout",name=name)
            ans=invoke(run,"after-restart-"+name,"turn",text=QUESTION,max_tokens=8)
            answers[name]={"text":ans["content"],"correct":ans["content"].strip()==num,"cache":ans["timings"]["cache_n"],"new":ans["timings"]["prompt_n"]}
        for name,key in zip(LABELS,("parent","A","B")):
            z=point(VS,name);assert z==tool[key] and digest(native.images/blob(z))==z["sha"]
        for t in val["prompts"]:
            lab=t["name"];name="fixed-"+lab
            invoke(run,"fork-"+lab,"fork",name=name,**{"from":"anchor-"+t["anchor"]})
            loaded=invoke(run,"install-"+lab,"checkout",name=name)
            assert loaded["exact_head"] and loaded["replayed"]==0
            phase(run,name)
            try:out=invoke(run,"read-"+lab,"turn",messages=t["messages"],max_tokens=t["max"])
            finally:phase(run,"idle")
            assert out["timings"]["cache_n"]==len(val["points"][t["anchor"]]["tokens"])
    write(EV/"restart-answers.json",answers)


def stock(run,binding):
    val=read(EV/"validation-cases.json");native=Native(binding);results=[]
    for z in val["points"].values():install_blob(native,z)
    for t in val["prompts"]:
        z=val["points"][t["anchor"]]
        native.restore(z["sha"],pack(z["tokens"]),z["bytes"])
        assert native.render(t["messages"])==t["ids"]
        phase(run,"fixed-"+t["name"])
        try:
            out=native.call("POST","/completion",{"id_slot":0,"prompt":t["ids"],"n_predict":t["max"],"temperature":0,"seed":19,
                       "cache_prompt":True,"return_tokens":True,"response_fields":["tokens","content","timings","tokens_cached"]})
        finally:phase(run,"idle")
        assert out["timings"]["cache_n"]==len(z["tokens"])
        results.append({"name":t["name"],**out})
    write(EV/"stock-answers.json",results)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stages",nargs="+",choices=["tool","restart","stock","terminal"])
    p.add_argument("--attempt",type=int,default=1);a=p.parse_args()
    with window(R,"BRANCH-COST-VALIDATION"):
        for stage in a.stages:
            run=R/"runs"/(stage+"-"+str(a.attempt))
            assert not run.exists()
            from terminal_cost import measure as terminal
            from study_lib import STORE
            store={"tool":VS,"restart":VS,"stock":R/"stock-store","terminal":STORE}[stage]
            with engine(R,store,run,observed=stage!="terminal",stock=stage=="stock") as binding:
                {"tool":tool_test,"restart":after_restart,"stock":stock,"terminal":terminal}[stage](run,binding)
            write(run/"DONE.json",{"stage":stage,"complete":True})
            index=R/"VALIDATION_RUNS.json";old=read(index) if index.exists() else {};old[stage]=run.name;write(index,old)
