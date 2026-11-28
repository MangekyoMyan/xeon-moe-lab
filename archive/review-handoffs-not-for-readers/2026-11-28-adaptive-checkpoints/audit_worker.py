"""Uniform lab observer. It writes nothing and returns only per-operation counters and timings."""
import hashlib
import os
from pathlib import Path
import stat
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent/"checkout/current/experiments/state-branches"))
import branchctl, native, storage, manager

audit=None
def count(label):
    if audit is not None:audit["counts"][label]=audit["counts"].get(label,0)+1
def wrap_func(module, name, label):
    original=getattr(module,name)
    def run(*a,**kw):
        count(label(*a,**kw) if callable(label) else label)
        return original(*a,**kw)
    setattr(module,name,run)
wrap_func(os,"fsync",lambda fd:"directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync")
wrap_func(storage,"digest","image_sha");wrap_func(native,"digest","image_sha")
class Db:
    def __init__(self,raw):self.raw=raw
    def __getattr__(self,n):return getattr(self.raw,n)
    def execute(self,sql,*args):
        typ=sql.lstrip().split(None,1)[0].upper()
        if typ=="COMMIT":count("sql_commit")
        elif typ in ("INSERT","DELETE","UPDATE") and not self.raw.in_transaction:count("sql_autocommit")
        return self.raw.execute(sql,*args)
construct=manager.Manager.__init__
def init(self,store,*a,**kw):
    construct(self,store,*a,**kw);store.db=Db(store.db)
manager.Manager.__init__=init
render=native.Native.render
def render_hash(self,msg):
    ids=render(self,msg)
    if audit is not None:audit["input_sha256"]=hashlib.sha256(native.pack(ids)).hexdigest()
    return ids
native.Native.render=render_hash
generate=native.Native.generate
def generate_receipt(self,*a,**kw):
    result=generate(self,*a,**kw)
    if audit is not None:audit["sampled_ids"]=list(result["sampled"])
    return result
native.Native.generate=generate_receipt
observe=native.observe_prefill
def pp(n,t):
    observe(n,t)
    if audit is not None:audit["prefills"].append({k:t.get(k) for k in ("prompt_n","prompt_ms","cache_n","predicted_n")})
native.observe_prefill=pp
save=storage.Store.save_active
def capture(self,*a,**kw):
    if audit is None:return save(self,*a,**kw)
    rec=kw.setdefault("receipt",{});begin=time.monotonic_ns()
    try:
        v=save(self,*a,**kw);rec["image_sha256"]=v["sha"];return v
    finally:
        audit["saves"].append({**rec,"optional":kw.get("on_publish") is not None,"common_ms":(time.monotonic_ns()-begin)/1e6})
storage.Store.save_active=capture
handle=manager.Manager.handle
def measured(self,req,emit=lambda _:None):
    global audit
    if req["op"]=="measure-inventory":
        s=self.store;ret=s.listing();ret["checkpoint_policy"]=self.policy.status()
        return {k:ret[k] for k in ("image_bytes","resident_images","budget_bytes","mode","ticket","checkpoint_policy")}|{
          "observation_windows":{"prefill":{str(k):list(v) for k,v in self.policy.prefills.items()},
             "saves":list(self.policy.saves),"restores":{k[0]+"/"+k[1]:list(v) for k,v in self.policy.restores.items() if k[0] is not None},
             "restore_route":self.policy.restore_route},
          "snapshots":s.db.execute("SELECT count(*) FROM snapshots").fetchone()[0],
          "metadata_files":{e:(s.root/("branches.sqlite3"+e)).stat().st_size if (s.root/("branches.sqlite3"+e)).exists() else 0 for e in ("","-wal","-shm")}}
    if not req.get("_measured"):return handle(self,req,emit)
    audit={"counts":{},"saves":[],"prefills":[]}
    try:return {**handle(self,req,emit),"_measurements":audit}
    finally:audit=None
manager.Manager.handle=measured
if __name__=="__main__":branchctl.cli()
