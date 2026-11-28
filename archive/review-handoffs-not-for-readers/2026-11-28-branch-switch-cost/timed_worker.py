"""Laboratory-only checkout instrumentation. Timing records stay in memory until fetched."""
from functools import wraps
import os
import stat
import sys
import time

sys.path.insert(0,os.environ["BRANCH_LAB_SOURCE"])
import branchctl
import manager
import native
import storage


class Recorder:
    def __init__(self):
        self.events=None
        self.stack=[]
        self.completed={}

    def watch(self, label, fn, args, kwargs):
        if self.events is None:
            return fn(*args,**kwargs)
        parent=self.stack[-1] if self.stack else None
        idx=len(self.events)
        record={"label":label,"parent":parent,"begin":time.monotonic_ns()}
        self.events.append(record);self.stack.append(idx)
        try:
            value=fn(*args,**kwargs)
            if label in ("native_save_http","native_restore_http"):
                record["server_ms"]=dict(value["timings"])
            return value
        finally:
            record["end"]=time.monotonic_ns()
            assert self.stack.pop()==idx

    def wrap(self,where,name,label):
        fn=getattr(where,name)
        @wraps(fn)
        def wrapped(*args,**kwargs):
            if self.events is None:
                return fn(*args,**kwargs)
            tag=label if isinstance(label,str) else label(*args,**kwargs)
            return self.watch(tag,fn,args,kwargs)
        setattr(where,name,wrapped)


def instrument():
    trace=Recorder()
    for where,name,label in (
        (storage.Store,"save_active","publish_outgoing"),(storage.Store,"choose","choose_target"),
        (storage.Store,"valid","check_snapshot"),(storage.Store,"clean_files","clean_files"),
        (manager.Manager,"place","place_target"),(native.Native,"save","native_save_and_check"),
        (native.Native,"restore","native_restore_and_check"),(storage,"digest","existing_blob_sha"),
        (native,"digest","staged_blob_sha"),(storage,"image_tokens","existing_token_check"),
        (native,"image_tokens","staged_token_check"),(storage,"sync_dir","sync_directory"),
        (os,"replace","publish_rename"),(os,"unlink","unlink"),
        (os,"fsync",lambda fd:"directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync"),
        (native.Native,"call",lambda self,method,route,*a,**kw:"native_save_http" if route.endswith("=save") else
         "native_restore_http" if route.endswith("=restore") else "other_native_http"),
    ):
        trace.wrap(where,name,label)

    class DB:
        def __init__(self,db):
            self.raw=db

        def __getattr__(self,name):
            return getattr(self.raw,name)

        def execute(self,sql,*args):
            if trace.events is None:
                return self.raw.execute(sql,*args)
            stmt=" ".join(sql.upper().split())
            label=None
            if stmt=="COMMIT":
                label="sql_commit"
            elif stmt.startswith("BEGIN"):
                label="sql_begin"
            elif stmt.startswith(("UPDATE ","INSERT ","DELETE ")):
                if not self.raw.in_transaction:
                    if stmt.startswith("UPDATE CONTROL SET MODE="):
                        label="sql_autocommit_"+str(args[0][0])
                    elif stmt.startswith("UPDATE BLOBS SET USED="):
                        label="sql_autocommit_used"
                    else:
                        label="sql_autocommit_other"
                elif stmt.startswith("UPDATE BLOBS SET USED="):
                    label="sql_used_inside_tx"
            return trace.watch(label,self.raw.execute,(sql,*args),{}) if label else self.raw.execute(sql,*args)

    init=manager.Manager.__init__
    def init_manager(self,store):
        init(self,store)
        store.db=DB(store.db)
    manager.Manager.__init__=init_manager
    handle=manager.Manager.handle
    def observe(self,req,emit=lambda _:None):
        if req.get("op")=="read-checkout-trace":
            return trace.completed.pop(req["label"])
        label=req.get("_measure")
        if req.get("op")!="checkout" or not label:
            return handle(self,req,emit)
        trace.events=[]
        try:
            return trace.watch("manager_request",handle,(self,req,emit),{})
        finally:
            trace.completed[label]={"events":trace.events,"journal_mode":self.store.db.execute("PRAGMA journal_mode").fetchone()[0],
                                     "synchronous":self.store.db.execute("PRAGMA synchronous").fetchone()[0]}
            trace.events=None
    manager.Manager.handle=observe


if __name__=="__main__":
    if os.environ["BRANCH_LAB_TIMED"]=="1":
        instrument()
    branchctl.cli()
