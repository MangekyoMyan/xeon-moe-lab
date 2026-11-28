from study_lib import R, STORE, Native, engine, make_template, prepare_case, window, write

if __name__=="__main__":
    run=R/"runs/prepare"
    with window(R,"BRANCH-COST-FIXTURES"),engine(R,STORE,run) as binding:
        native=Native(binding)
        for size in ("short","long"):
            case=prepare_case(native,size)
            for duplicate in (False,True):
                make_template(native,case,duplicate)
            print(size,[(k,len(v["tokens"]),v["bytes"]) for k,v in case["points"].items()],flush=True)
    write(R/"PREPARED.json",{"sizes":["short","long"],"complete":True})
