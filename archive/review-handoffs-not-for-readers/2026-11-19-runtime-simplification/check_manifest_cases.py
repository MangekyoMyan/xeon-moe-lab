"""Check what the fast and full identity checks can distinguish, using tiny fixtures."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("current_check",(ROOT/"current/check.py"))
checker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

def asset(root,name,payload):
    path=root/name
    path.write_bytes(payload)
    return {"file":name,"sha256":checker.digest(path)}

def reject(call):
    try:call()
    except ValueError:return
    raise RuntimeError("Expected an identity mismatch")

def main():
    with TemporaryDirectory() as d:
        base=Path(d)
        shard=asset(base,"model.gguf",b"same size fixture A")
        projector=asset(base,"projector.gguf",b"projector")
        plan=asset(base,"selected.plan",b"plan for model A")
        model_receipt=asset(base,"model-download.json",json.dumps(shard).encode())
        vision_receipt=asset(base,"vision-download.json",json.dumps(projector).encode())
        plan["published_copy"]="exported.plan"
        (base/plan["published_copy"]).write_bytes((base/plan["file"]).read_bytes())
        plan["model_download_fingerprint"]=model_receipt["sha256"]
        config={"environment":{"GGML_CPU_IQ_PLAN":str(base/plan["file"])},
                "model":str(base/shard["file"]),"vision_profile":{"projector":str(base/projector["file"])}}
        profile={"service_config":asset(base,"config.json",json.dumps(config).encode()),"server_executable":asset(base,"server",b"fixture"),"libraries":{},"plan":plan,
                 "model":{"first_shard":shard["file"],"download_receipt":model_receipt,"files":[dict(shard,size=19)]},
                 "vision":{"download_receipt":vision_receipt,"files":[dict(projector,size=9)]}}
        def verify(deep=False):return checker.check(profile,base,False,deep,asset_root=base)
        assert verify()["checks_passed"] and verify(True)["checks_passed"]
        plan["model_download_fingerprint"]="0"*64
        reject(verify)
        plan["model_download_fingerprint"]=model_receipt["sha256"]
        (base/plan["published_copy"]).write_bytes(b"foreign plan")
        reject(verify)
        (base/plan["published_copy"]).write_bytes((base/plan["file"]).read_bytes())
        (base/shard["file"]).write_bytes(b"same size fixture B")
        assert verify()["weight_contents_rehashed_in_this_call"] is False
        reject(lambda:verify(True))
        (base/model_receipt["file"]).write_text("{}")
        reject(verify)
        (base/model_receipt["file"]).write_bytes(json.dumps(shard).encode())
        (base/plan["file"]).write_bytes(b"different plan!!")
        reject(verify)
    result={"passed":True,"full_mode_detects_same_length_changed_weights":True,"fast_mode_validates_receipt_and_sizes":True,"changed_model_receipt_rejected":True,"changed_plan_rejected":True,"swapped_profile_association_rejected":True,"changed_published_plan_rejected":True,"tested_with_models":False}
    (ROOT/"MANIFEST_FIXTURE_RESULT.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":main()
