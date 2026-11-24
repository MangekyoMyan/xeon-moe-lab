"""Publish the accepted operator manifest without editing the previous frozen study."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
OLD=LAB/"ecc9-simplification-20261119/current"
NEW=ROOT/"current"
CURRENT=LAB/"current"
TAG="normal24-adoption-20261124"


def read(p):
    return json.loads(p.read_bytes())


def create():
    ack=read(ROOT/"ACCEPTANCE.json");who=read(ROOT/"AFTER_IDENTITY.json");after=read(ROOT/"AFTER.json")
    assert ack["health_after_request"]=="ok" and ack["same_content_as_prior_pilot"] and who["pid"]==ack["pid"]==after["pid"]
    assert CURRENT.is_symlink() and CURRENT.resolve()==OLD and not NEW.exists()
    shutil.copytree(OLD,NEW,ignore=shutil.ignore_patterns("__pycache__","*.pyc","README.template"))
    profile=read(OLD/"runtime.json")
    topology=json.loads(subprocess.check_output([sys.executable,"-I",str(NEW/"check.py"),"topology"]))
    ids=topology["one_available_thread_per_core"]
    assert ids==list(range(24)) and topology["physical_cpu_mask"]==who["decode_mask"]==who["prefill_mask"]=="0xffffff"
    assert all(sum(x["socket"]==s for x in topology["cores"])==12 for s in (0,1))
    real=read(LAB/"postswap-health-20261124/evidence/hardware.json")
    assert real["boot"]==who["boot"]
    profile["profile"]="E5-2697 v2 x2 · 512 GiB · PG500-216 · DeepSeek Mix-256G"
    profile["verified_jst_date"]="2026-11-24"
    profile["hardware"].update(cpu=after["CPU_model"],sockets=2,physical=len(topology["cores"]),logical=topology["logical_cpus_online"],
        cpu_topology=topology["cores"],one_cpu_per_physical_core=ids,GPU_limit_W=after["gpu_watts_limit"],gpu_ecc=after["gpu_ecc"])
    profile["inference"].update(threads_decode=who["decode"],threads_prefill=who["prefill"],decode_cpu_mask=who["decode_mask"],
        prefill_cpu_mask=who["prefill_mask"],projector_initialization_threads_from_server_source=who["decode"])
    profile["service_config"]["sha256"]=after["config_sha256"]
    runtime_hashes={str(LAB/e["file"]):e["sha256"] for e in profile["libraries"].values()}
    assert runtime_hashes==after["mapped_runtime_hashes"] and profile["ld_library_path"]==after["ld_library_path"]
    profile["external_libraries_from_proc_maps"]=after["dependencies_mapped"]
    profile["planned_cpu_upgrade"].update(status="installed and measured; normal text service uses 24 decode / 24 prefill",
        next="Recalibrate CPU/GPU cost on this topology if changing expert assignment.")
    profile["rollback"]["prior_profile_backup"]={"file":"adopt24-20261124/private/normal-12-24.json",
        "sha256":read(ROOT/"BACKUP.json")["sha256"],"mode_and_owner_receipt":"adopt24-20261124/BACKUP.json"}
    (NEW/"runtime.json").write_text(json.dumps(profile,ensure_ascii=False,indent=2)+"\n")
    doc=(OLD/"README.md").read_text()
    doc=doc.replace("2026年11月19日に実機で照合したDeepSeekの構成です。","2026年11月24日に通常APIで受入を確認したDeepSeekの構成です。",1)
    doc=doc.replace("E5-2630L v2×2、512 GiB","E5-2697 v2×2、512 GiB",1)
    assert "入力24スレッド、生成12スレッド" in doc
    doc=doc.replace("入力24スレッド、生成12スレッド","入力24・生成24物理コア（各ソケット12）",1)
    doc=doc.replace("[CPU交換後の測り方](CPU_UPGRADE.md)","[CPU交換前後の実測](CPU_UPGRADE.md)",1)
    title="| 通常設定 | 値 |"
    assert title in doc
    doc=doc.replace(title,"通常の待受で、同じ短文への生成速度は %.2f tok/s、最初の本文は%.2f秒でした。[24/24の採用記録](https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/normal24-adoption-20261124)\n\n"%
                    (ack["server_timings"]["predicted_per_second"],ack["ttft_content_ms"]/1000)+title,1)
    (NEW/"README.md").write_text(doc)
    (NEW/"CPU_UPGRADE.md").write_text("""# E5-2697 v2 ×2への移行結果

2026年11月24日にE5-2697 v2を2基、24物理コア・48論理CPUとして確認しました。両ソケットのメモリは合計16×32 GiB、設定速度1600 MT/sです。通常テキストAPIは左右12コアずつで入力・生成を行います。

| 固定入力の生成速度 | 日本語 | Python |
|---|---:|---:|
| 交換前の左右6＋6コア | 7.64 tok/s | 7.61 tok/s |
| 交換後の左右6＋6コア | 8.59 tok/s | 8.50 tok/s |
| 交換後の左右12＋12コア | 10.06 tok/s | 9.86 tok/s |

この表は入力を12コアにそろえた固定トークン計時です。通常APIの候補は生成24・入力24で別に試し、今回の採用時には通常待受で短いテキスト1要求を確認しました。[固定入力の比較と一時API試験](https://github.com/MangekyoMyan/xeon-moe-lab/blob/901c0d14f8b9b73065a8b59ce2516185934f84d7/archive/review-handoffs-not-for-readers/2026-11-24-postload-measurement/SUMMARY_JA.txt) · [通常APIでの受入と温度](https://github.com/MangekyoMyan/xeon-moe-lab/releases/tag/normal24-adoption-20261124)

この起動で物理コアの代表はCPU 0～23、マスクは両方とも0xffffffです。左右6コアずつの比較条件では0～5および12～17を使いました。次にCPU構成を変えるときは、<code>python3 current/check.py topology</code>で番号を取得できます。
""")
    return profile


def activate():
    assert CURRENT.is_symlink() and CURRENT.resolve()==OLD
    saved=read(ROOT/"CURRENT_VERIFY_PRELINK.json")
    assert saved["checks_passed"] and saved["verified_api_pid"]==read(ROOT/"AFTER.json")["pid"]
    old_target=os.readlink(CURRENT);tmp=LAB/(".current-adopt24-"+str(os.getpid()))
    os.symlink(str(NEW.relative_to(LAB)),tmp)
    try:
        assert os.readlink(CURRENT)==old_target
        os.replace(tmp,CURRENT)
    finally:
        tmp.unlink(missing_ok=True)
    record={"old_target":old_target,"new_target":os.readlink(CURRENT),
        "current_runtime_sha256":hashlib.sha256((CURRENT/"runtime.json").read_bytes()).hexdigest()}
    (ROOT/"CURRENT_ENTRY.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    if sys.argv[1:]==["prepare"]:create()
    elif sys.argv[1:]==["activate"]:activate()
    else:raise SystemExit("specify prepare or activate")
