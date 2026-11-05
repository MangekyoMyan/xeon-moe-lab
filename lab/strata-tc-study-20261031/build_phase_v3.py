#!/usr/bin/env python3
import difflib,hashlib,json,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;old=R/'variants/phase-v2';v=R/'variants/phase-v3';s=v/'source';rt=v/'runtime'
s.mkdir(parents=True,exist_ok=False);rt.mkdir()
for p in (old/'source').iterdir():
 if p.is_file():shutil.copy2(p,s/p.name);(s/p.name).chmod(0o644)
for p in (old/'runtime').iterdir():
 if p.name!='libxv-experiment.so':(rt/p.name).symlink_to(p.resolve())
p=s/'expert-cache.cpp';before=p.read_text();text=before
def replace(a,b,n=1):
 global text
 assert text.count(a)==n,(a,text.count(a));text=text.replace(a,b)
replace('struct xv_graph {','struct xv_graph {\n    int allocated_capacity = 0;\n    bool streamed_prefill = false;')
replace('xv_env_one("GGML_XV_TC_PREFILL") && phase_index == 1','xv_env_one("GGML_XV_TC_PREFILL") && phase_index == 1 &&\n                            (!xv_env_one("GGML_XV_STREAM_PREFILL_ONLY") || graph->streamed_prefill)')
replace('xv_env_int("GGML_XV_STREAM_MIN_TOKENS",512,2,2048)','xv_env_int("GGML_XV_STREAM_MIN_TOKENS",1024,2,2048)')
replace('auto item = std::make_unique<xv_graph>();','auto item = std::make_unique<xv_graph>();\n    item->allocated_capacity = capacity;')
replace('hits <= 4096','hits <= 8192')
replace('if (!item) item = create_graph(pool, bank, 4096, token_axis);',
        'if (!item || item->allocated_capacity < hits)\n            item = create_graph(pool, bank, std::max(1024, GGML_PAD(hits,256)), token_axis);')
replace('selections > 4096','selections > 8192',2)
replace('prefill ? 4096 : hits','prefill ? 8192 : hits')
replace('auto graph=create_graph(pool,*bank,hits);','auto graph=create_graph(pool,*bank,hits);\n            graph->streamed_prefill = prefill;')
p.write_text(text);(v/'phase-v3.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='phase-v2',tofile='phase-v3')))
cmd=json.loads((old/'BUILD.json').read_text())['command'];cmd=[x.replace(str(old),str(v)) for x in cmd];subprocess.run(cmd,check=True)
(v/'BUILD.json').write_text(json.dumps({'parent':str(old),'command':cmd,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [s/'expert-cache.cpp',rt/'libxv-experiment.so',rt/'libxv-q8-exact.so']}},indent=2)+'\n')
