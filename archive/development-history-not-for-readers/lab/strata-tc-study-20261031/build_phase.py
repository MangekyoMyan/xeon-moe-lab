#!/usr/bin/env python3
import difflib,hashlib,json,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;LAB=R.parent
parent=R/'variants/stream-tc';v=R/'variants/phase-tc';s=v/'source';rt=v/'runtime';s.mkdir(parents=True,exist_ok=False);rt.mkdir()
for p in parent.joinpath('source').iterdir():
    if p.is_file():shutil.copy2(p,s/p.name);(s/p.name).chmod(0o644)
for p in parent.joinpath('runtime').iterdir():
    if p.name!='libxv-experiment.so':(rt/p.name).symlink_to(p.resolve())
p=s/'expert-cache.cpp';before=p.read_text();text=before
def replace(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
replace('bool stream_active=xv_env_one("GGML_XV_STREAM_ACTIVE");','bool stream_active=xv_env_one("GGML_XV_STREAM_ACTIVE");\n    bool stream_prefill_only=xv_env_one("GGML_XV_STREAM_PREFILL_ONLY");')
replace('exact_q8_enabled() && !adaptive && !validate','exact_q8_enabled() && (!adaptive || stream_prefill_only) && !validate')
replace('conf.stream_active && exact_q8_enabled() && !conf.validate && !conf.adaptive','conf.stream_active && exact_q8_enabled() && !conf.validate && (!conf.adaptive || conf.stream_prefill_only)')
replace('if (conf.stream_active) { stream_begin(pool,dst,row_counts); return; }','if (conf.stream_active && (!conf.stream_prefill_only || dst->src[2]->ne[1]>1)) { stream_begin(pool,dst,row_counts); return; }')
replace('if (conf.stream_active) return 0;','if (conf.stream_active && (!conf.stream_prefill_only || gate->src[2]->ne[1]>1)) return 0;')
# Preserve the accepted token-based history and phase-specific update policy.
replace('int adaptive_interval=xv_env_int("GGML_XV_ADAPTIVE_INTERVAL",32,1,4096);','int adaptive_interval=xv_env_int("GGML_XV_ADAPTIVE_INTERVAL",32,1,4096);\n    int prefill_interval=xv_env_int("GGML_XV_ADAPTIVE_PREFILL_INTERVAL",adaptive_interval,1,4096);\n    bool last_prefill=false;')
replace('int adaptive_per_layer=xv_env_int("GGML_XV_ADAPTIVE_PER_LAYER",1,0,16);','int adaptive_per_layer=xv_env_int("GGML_XV_ADAPTIVE_PER_LAYER",1,0,16);\n    int prefill_per_layer=xv_env_int("GGML_XV_ADAPTIVE_PREFILL_PER_LAYER",adaptive_per_layer,0,16);')
accepted=(LAB/'prefill-study-20261030/release/prefill-v1/source/expert-cache.cpp').read_text()
a=accepted.index('static void adaptive_observe(');b=accepted.index('\nstatic ',a+1)
c=text.index('static void adaptive_observe(');d=text.index('\nstatic ',c+1);text=text[:c]+accepted[a:b]+text[d:]
replace('if(record && record%conf.adaptive_interval)return 0;\n    const int limit=record?conf.adaptive_per_layer:conf.adaptive_initial;',
        'const int interval=conf.last_prefill?conf.prefill_interval:conf.adaptive_interval;\n    if(record && record%interval)return 0;\n    const int limit=record?(conf.last_prefill?conf.prefill_per_layer:conf.adaptive_per_layer):conf.adaptive_initial;')
p.write_text(text);(v/'phase.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='stream-tc',tofile='phase-tc')))
cmd=['g++','-std=c++17','-O2','-march=ivybridge','-fPIC','-shared','-Wall','-Wextra','-Werror','-pthread','-DXV_EXACT_Q8','-I'+str(LAB/'xeon-volta-engine/source/ggml/include'),'-I'+str(s),str(p),'-L'+str(rt),'-lggml-cuda','-lggml-base','-lxv-q8-exact','-lggml-cpu','-Wl,-rpath,$ORIGIN','-o',str(rt/'libxv-experiment.so')]
subprocess.run(cmd,check=True)
(v/'BUILD.json').write_text(json.dumps({'parent':str(parent),'command':cmd,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [s/'expert-cache.cpp',rt/'libxv-experiment.so',rt/'libxv-q8-exact.so']}},indent=2)+'\n')
