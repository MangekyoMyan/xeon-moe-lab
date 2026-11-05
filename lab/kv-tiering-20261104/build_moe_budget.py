from pathlib import Path
import subprocess,json
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';s=E/'source';rt=L/'iq-cache-next-20261103/runtime'
cmd=['g++','-O2','-std=c++17','-I'+str(s/'include'),'-I'+str(s/'src'),'-I'+str(s/'ggml/include'),'-I'+str(s/'ggml/src'),str(R/'moe_budget_probe.cpp'),'-L'+str(rt),'-L'+str(E/'build/bin'),'-Wl,-rpath,'+str(rt),'-Wl,-rpath,'+str(L/'gpu-stream-20261103/runtime'),'-Wl,-rpath,'+str(E/'build/bin'),'-lllama','-lggml-cuda','-lggml-cpu','-lggml-base','-lggml','-o',str(R/'moe_budget_probe')]
subprocess.run(cmd,check=True,timeout=60);(R/'MOE_BUDGET_BUILD.json').write_text(json.dumps(cmd,indent=2)+'\n')
