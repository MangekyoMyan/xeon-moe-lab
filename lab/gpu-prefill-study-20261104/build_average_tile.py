from pathlib import Path
import json,shlex,subprocess,hashlib,difflib,time,sys
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';S=E/'source/ggml/src/ggml-cuda';B=E/'build/ggml/src/ggml-cuda'
orig=(S/'mmq.cu').read_text()
text=orig.replace('#include <cstdint>','#include <cstdint>\nextern "C" int xeon_grouped_current_mode() __attribute__((weak));')
old='if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {'
new='if ((cc == GGML_CUDA_CC_VOLTA && xeon_grouped_current_mode && xeon_grouped_current_mode() >= 2) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {'
assert text.count(old)==1;text=text.replace(old,new)
(R/'mmq-average.cu').write_text(text)
(R/'mmq-average.patch').write_text(''.join(difflib.unified_diff(orig.splitlines(True),text.splitlines(True),fromfile='a/mmq.cu',tofile='b/mmq.cu')))
flags={}
for line in (B/'CMakeFiles/ggml-cuda.dir/flags.make').read_text().splitlines():
    if line.startswith('CUDA_') and ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
cmd=[str(L/'v100-prep-20261024/cuda-12.9.2/bin/nvcc'),*flags['CUDA_DEFINES'],*flags['CUDA_INCLUDES'],*flags['CUDA_FLAGS'],'-I'+str(S),'-c',str(R/'mmq-average.cu'),'-o',str(R/'mmq-average.o')]
if '--link-only' not in sys.argv:
    started=time.monotonic();p=subprocess.run(cmd,cwd=B,capture_output=True,text=True,timeout=60)
    (R/'mmq-average-build.log').write_text(p.stdout+p.stderr);print('compile',p.returncode,'seconds',round(time.monotonic()-started,2),p.stderr[-1800:],flush=True)
    if p.returncode:raise SystemExit(p.returncode)
out=R/'runtime';out.mkdir(exist_ok=True)
rawlink=shlex.split((B/'CMakeFiles/ggml-cuda.dir/link.txt').read_text());link=[]
for arg in rawlink:link.extend(shlex.split((B/arg[1:]).read_text()) if arg.startswith('@') else [arg])
target='CMakeFiles/ggml-cuda.dir/mmq.cu.o';assert link.count(target)==1;link[link.index(target)]=str(R/'mmq-average.o')
link[link.index('-o')+1]=str(out/'libggml-cuda.so.0.28.0')
link=[('-Wl,--dependency-file='+str(R/'cuda-link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link]
p=subprocess.run(link,cwd=B,capture_output=True,text=True,timeout=60)
(R/'mmq-average-link.log').write_text(p.stdout+p.stderr);print('link',p.returncode,p.stderr[-1200:],flush=True)
if p.returncode:raise SystemExit(p.returncode)
for name in ['libggml-cuda.so','libggml-cuda.so.0']:
    if not (out/name).exists():(out/name).symlink_to('libggml-cuda.so.0.28.0')
(R/'CUDA_BUILD.json').write_text(json.dumps({'compile':cmd,'link':link,'cwd':str(B),'source_sha256':hashlib.sha256((R/'mmq-average.cu').read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((out/'libggml-cuda.so.0.28.0').read_bytes()).hexdigest(),'production_changed':False},indent=2)+'\n')
