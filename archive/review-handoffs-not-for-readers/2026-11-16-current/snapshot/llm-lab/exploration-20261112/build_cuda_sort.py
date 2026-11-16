from pathlib import Path
import argparse,difflib,hashlib,json,shlex,subprocess,time
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'cuda-sort';E=ROOT.parent/'deepseek41-bringup-20261101';SOURCE=E/'source/ggml/src/ggml-cuda/ggml-cuda.cu';BUILD=E/'build/ggml/src/ggml-cuda'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 original=SOURCE.read_text();source=original.replace('#include <vector>','#include <vector>\n#include "sort-ids.inc"',1)
 start=source.index('    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices',source.index('std::vector<char> ids_host'))
 end=source.index('    ids_to_sorted_host.insert',start)
 source=source[:start]+'''    xeon_sort_moe_ids(ids_host.data(), ids->nb[0], ids->nb[1], ne02, ne12,
        n_expert_used, ne11, ids_to_sorted_host, ids_from_sorted_host, tokens_per_expert);

'''+source[end:]
 target=OUT/'ggml-cuda.cu';target.write_text(source)
 (OUT/'sort.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='a/ggml-cuda.cu',tofile='b/ggml-cuda.cu')))
 if a.prepare_only:print('Prepared isolated CUDA sorting source; no build/load');return
 selected=[x for x in json.loads((E/'build/compile_commands.json').read_text()) if x['file']==str(SOURCE)];assert len(selected)==1
 command=shlex.split(selected[0]['command']);command[command.index('-c')+1]=str(target);command[command.index('-o')+1]=str(OUT/'ggml-cuda.cu.o');command.insert(1,'-I'+str(SOURCE.parent));command.insert(1,'-I'+str(OUT))
 started=time.monotonic();result=subprocess.run(command,cwd=BUILD,capture_output=True,text=True,timeout=300);(OUT/'compile.log').write_text(result.stdout+result.stderr);result.check_returncode()
 link=[]
 for item in shlex.split((BUILD/'CMakeFiles/ggml-cuda.dir/link.txt').read_text()):link.extend(shlex.split((BUILD/item[1:]).read_text()) if item.startswith('@') else [item])
 needle='CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o';assert link.count(needle)==1
 objects={str((BUILD/item).resolve()):digest(BUILD/item) for item in link if item.endswith('.o')}
 record={'compile':command,'cwd':str(BUILD),'original_source_sha256':digest(SOURCE),'candidate_source_sha256':digest(target),'helper_sha256':digest(OUT/'sort-ids.inc'),'original_link_inputs':objects,'variants':{}}
 for label in ['control','candidate']:
  runtime=OUT/('runtime-'+label);runtime.mkdir(exist_ok=True);cmd=link.copy()
  if label=='candidate':cmd[cmd.index(needle)]=str(OUT/'ggml-cuda.cu.o')
  final=runtime/'libggml-cuda.so.0.28.0';temp=runtime/'libggml-cuda.so.0.28.0.new'
  cmd[cmd.index('-o')+1]=str(temp);cmd=[('-Wl,--dependency-file='+str(OUT/(label+'-link.d'))) if x.startswith('-Wl,--dependency-file=') else x for x in cmd]
  result=subprocess.run(cmd,cwd=BUILD,capture_output=True,text=True,timeout=120);(OUT/(label+'-link.log')).write_text(result.stdout+result.stderr);result.check_returncode();temp.replace(final)
  for name in ['libggml-cuda.so','libggml-cuda.so.0']:
   if not (runtime/name).exists():(runtime/name).symlink_to(final.name)
  record['variants'][label]={'link':cmd,'library':str(final),'sha256':digest(final)}
 record['elapsed_s']=time.monotonic()-started;(OUT/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'compiled':True,'elapsed_s':record['elapsed_s'],'variants':list(record['variants'])}))
if __name__=='__main__':main()
