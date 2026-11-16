"""Isolated CPU coordination experiments, preserving arithmetic and old artifacts."""
from pathlib import Path
import argparse,difflib,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent;BASE=ROOT/'sparse-cache';GROUP=ROOT/'cpu-coordination'
def main():
 p=argparse.ArgumentParser();p.add_argument('variant',choices=['control','relaxed','active-only','final-barrier','worker-mask','combined','main-last']);p.add_argument('--prepare-only',action='store_true');base=p.add_mutually_exclusive_group();base.add_argument('--base-mx',action='store_true');base.add_argument('--base-iq',action='store_true');p.add_argument('--sanitize',action='store_true');a=p.parse_args()
 has_mx=a.base_mx or a.base_iq
 suffix='-san' if a.sanitize else ''
 extra=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-g','-O1'] if a.sanitize else []
 group=ROOT/'cpu-coordination-iq' if a.base_iq else ROOT/'cpu-coordination-mx' if a.base_mx else GROUP
 out=group/(a.variant+suffix);out.mkdir(parents=True,exist_ok=True);source_base=ROOT/'iq-rhs/ggml-cpu.c' if a.base_iq else ROOT/'mx-prepack/backend/ggml-cpu.c' if a.base_mx else BASE/'ggml-cpu.c';original=source_base.read_text();s=original
 if a.variant=='main-last':
  before='''    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }'''
  after='''    // Optional master-last affinity; arithmetic and worker indices are unchanged.
    const char * main_last_option = getenv("GGML_CPU_MAIN_LAST");
    const bool main_last = main_last_option && strcmp(main_last_option, "1") == 0;
    for (int j = 0; j < tpp->n_threads; j++) {
        const int worker = main_last ? (j + 1) % tpp->n_threads : j;
        ggml_thread_cpumask_next(tpp->cpumask, workers[worker].cpumask, tpp->strict_cpu, &cpumask_iter);
    }'''
  assert s.count(before)==1;s=s.replace(before,after)
 old='''    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }
'''
 assert s.count(old)==1
 if a.variant=='relaxed':s=s.replace(old,old.replace('*current_chunk_ctr = nth;','atomic_store_explicit(current_chunk_ctr, nth, memory_order_relaxed);'))
 if a.variant in ['active-only','combined']:
  before='''                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};'''
  after='''                if (matrix_row_counts[i02] == 0) {
                    atomic_int * counter = (atomic_int *)(atomic_current_chunk + i02);
                    atomic_store_explicit(counter, nth, memory_order_relaxed);
                }
'''+before
  assert s.count(before)==1;s=s.replace(before,after);s=s.replace(old,'    // Active counters are initialized by ith0 and published by this barrier.\n')
 if a.variant in ['final-barrier','combined']:
  before='''    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT'''
  after='''#ifndef GGML_USE_OPENMP
    ggml_barrier(state->threadpool);
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT'''
  assert s.count(before)==1;s=s.replace(before,after)
 if a.variant=='worker-mask' and not has_mx:raise ValueError('worker-mask requires --base-mx or --base-iq')
 if has_mx and a.variant in ['worker-mask','combined']:
  before='uint8_t * mx_flags = mx_shape ? incr_ptr_aligned(&wdata_cur, ne11*nth, sizeof(int64_t)) : NULL;'
  after='uint64_t * mx_worker_masks = mx_shape ? incr_ptr_aligned(&wdata_cur, nth*sizeof(uint64_t), sizeof(uint64_t)) : NULL;'
  assert s.count(before)==1;s=s.replace(before,after)
  before='        const size_t qrow = ggml_row_size(vec_dot_type, ne10);\n        for (int64_t input=0;input<ne11;++input) {'
  after=before.replace('        for (int64_t','        uint64_t local_mask = 0;\n        for (int64_t')
  assert s.count(before)==1;s=s.replace(before,after)
  before='            mx_flags[input*nth+ith] = xeon_mxfp4_rhs_prepare_v1(ne10, rhs, first, end,\n                (char *)mx_rhs + input*mx_stride);\n        }\n    }'
  after='            if (xeon_mxfp4_rhs_prepare_v1(ne10, rhs, first, end, (char *)mx_rhs + input*mx_stride)) {\n                local_mask |= UINT64_C(1)<<input;\n            }\n        }\n        mx_worker_masks[ith] = local_mask;\n    }'
  assert s.count(before)==1;s=s.replace(before,after)
  before='        for (int64_t input=0;input<ne11;++input) {\n            bool safe=true;\n            for (int worker=0;worker<nth;++worker) safe = safe && mx_flags[input*nth+worker]!=0;\n            if (safe) mx_safe_mask |= UINT64_C(1)<<input;\n        }'
  after='        mx_safe_mask = UINT64_MAX;\n        for (int worker=0;worker<nth;++worker) mx_safe_mask &= mx_worker_masks[worker];'
  assert s.count(before)==1;s=s.replace(before,after)
  before='cur += src1->ne[1]*n_tasks;';after='cur += n_tasks*sizeof(uint64_t);'
  assert s.count(before)==1;s=s.replace(before,after)
 (out/'ggml-cpu.c').write_text(s);(out/'change.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/ggml-cpu.c',tofile='b/ggml-cpu.c')))
 if a.prepare_only:print('Prepared '+a.variant);return
 prior=json.loads((BASE/'CPU_BUILD.json').read_text());cmd=next(c for c in prior['commands'] if isinstance(c,list) and '-c' in c and c[c.index('-c')+1]==str(BASE/'ggml-cpu.c')).copy()
 if has_mx:cmd.insert(1,'-I'+str(ROOT/'mx-prepack/backend'))
 if a.base_iq:cmd.insert(1,'-I'+str(ROOT/'iq-rhs'))
 cmd[cmd.index('-c')+1]=str(out/'ggml-cpu.c');cmd[cmd.index('-o')+1]=str(out/'ggml-cpu.c.o');cmd+=extra;subprocess.run(cmd,check=True,timeout=180)
 linkinfo=json.loads((ROOT/'row-local/BUILD.json').read_text())['link'];link=[str(out/'ggml-cpu.c.o') if x==str(BASE/'ggml-cpu.c.o') else x for x in linkinfo['argv']]
 runtime=out/'runtime';runtime.mkdir(exist_ok=True);lib=runtime/'libggml-cpu.so.0.28.0';temp=lib.with_suffix('.new');link[link.index('-o')+1]=str(temp);link=[('-Wl,--dependency-file='+str(out/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link]
 if has_mx:link.append(str(ROOT/'mx-prepack/backend'/('mx-rhs'+suffix+'.o')))
 if a.base_iq:link.append(str(ROOT/'iq-rhs'/('iq-rhs'+suffix+'.o')))
 link+=extra
 subprocess.run(link,cwd=linkinfo['cwd'],check=True,timeout=120);temp.replace(lib)
 for name in ['libggml-cpu.so','libggml-cpu.so.0']:
  if not (runtime/name).exists():(runtime/name).symlink_to(lib.name)
 record={'variant':a.variant,'source_base':str(source_base),'base_sha256':hashlib.sha256(original.encode()).hexdigest(),'compile':cmd,'link':{'argv':link,'cwd':linkinfo['cwd']},'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'library':str(lib),'sha256':hashlib.sha256(lib.read_bytes()).hexdigest()}
 (out/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n');print('Built '+a.variant)
if __name__=='__main__':main()
