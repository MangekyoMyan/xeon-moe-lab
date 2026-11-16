from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;OUT=R/'mx-prepack/backend'
math=(R/'mx-prepack/kernel.inc').read_text();math=math[math.index('static inline float mx_hsum'):]
math=math.replace('const mx_prepared & rhs','const mx_rhs_view & rhs').replace('rhs.broadcast.data()','rhs.broadcast').replace('__attribute__((noinline)) static float mx_dot','static inline float mx_dot')
(OUT/'mx-math.inc').write_text(math)
original=(R/'sparse-cache/ggml-cpu.c').read_text();s=original.replace('#include "iq3-cache.h"','#include "iq3-cache.h"\n#include "mx-rhs.h"\n#ifndef XEON_MX_PREPACK\n#define XEON_MX_PREPACK 1\n#endif',1)
old='''    const void * wdata,
    const void * compact_data) {''';new='''    const void * wdata,
    const void * compact_data,
    const void * mx_rhs,
    const size_t mx_stride,
    const uint64_t mx_safe_mask) {''';assert s.count(old)==1;s=s.replace(old,new)
old='''                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, weight_data + ir0*weight_stride, 0, src1_col, 0, 1);
                }'''
new='''                const void * prepared = mx_rhs && ((mx_safe_mask >> i11) & 1)
                    ? (const char *)mx_rhs + i11*mx_stride : NULL;
                if (prepared) {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        xeon_mxfp4_rhs_dot_v1(ne00, &tmp[ir0-iir0], weight_data+ir0*weight_stride, src1_col, prepared);
                    }
                } else {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        vec_dot(ne00, &tmp[ir0 - iir0], 0, weight_data + ir0*weight_stride, 0, src1_col, 0, 1);
                    }
                }''';assert s.count(old)==1;s=s.replace(old,new)
old='''    char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);
'''
new=old+'''
    const bool mx_shape = xeon_mxfp4_rhs_shape(src0, src1, ids);
    const bool mx_enabled = XEON_MX_PREPACK && mx_shape && !params->use_ref;
    const size_t mx_stride = mx_shape ? xeon_mxfp4_rhs_stride(ne10) : 0;
    void * mx_rhs = mx_shape ? incr_ptr_aligned(&wdata_cur, mx_stride*ne11, 64) : NULL;
    uint8_t * mx_flags = mx_shape ? incr_ptr_aligned(&wdata_cur, ne11*nth, sizeof(int64_t)) : NULL;
''';assert s.count(old)==1;s=s.replace(old,new)
start=s.index('static void ggml_compute_forward_mul_mat_id(\n');pos=s.index('    if (ith == 0) {\n        // initialize matrix_row_counts',start)
s=s[:pos]+'''    if (mx_enabled) {
        // The same worker has just quantized these blocks. Prequantized inputs
        // use their original strides. No worker reads another worker's blocks.
        const int first = (ith*(ne10/32))/nth;
        const int end = ((ith+1)*(ne10/32))/nth;
        const size_t qrow = ggml_row_size(vec_dot_type, ne10);
        for (int64_t input=0;input<ne11;++input) {
            const char * rhs = src1->type==vec_dot_type
                ? (const char *)src1->data + input*nb11
                : (const char *)params->wdata + input*qrow;
            mx_flags[input*nth+ith] = xeon_mxfp4_rhs_prepare_v1(ne10, rhs, first, end,
                (char *)mx_rhs + input*mx_stride);
        }
    }

'''+s[pos:]
old='''    const void * const * compact_experts = (!params->use_ref'''
new='''    uint64_t mx_safe_mask = 0;
    if (mx_enabled) {
        for (int64_t input=0;input<ne11;++input) {
            bool safe=true;
            for (int worker=0;worker<nth;++worker) safe = safe && mx_flags[input*nth+worker]!=0;
            if (safe) mx_safe_mask |= UINT64_C(1)<<input;
        }
    }

    const void * const * compact_experts = (!params->use_ref''';assert s.count(old)==1;s=s.replace(old,new)
old='''                    compact_data
                );''';new='''                    compact_data, mx_enabled ? mx_rhs : NULL, mx_stride, mx_safe_mask
                );''';assert s.count(old)==1;s=s.replace(old,new)
start=s.index('                case GGML_OP_MUL_MAT_ID:',s.index('struct ggml_cplan ggml_graph_plan'))
needle='''                        cur += ggml_tiled_wdata_size(n_tasks, node);''';pos=s.index(needle,start)+len(needle)
s=s[:pos]+'''
                        if (xeon_mxfp4_rhs_shape(src0, src1, ids)) {
                            cur = GGML_PAD(cur, 64);
                            cur += xeon_mxfp4_rhs_stride(src1->ne[0])*src1->ne[1];
                            cur = GGML_PAD(cur, sizeof(int64_t));
                            cur += src1->ne[1]*n_tasks;
                        }'''+s[pos:]
(OUT/'ggml-cpu.c').write_text(s);(OUT/'change.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/ggml-cpu.c',tofile='b/ggml-cpu.c')))
(OUT/'SOURCE.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.h','.cpp','.inc','.c']},indent=2)+'\n')
print('Prepared MXFP4 backend source; no build/profile change')
