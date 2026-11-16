from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;O=R/'iq-rhs';original=(R/'mx-prepack/backend/ggml-cpu.c').read_text();s=original.replace('#include "mx-rhs.h"','#include "mx-rhs.h"\n#include "iq-rhs.h"\n#ifndef XEON_IQ_PREPACK\n#define XEON_IQ_PREPACK 1\n#endif',1)
old='''    const size_t mx_stride,
    const uint64_t mx_safe_mask) {''';new='''    const size_t mx_stride,
    const uint64_t mx_safe_mask,
    const void * iq_rhs,
    const size_t iq_stride,
    const uint64_t iq_safe_mask) {''';assert s.count(old)==1;s=s.replace(old,new)
old='''                if (prepared) {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        xeon_mxfp4_rhs_dot_v1'''
new='''                const void * iq_prepared = compact_data && type==GGML_TYPE_IQ3_XXS && iq_rhs && ((iq_safe_mask >> i11)&1)
                    ? (const char *)iq_rhs + i11*iq_stride : NULL;
                if (iq_prepared) {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        xeon_iq3_rhs_dot_v1(ne00, &tmp[ir0-iir0], weight_data+ir0*weight_stride, src1_col, iq_prepared);
                    }
                } else if (prepared) {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        xeon_mxfp4_rhs_dot_v1''';assert s.count(old)==1;s=s.replace(old,new)
start=s.index('    const void * const * compact_experts = (!params->use_ref');end=s.index('\n\n',start)+2;lookup=s[start:end];s=s[:start]+s[end:]
start=s.index('static void ggml_compute_forward_mul_mat_id(\n');pos=s.index('    void * wdata_cur = params->wdata;',start);s=s[:pos]+lookup+s[pos:]
old='''    uint8_t * mx_flags = mx_shape ? incr_ptr_aligned(&wdata_cur, ne11*nth, sizeof(int64_t)) : NULL;
''';new=old+'''
    const bool iq_shape = xeon_iq3_rhs_shape(src0, src1, ids);
    const bool iq_enabled = XEON_IQ_PREPACK && iq_shape && !params->use_ref && compact_experts;
    const size_t iq_stride = iq_shape ? (size_t)ne10 : 0;
    void * iq_rhs = iq_shape ? incr_ptr_aligned(&wdata_cur, iq_stride*ne11, 64) : NULL;
    uint64_t * iq_worker_masks = iq_shape ? incr_ptr_aligned(&wdata_cur, nth*sizeof(uint64_t), sizeof(uint64_t)) : NULL;
''';assert s.count(old)==1;s=s.replace(old,new)
start=s.index('static void ggml_compute_forward_mul_mat_id(\n');pos=s.index('    if (ith == 0) {\n        // initialize matrix_row_counts',start)
s=s[:pos]+'''    if (iq_enabled) {
        GGML_ASSERT(vec_dot_type==GGML_TYPE_Q8_K);
        const int first=(ith*(ne10/256))/nth;
        const int end=((ith+1)*(ne10/256))/nth;
        const size_t qrow=ggml_row_size(vec_dot_type,ne10);
        uint64_t local_mask=0;
        for (int64_t input=0;input<ne11;++input) {
            const char * rhs = src1->type==vec_dot_type
                ? (const char *)src1->data + input*nb11
                : (const char *)params->wdata + input*qrow;
            if (xeon_iq3_rhs_prepare_v1(ne10,rhs,first,end,(char *)iq_rhs+input*iq_stride)) {
                local_mask |= UINT64_C(1)<<input;
            }
        }
        iq_worker_masks[ith]=local_mask;
    }

'''+s[pos:]
start=s.index('static void ggml_compute_forward_mul_mat_id(\n');pos=s.index('    for (int cur_a = 0; cur_a < n_as; ++cur_a)',start)
s=s[:pos]+'''    uint64_t iq_safe_mask=0;
    if (iq_enabled) {
        iq_safe_mask=UINT64_MAX;
        for (int worker=0;worker<nth;++worker) iq_safe_mask &= iq_worker_masks[worker];
    }

'''+s[pos:]
old='''                    compact_data, mx_enabled ? mx_rhs : NULL, mx_stride, mx_safe_mask
''';new='''                    compact_data, mx_enabled ? mx_rhs : NULL, mx_stride, mx_safe_mask,
                    iq_enabled ? iq_rhs : NULL, iq_stride, iq_safe_mask
''';assert s.count(old)==1;s=s.replace(old,new)
start=s.index('                case GGML_OP_MUL_MAT_ID:',s.index('struct ggml_cplan ggml_graph_plan'));needle='''                            cur += src1->ne[1]*n_tasks;
                        }''';pos=s.index(needle,start)+len(needle)
s=s[:pos]+'''
                        if (xeon_iq3_rhs_shape(src0,src1,ids)) {
                            cur = GGML_PAD(cur,64);
                            cur += (size_t)src1->ne[0]*src1->ne[1];
                            cur = GGML_PAD(cur,sizeof(uint64_t));
                            cur += n_tasks*sizeof(uint64_t);
                        }'''+s[pos:]
(O/'ggml-cpu.c').write_text(s);(O/'change.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/MX-baseline/ggml-cpu.c',tofile='b/IQ-RHS/ggml-cpu.c')))
(O/'SOURCE.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.suffix in ['.h','.cpp','.inc','.c']},indent=2)+'\n');print('Prepared isolated IQ RHS integration on verified MX baseline; notbuilt/deployed')
