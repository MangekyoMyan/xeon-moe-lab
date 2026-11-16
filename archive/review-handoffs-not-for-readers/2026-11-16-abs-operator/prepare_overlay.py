"""Create an isolated overlay of the frozen final CPU source; do not deploy."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent
BASE=R.parents[1]/'exploration-20261112/mx-prepack/release-candidate-a/sources/ggml-cpu.c'
s=BASE.read_text();original=s
if hashlib.sha256(s.encode()).hexdigest()!='e8fdf3d424463f9852fa774d9de6057dce35855a872936bda87d12d3d756e3a8':raise RuntimeError('Baseline changed')
def replace(old,new):
    global s
    if s.count(old)!=1:raise RuntimeError('Anchor not unique: '+old[:80])
    s=s.replace(old,new)
replace('#include "mx-rhs.h"','#include "mx-rhs.h"\n#include "iq-abs.h"')
replace('    const uint64_t mx_safe_mask) {','    const uint64_t mx_safe_mask,\n    struct xeon_iq_abs_state * abs_state) {')
replace('''                if (prepared) {
                    for (int64_t ir0''','''                const void * abs_prepared = NULL;
                if (compact_data && abs_state->mode) {
                    if (abs_state->mode == 1) {
                        if ((abs_state->safe_mask >> i11) & 1) abs_prepared = abs_state->data + i11*abs_state->stride;
                    } else {
                        if (abs_state->last_input != i11) {
                            abs_state->safe = xeon_iq_abs_prepare(ne00, src1_col, 0, ne00/256, abs_state->data);
                            abs_state->last_input = i11;
                            if (abs_state->diagnostic) { abs_state->prepared_blocks += ne00/256; abs_state->unsafe_ranges += !abs_state->safe; }
                        }
                        if (abs_state->safe) abs_prepared = abs_state->data;
                    }
                }
                if (abs_prepared) {
                    void (*fn)(int,float *,const void *,const void *,const void *) = type == GGML_TYPE_IQ2_S ? xeon_iq2_abs_dot : xeon_iq3s_abs_dot;
                    const int64_t end = MIN(iir0+blck_0, ir0_end);
                    if (abs_state->diagnostic) abs_state->rows += end-iir0;
                    for (int64_t ir0=iir0; ir0<end; ++ir0) fn(ne00,&tmp[ir0-iir0],weight_data+ir0*weight_stride,src1_col,abs_prepared);
                } else if (prepared) {
                    for (int64_t ir0''')
anchor='    uint8_t * mx_flags = mx_shape ? incr_ptr_aligned(&wdata_cur, ne11*nth, sizeof(int64_t)) : NULL;'
replace(anchor,anchor+'''

    const bool abs_shape = xeon_iq_abs_shape(src0, src1, ids);
    struct xeon_iq_abs_state abs_state = {0};
    abs_state.last_input = -1;
    abs_state.mode = abs_shape && !params->use_ref ? xeon_iq_abs_get_mode() : 0;
    abs_state.diagnostic = abs_state.mode ? xeon_iq_abs_diagnostics() : 0;
    abs_state.stride = (size_t) ne10;
    char * abs_data = abs_shape ? incr_ptr_aligned(&wdata_cur, xeon_iq_abs_bytes(ne10, ne11, nth), 64) : NULL;
    uint64_t * abs_masks = abs_shape ? (uint64_t *)(abs_data + (size_t)ne10*MAX(ne11,nth)) : NULL;
    abs_state.data = abs_state.mode == 2 ? abs_data + ith*abs_state.stride : abs_data;
''')
anchor='    if (ith == 0) {\n        // initialize matrix_row_counts'
replace(anchor,'''    if (abs_state.mode == 1) {
        const int first = (ith*(ne10/256))/nth;
        const int end = ((ith+1)*(ne10/256))/nth;
        const size_t qrow = ggml_row_size(vec_dot_type, ne10);
        uint64_t mask = 0;
        for (int64_t input=0; input<ne11; ++input) {
            const char * rhs = src1->type == vec_dot_type ? (const char *)src1->data+input*nb11 : (const char *)params->wdata+input*qrow;
            const int safe = xeon_iq_abs_prepare(ne10, rhs, first, end, abs_data+input*abs_state.stride);
            if (safe) mask |= UINT64_C(1)<<input;
            if (abs_state.diagnostic) { abs_state.prepared_blocks += end-first; abs_state.unsafe_ranges += !safe; }
        }
        abs_masks[ith] = mask;
    }

'''+anchor)
anchor='    uint64_t mx_safe_mask = 0;'
replace(anchor,'''    if (abs_state.mode == 1) {
        abs_state.safe_mask = UINT64_MAX;
        for (int worker=0; worker<nth; ++worker) abs_state.safe_mask &= abs_masks[worker];
    }

'''+anchor)
replace('                    compact_data, mx_enabled ? mx_rhs : NULL, mx_stride, mx_safe_mask\n','                    compact_data, mx_enabled ? mx_rhs : NULL, mx_stride, mx_safe_mask, &abs_state\n')
anchor='''            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

static void ggml_compute_forward'''
replace(anchor,anchor.replace('\n}\n\n/////////////////////////////////', '\n    if (abs_state.diagnostic) xeon_iq_abs_report(abs_state.prepared_blocks, abs_state.rows, abs_state.unsafe_ranges);\n}\n\n/////////////////////////////////'))
anchor='''                            cur += src1->ne[1]*n_tasks;
                        }
                    } break;
                case GGML_OP_OUT_PROD:'''
replace(anchor,'''                            cur += src1->ne[1]*n_tasks;
                        }
                        if (xeon_iq_abs_shape(src0, src1, ids)) {
                            const size_t extra = xeon_iq_abs_bytes(src1->ne[0], src1->ne[1], n_tasks);
                            GGML_ASSERT(cur <= SIZE_MAX-extra-63);
                            cur += extra+63;
                        }
                    } break;
                case GGML_OP_OUT_PROD:''')
(R/'ggml-cpu.c').write_text(s)
(R/'change.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='final/ggml-cpu.c',tofile='experimental/ggml-cpu.c')))
(R/'SOURCE.json').write_text(json.dumps({'baseline':str(BASE),'baseline_sha256':hashlib.sha256(original.encode()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'ggml-cpu.c',R/'iq-abs.h',R/'iq-abs.cpp']},'scope':'Isolated mode0/current,mode1/shared-all-inputs,mode2/private-lazy; production untouched'},indent=2)+'\n')
print('Prepared isolated overlay')
