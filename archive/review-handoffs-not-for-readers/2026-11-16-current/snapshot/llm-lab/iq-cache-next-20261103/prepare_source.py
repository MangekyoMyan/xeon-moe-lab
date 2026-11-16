from pathlib import Path
import json
R=Path(__file__).resolve().parent;L=R.parent;P=L/'iq3-cache-20261103'
if (R/'BUILD.json').exists():raise SystemExit('Initial generation already completed; edit preserved candidate sources instead.')
# New study only; never mutate the active 64 GiB overlay.
for name in ['iq3-cache.h','iq3-cache.cpp','compact-kernel.inc','model-cache.h','model-cache-body.inc','build.py','test-cache.cpp','build_tests.py','build_sanitized.py','test-loader.cpp','make_fixture.cpp','build_loader_tests.py']:
 (R/name).write_bytes((P/name).read_bytes())
s=(L/'postcompact-20261103/iq-small-kernels.inc').read_text();s=s[:s.index('template<bool two> __attribute__((noinline)) static float iq_direct_dot')]
(R/'iq-small-kernels.inc').write_text(s)
p=R/'iq3-cache.h';s=p.read_text();s+='\n#ifdef __cplusplus\nextern "C" {\n#endif\nvoid xeon_iq2s_compact_dot(int, float *, size_t, const void *, size_t, const void *, size_t, int);\nvoid xeon_iq3s_compact_dot(int, float *, size_t, const void *, size_t, const void *, size_t, int);\n#ifdef __cplusplus\n}\n#endif\n';p.write_text(s)
p=R/'iq3-cache.cpp';s=p.read_text().replace('#include <vector>','#include <vector>\n#include <type_traits>')
s=s.replace('#include "compact-kernel.inc"','#include "compact-kernel.inc"\n#include "iq-small-kernels.inc"\nstatic_assert(sizeof(compact_block)==sizeof(iq_compact_block), "shared compact stride");')
s=s.replace('constexpr size_t cap = size_t(64) << 30;','constexpr size_t cap = size_t(96) << 30;')
s=s.replace('    int64_t ne[4];','    enum ggml_type type;\n    int64_t ne[4];')
s=s.replace('t->type == GGML_TYPE_IQ3_XXS','(t->type == GGML_TYPE_IQ3_XXS || t->type == GGML_TYPE_IQ2_S || t->type == GGML_TYPE_IQ3_S)')
s=s.replace('ggml_nbytes(t)/sizeof(block_iq3_xxs)','ggml_nbytes(t)/ggml_type_size(t->type)')
s=s.replace('        e->source = t->data;', '        e->source = t->data;\n        e->type = t->type;')
a='''        const auto * src = static_cast<const block_iq3_xxs *>(t->data);
        auto * dst = static_cast<compact_block *>(e->data);
        #pragma omp parallel for schedule(static) num_threads(o.threads)
        for (size_t b = 0; b < blocks; ++b) {
            expanded_block temp;
            expand(src+b, &temp, 1);
            compact(temp, dst[b]);
        }'''
b='''        #pragma omp parallel for schedule(static) num_threads(o.threads)
        for (size_t b = 0; b < blocks; ++b) {
            if (t->type == GGML_TYPE_IQ3_XXS) {
                expanded_block temp;
                expand(static_cast<const block_iq3_xxs *>(t->data)+b, &temp, 1);
                compact(temp, static_cast<compact_block *>(e->data)[b]);
            } else if (t->type == GGML_TYPE_IQ2_S) {
                iq_pack<true>(static_cast<const block_iq2_s *>(t->data)[b], static_cast<iq_compact_block *>(e->data)[b]);
            } else {
                iq_pack<false>(static_cast<const block_iq3_s *>(t->data)[b], static_cast<iq_compact_block *>(e->data)[b]);
            }
        }'''
assert a in s;s=s.replace(a,b).replace('if (std::memcmp(e.ne, t->ne, sizeof(e.ne))', 'if (e.type != t->type || std::memcmp(e.ne, t->ne, sizeof(e.ne))')
s+='''
void xeon_iq2s_compact_dot(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1 && n%256 == 0);
    *s = iq_compact_dot<true>(n, static_cast<const iq_compact_block *>(x), static_cast<const block_q8_K *>(y));
    (void) bs; (void) bx; (void) by;
}
void xeon_iq3s_compact_dot(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1 && n%256 == 0);
    *s = iq_compact_dot<false>(n, static_cast<const iq_compact_block *>(x), static_cast<const block_q8_K *>(y));
    (void) bs; (void) bx; (void) by;
}
''';p.write_text(s)
p=R/'build.py';s=p.read_text().replace('compact_data ? xeon_iq3_compact_dot : type_traits_cpu[type].vec_dot', 'compact_data ? (type == GGML_TYPE_IQ2_S ? xeon_iq2s_compact_dot : type == GGML_TYPE_IQ3_S ? xeon_iq3s_compact_dot : xeon_iq3_compact_dot) : type_traits_cpu[type].vec_dot')
s=s.replace('type == GGML_TYPE_IQ3_XXS && ids->ne[1] == 1', '(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S) && ids->ne[1] == 1')
s=s.replace("'model-cache-body.inc']", "'model-cache-body.inc','iq-small-kernels.inc']")
p.write_text(s)
p=R/'model-cache.h';s=p.read_text().replace('#include <stdexcept>','#include <stdexcept>\n#include <algorithm>');p.write_text(s)
p=R/'model-cache-body.inc';s=p.read_text().replace('65536','98304').replace('IQ3 cache:', 'IQ cache:')
a=s.index('        // Stable layer/projection order;');b=s.index('        LLAMA_LOG_WARN("IQ cache: complete',a)
s=s[:a]+'''        std::vector<const ggml_tensor *> candidates;
        for (const auto & layer : layers) {
            for (const auto * tensor : {layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps}) {
                if (!tensor || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || ml.lazy.has(tensor)) continue;
                if (tensor->type == GGML_TYPE_IQ3_XXS || tensor->type == GGML_TYPE_IQ2_S || tensor->type == GGML_TYPE_IQ3_S) candidates.push_back(tensor);
            }
        }
        // Type priority measured in the CPU single-input probes. Stable layer order breaks ties.
        const auto rank = [](const ggml_tensor * t) {
            if (t->type == GGML_TYPE_IQ3_S) return 0;
            if (t->type == GGML_TYPE_IQ2_S) return 1;
            return 2;
        };
        std::stable_sort(candidates.begin(), candidates.end(), [&](const ggml_tensor * a, const ggml_tensor * b) { return rank(a) < rank(b); });
        for (const auto * tensor : candidates) {
            if (low_memory) { ++skipped; continue; }
            LLAMA_LOG_INFO("IQ cache: checking %s (%s)\\n", tensor->name, ggml_type_name(tensor->type));
            const int status = pimpl->iq3_compact.add(pimpl->iq3_compact.owner, tensor);
            if (status == 1) {
                ++count;
                LLAMA_LOG_WARN("IQ cache: ready %s (%s); total %.3f GiB\\n", tensor->name, ggml_type_name(tensor->type), pimpl->iq3_compact.used(pimpl->iq3_compact.owner)/1073741824.0);
            } else {
                ++skipped;
                if (status < 0) {
                    low_memory = true;
                    LLAMA_LOG_WARN("IQ cache: allocation/reserve limit; remaining tensors use original CPU path\\n");
                }
            }
        }
'''+s[b:]
p.write_text(s)
# Generalize graph validation without changing the current engine.
p=R/'test-cache.cpp';s=p.read_text().replace('static size_t cases=0,outputs=0;', 'static size_t cases=0,outputs=0;\nstatic ggml_type weight_type=GGML_TYPE_IQ3_XXS;')
s=s.replace('ggml_new_tensor_3d(ctx,GGML_TYPE_IQ3_XXS,k,m,experts)', 'ggml_new_tensor_3d(ctx,weight_type,k,m,experts)')
a='''        auto p=reinterpret_cast<block_iq3_xxs *>(weights.data());
        for(size_t b=0;b<weights.size()/sizeof(*p);++b){p[b].d=0x1800;for(auto &q:p[b].qs)q=rng();}'''
b='''        const size_t stride=ggml_type_size(weight_type);
        for(auto &q:weights)q=rng();
        for(size_t b=0;b<weights.size();b+=stride){uint16_t d=0x1800;std::memcpy(weights.data()+b,&d,2);}'''
assert a in s;s=s.replace(a,b).replace('weights.size()/sizeof(block_iq3_xxs)*144', 'weights.size()/ggml_type_size(weight_type)*144')
s=s.replace('(size_t(64)<<30)+1','(size_t(96)<<30)+1')
s=s.replace('            for(int k:{256,512,2304,5120})', '            for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_S}) { weight_type=type;\n            for(int k:{256,512,2304,5120})')
s=s.replace('                    run(cpu,nth,k,m,tokens,broadcast,strided,fp32);', '                    run(cpu,nth,k,m,tokens,broadcast,strided,fp32);\n            }')
s=s.replace('need(ti>=0 && gguf_get_tensor_type(g,ti)==GGML_TYPE_IQ3_XXS,"IQ3 tensor");auto ne=gguf_get_tensor_ne(g,ti);', 'need(ti>=0,"tensor");weight_type=gguf_get_tensor_type(g,ti);need(weight_type==GGML_TYPE_IQ3_XXS||weight_type==GGML_TYPE_IQ2_S||weight_type==GGML_TYPE_IQ3_S,"IQ type");auto ne=gguf_get_tensor_ne(g,ti);')
s=s.replace('ggml_row_size(GGML_TYPE_IQ3_XXS,k)*m*48','ggml_row_size(weight_type,k)*m*48')
p.write_text(s)
# Mixed-type small fixture, with the same 144-byte compact size per block.
p=R/'make_fixture.cpp';s=p.read_text().replace('auto t=ggml_new_tensor_3d(ctx,e>1?GGML_TYPE_IQ3_XXS:GGML_TYPE_F32,k,m,e);', 'auto type=e>1?(std::strstr(name,"gate_exps")?GGML_TYPE_IQ2_S:std::strstr(name,"up_exps")?GGML_TYPE_IQ3_S:GGML_TYPE_IQ3_XXS):GGML_TYPE_F32;\n        auto t=ggml_new_tensor_3d(ctx,type,k,m,e);')
a='''            auto p=static_cast<block_iq3_xxs *>(t->data);
            for(size_t b=0;b<ggml_nbytes(t)/sizeof(*p);++b){p[b].d=0x1000;for(auto &q:p[b].qs)q=rng();}'''
b='''            auto p=static_cast<uint8_t *>(t->data);size_t bytes=ggml_nbytes(t),stride=ggml_type_size(type);
            for(size_t b=0;b<bytes;++b)p[b]=rng();
            for(size_t b=0;b<bytes;b+=stride){uint16_t d=0x1000;std::memcpy(p+b,&d,2);}'''
assert a in s;s=s.replace(a,b);p.write_text(s)
# Stamp where this new candidate came from.
(R/'STATE.json').write_text(json.dumps({'status':'building','production_changed':False,'active_profile':'compact64','candidate':'IQ2_S + IQ3_S + IQ3_XXS compact, 96 GiB hard cap, 64 GiB reserve; initial type-first selection pending graph results'},indent=2)+'\n')
print('Prepared independent multi-format cache candidate.')
