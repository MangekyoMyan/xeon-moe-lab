"""Stage the observer source patch without modifying the frozen engine."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;E=R.parent/'deepseek41-bringup-20261101'
original=(E/'source/src/llama-context.cpp').read_text();text=original
def replace(a,b):
    global text
    if text.count(a)!=1:raise RuntimeError('Patch anchor changed: '+a[:100])
    text=text.replace(a,b)
replace('#include "llama-context.h"','#include "llama-context.h"\n#include "shadow_capture.h"\n#include <optional>')
replace('    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);',
        '    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);\n    xeon_shadow::initialize();')
begin=text.index('bool llama_context::sched_copy_experts(');end=text.index('\nllm_graph_cb llama_context::graph_get_cb()',begin)
part=text[begin:end]
def local(a,b):
    global part
    if part.count(a)!=1:raise RuntimeError('Local patch anchor changed: '+a[:100])
    part=part.replace(a,b)
local('    auto * lctx = static_cast<llama_context *>(user_data);',
      '    auto * lctx = static_cast<llama_context *>(user_data);\n    const bool shadow_enabled = xeon_shadow::enabled();\n    const uint64_t shadow_start = shadow_enabled ? xeon_shadow::ns() : 0;\n    std::optional<xeon_shadow::capture> shadow;\n    if (shadow_enabled) { shadow.emplace(); }')
local('    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {\n        return true;',
      '    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {\n        if (shadow_enabled) { xeon_shadow::skipped(xeon_shadow::native_cache); }\n        return true;')
for condition,reason in [('ggml_graph_n_nodes(graph) == 0','empty_graph'),('node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst','other_node'),('ggml_nelements(ids) == 0','empty_ids')]:
    local('    if ('+condition+') {','    if ('+condition+') {\n        if (shadow_enabled) { xeon_shadow::skipped(xeon_shadow::'+reason+'); }')
local('    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {',
      '    const bool shadow_refreshed = ids != st.ids || (int64_t) st.used.size() != n_expert;\n    uint64_t shadow_fetch_ns = 0;\n    if (shadow_refreshed) {\n        const uint64_t shadow_fetch_start = shadow_enabled ? xeon_shadow::ns() : 0;')
local('        ggml_backend_synchronize(backend);','        ggml_backend_synchronize(backend);\n        if (shadow_enabled) { shadow_fetch_ns = xeon_shadow::ns() - shadow_fetch_start; }')
local('        const uint64_t shadow_fetch_start = shadow_enabled ? xeon_shadow::ns() : 0;',
      '        const uint64_t shadow_fetch_start = shadow_enabled ? xeon_shadow::ns() : 0;\n        if (shadow_enabled) { xeon_shadow::invalidate_route(); }')
local('    // group consecutive experts and copy them together',
      '    if (shadow_enabled && !lctx->moe_cache && ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {\n        shadow->begin(&st, ids, st.ids_data.data(), ggml_nbytes(ids), src, shadow_refreshed, shadow_fetch_ns, shadow_start);\n    } else if (shadow_enabled) {\n        xeon_shadow::skipped(lctx->moe_cache ? xeon_shadow::native_cache : xeon_shadow::unsupported_route);\n    }\n\n    // group consecutive experts and copy them together')
local('                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);',
      '                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);\n                if (shadow) { shadow->submitted((e - next)*expert_size); }')
local('            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);',
      '            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);\n            if (shadow) { shadow->submitted(size); }')
assert part.endswith('    return true;\n}\n')
part=part[:-len('    return true;\n}\n')]+'    if (shadow) { shadow->finish(); }\n    return true;\n}\n'
text=text[:begin]+part+text[end:]
(R/'llama-context-shadow.cpp').write_text(text)
(R/'capture.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/src/llama-context.cpp',tofile='b/src/llama-context.cpp')))
(R/'CAPTURE_SOURCE.json').write_text(json.dumps({'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(text.encode()).hexdigest(),
    'class_layout_changed':False,'additional_tensor_transfers':0,'additional_backend_synchronizations':0,'production_modified':False},indent=2)+'\n')
print('Staged observer only; assignments, original engine and class layout unchanged')
