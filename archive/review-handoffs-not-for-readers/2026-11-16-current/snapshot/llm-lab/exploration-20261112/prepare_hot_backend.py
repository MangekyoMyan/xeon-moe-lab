"""Overlay two llama source files without modifying the baseline checkout/headers."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn';SRC=R.parent/'deepseek41-bringup-20261101/source/src'
def main():
    context=(SRC/'llama-context.cpp').read_text();graph=(SRC/'llama-graph.cpp').read_text()
    context='#include "hot-backend.h"\n'+context
    old='''        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions'''
    assert context.count(old)==1
    context=context.replace(old,'''        backends.emplace_back(backend_cpu);
        if (auto hot = xeon_hot_backend_create(model, backend_cpu,
                backends.size() > 1 ? backends.front().get() : nullptr,
                cparams.n_threads, cparams.n_threads_batch)) {
            backends.insert(backends.end() - 1, ggml_backend_ptr(hot));
        }

        // create a list of the set_n_threads functions''',1)
    old='''        sched_reserve();

        if (!cparams.flash_attn)'''
    assert context.count(old)==1
    context=context.replace(old,'''        sched_reserve();
        // This is constructor initialization only, never a later sched_reserve().
        xeon_hot_backend_finish(backend_cpu);

        if (!cparams.flash_attn)''',1)
    graph='#include "hot-backend.h"\n'+graph
    start=graph.index('ggml_tensor * llm_graph_context::build_moe_ffn(\n         ggml_tensor * cur,\n         ggml_tensor * gate_inp,\n         ggml_tensor * gate_inp_b,')
    split=graph.index('    ggml_tensor * up = nullptr;\n    ggml_tensor * experts = nullptr;',start)
    end=graph.index('    if (down_exps_s) {\n        cb(experts, "ffn_moe_down_scaled", il);',split)
    original=graph[split:end]
    body=original[len('    ggml_tensor * up = nullptr;\n    ggml_tensor * experts = nullptr;\n'):]
    replacement='''    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;
    const auto has_custom_precision = [&](const ggml_tensor * weight) {
        return prec_policy && (prec_policy->prec_acc.count(weight) || prec_policy->prec_src1.count(weight));
    };
    if (arch == LLM_ARCH_DEEPSEEK41 && il >= 0 && uint32_t(il) < hparams.n_layer_all && n_tokens == 1 && n_embd == 5120 && n_expert == 384 && n_expert_used == 6 &&
        type_op == LLM_FFN_SILU && gate_exps && up_exps && down_exps && !gate_up_exps &&
        !up_exps_b && !gate_exps_b && !down_exps_b && !gate_up_exps_b &&
        !up_exps_s && !gate_exps_s && !down_exps_s && !post_exp_norm && !slots && !moe_cache &&
        (!loras || loras->empty()) && !hdmd &&
        !has_custom_precision(gate_exps) && !has_custom_precision(up_exps) && !has_custom_precision(down_exps)) {
        experts = xeon_hot_build(backend_cpu, sched, ctx0, cur, selected_experts,
                gate_exps, up_exps, down_exps, hparams.swiglu_clamp_exp[il], il);
    }
    const bool xeon_hot_used = experts != nullptr;
    if (!xeon_hot_used) {
'''+body+'''    }

'''
    graph=graph[:split]+replacement+graph[end:]
    old='''    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }'''
    assert graph.count(old)==1
    graph=graph.replace(old,'''    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        if (xeon_hot_used) {
            xeon_hot_place_tail(backend_cpu, sched, experts);
        }
        cb(experts, "ffn_moe_weighted", il);
    }''',1)
    (OUT/'llama-context.cpp').write_text(context);(OUT/'llama-graph.cpp').write_text(graph)
    (OUT/'OVERLAY.json').write_text(json.dumps({'scope':'Only context backend insertion/finish and strict T1 MoE FFN replacement; public headers/layouts unchanged',
        'baseline_sources':{name:hashlib.sha256((SRC/name).read_bytes()).hexdigest() for name in ['llama-context.cpp','llama-graph.cpp']},
        'overlay_sources':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['llama-context.cpp','llama-graph.cpp','hot-backend.cpp','hot-backend.h']}},indent=2)+'\n')
    print('Prepared strict hot-FFN llama overlay; not compiled.',flush=True)
if __name__=='__main__':main()
