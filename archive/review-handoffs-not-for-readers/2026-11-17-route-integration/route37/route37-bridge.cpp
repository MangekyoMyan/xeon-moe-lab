#include "route37-bridge.h"
#include "route37.h"
#include "llama-context.h"
#include "llama-batch.h"
#include "llama-model.h"
#include <array>
int xeon_route37_bridge_begin(llama_context *ctx,const llama_ubatch &u,bool rebuilt,bool pipeline){
    if(!xeon_route37_matches_v1(ctx))return 0;
    const auto &model=ctx->get_model();std::array<const ggml_tensor*,37> gates{};
    bool valid=model.layers.size()==40;
    for(size_t i=0;valid&&i<gates.size();++i){
        const auto *w=model.layers[i].ffn_gate_exps;
        valid=w&&w->ne[0]==5120&&w->ne[1]==2304&&w->ne[2]==384&&w->ne[3]==1;
        gates[i]=w;
    }
    const bool one_sequence=u.n_tokens&&u.n_seq_id&&u.seq_id&&u.n_seq_id[0]==1&&u.seq_id[0];
    // Never infer a decode phase from T=1. The controlling diagnostic caller
    // must open and label this decode call explicitly through the route API.
    return xeon_route37_begin_eval_v1(ctx,valid?gates.data():nullptr,valid?gates.size():0,
        rebuilt,u.n_tokens,u.n_pos,one_sequence?1:0,
        one_sequence?u.seq_id[0][0]:-1,u.pos&&u.n_tokens?u.pos[0]:-1,pipeline);
}
