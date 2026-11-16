#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <random>

int main(int argc,char **argv) {
    if (argc!=2) return 1;
    auto file=gguf_init_empty();auto ctx=ggml_init({4*1024*1024,nullptr,false});
    if (!file||!ctx) return 2;
    gguf_set_val_str(file,"general.architecture","llama");
    gguf_set_val_str(file,"tokenizer.ggml.model","none");
    gguf_set_val_u32(file,"llama.context_length",128);
    gguf_set_val_u32(file,"llama.embedding_length",64);
    gguf_set_val_u32(file,"llama.block_count",1);
    gguf_set_val_u32(file,"llama.feed_forward_length",128);
    gguf_set_val_u32(file,"llama.attention.head_count",4);
    gguf_set_val_u32(file,"llama.attention.head_count_kv",4);
    gguf_set_val_u32(file,"llama.expert_count",4);
    gguf_set_val_u32(file,"llama.expert_used_count",2);
    gguf_set_val_u32(file,"llama.vocab_size",32);
    gguf_set_val_f32(file,"llama.attention.layer_norm_rms_epsilon",1e-5f);
    std::mt19937 rng(123);std::uniform_real_distribution<float> dist(-.02f,.02f);
    auto tensor=[&](const char *name,int k,int m=1,int e=1) {
        auto t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,m,e);ggml_set_name(t,name);
        for (int64_t i=0;i<ggml_nelements(t);++i) static_cast<float *>(t->data)[i]=m==1&&e==1?1.f:dist(rng);
        gguf_add_tensor(file,t);
    };
    tensor("token_embd.weight",64,32);tensor("output.weight",64,32);tensor("output_norm.weight",64);
    tensor("blk.0.attn_norm.weight",64);tensor("blk.0.ffn_norm.weight",64);
    for(const char *n:{"blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight","blk.0.attn_output.weight"}) tensor(n,64,64);
    tensor("blk.0.ffn_gate_inp.weight",64,4);
    tensor("blk.0.ffn_gate_exps.weight",64,128,4);tensor("blk.0.ffn_up_exps.weight",64,128,4);
    tensor("blk.0.ffn_down_exps.weight",128,64,4);
    const bool ok=gguf_write_to_file(file,argv[1],false);
    gguf_free(file);ggml_free(ctx);
    if(ok)std::puts("Synthetic sub-megabyte MoE fixture written");return ok?0:3;
}
