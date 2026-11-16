#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu/quants.h"
#include <cstring>
#include <cstdio>
#include <random>

int main(int argc,char **argv) {
    if (argc!=2) return 1;
    auto file=gguf_init_empty();auto ctx=ggml_init({32*1024*1024,nullptr,false});
    if (!file||!ctx) return 2;
    gguf_set_val_str(file,"general.architecture","llama");
    gguf_set_val_str(file,"tokenizer.ggml.model","none");
    gguf_set_val_u32(file,"llama.context_length",128);
    gguf_set_val_u32(file,"llama.embedding_length",512);
    gguf_set_val_u32(file,"llama.block_count",1);
    gguf_set_val_u32(file,"llama.feed_forward_length",256);
    gguf_set_val_u32(file,"llama.attention.head_count",4);
    gguf_set_val_u32(file,"llama.attention.head_count_kv",4);
    gguf_set_val_u32(file,"llama.expert_count",4);
    gguf_set_val_u32(file,"llama.expert_used_count",2);
    gguf_set_val_u32(file,"llama.vocab_size",32);
    gguf_set_val_f32(file,"llama.attention.layer_norm_rms_epsilon",1e-5f);
    std::mt19937 rng(123);std::uniform_real_distribution<float> dist(-.02f,.02f);
    auto tensor=[&](const char *name,int k,int m=1,int e=1) {
        auto t=ggml_new_tensor_3d(ctx,e>1?GGML_TYPE_IQ3_XXS:GGML_TYPE_F32,k,m,e);ggml_set_name(t,name);
        if(e>1) {
            auto p=static_cast<block_iq3_xxs *>(t->data);
            for(size_t b=0;b<ggml_nbytes(t)/sizeof(*p);++b){p[b].d=0x1000;for(auto &q:p[b].qs)q=rng();}
        } else for (int64_t i=0;i<ggml_nelements(t);++i) static_cast<float *>(t->data)[i]=m==1?1.f:dist(rng);
        gguf_add_tensor(file,t);
    };
    tensor("token_embd.weight",512,32);tensor("output.weight",512,32);tensor("output_norm.weight",512);
    tensor("blk.0.attn_norm.weight",512);tensor("blk.0.ffn_norm.weight",512);
    for(const char *n:{"blk.0.attn_q.weight","blk.0.attn_k.weight","blk.0.attn_v.weight","blk.0.attn_output.weight"}) tensor(n,512,512);
    tensor("blk.0.ffn_gate_inp.weight",512,4);
    tensor("blk.0.ffn_gate_exps.weight",512,256,4);tensor("blk.0.ffn_up_exps.weight",512,256,4);
    tensor("blk.0.ffn_down_exps.weight",256,512,4);
    const bool ok=gguf_write_to_file(file,argv[1],false);
    gguf_free(file);ggml_free(ctx);
    if(ok)std::puts("Synthetic IQ3 MoE fixture written");return ok?0:3;
}
