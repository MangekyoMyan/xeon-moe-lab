#!/usr/bin/env python3
"""Generate a sub-megabyte random MoE loader fixture, not a usable language model."""
from pathlib import Path
import sys
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'deepseek41-bringup-20261101/source/gguf-py'))
import gguf

w=gguf.GGUFWriter(str(R/'tiny-moe.gguf'),'llama')
for key,value in {'llama.context_length':128,'llama.embedding_length':64,'llama.block_count':1,
 'llama.feed_forward_length':128,'llama.attention.head_count':4,'llama.attention.head_count_kv':4,
 'llama.expert_count':4,'llama.expert_used_count':2,'llama.vocab_size':32}.items():w.add_uint32(key,value)
w.add_float32('llama.attention.layer_norm_rms_epsilon',1e-5)
w.add_string('tokenizer.ggml.model','none')
rng=np.random.default_rng(123)
shapes={'token_embd.weight':(32,64),'output.weight':(32,64),'output_norm.weight':(64,),
 'blk.0.attn_norm.weight':(64,),'blk.0.ffn_norm.weight':(64,),
 'blk.0.attn_q.weight':(64,64),'blk.0.attn_k.weight':(64,64),'blk.0.attn_v.weight':(64,64),
 'blk.0.attn_output.weight':(64,64),'blk.0.ffn_gate_inp.weight':(4,64),
 'blk.0.ffn_gate_exps.weight':(4,128,64),'blk.0.ffn_up_exps.weight':(4,128,64),
 'blk.0.ffn_down_exps.weight':(4,64,128)}
for name,shape in shapes.items():
    value=np.ones(shape,dtype=np.float32) if len(shape)==1 else rng.normal(0,.02,shape).astype(np.float32)
    w.add_tensor(name,value)
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
print('Synthetic loader fixture:',R/'tiny-moe.gguf')
