#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace expert_shadow {
struct features {
    int64_t experts=0, tokens=0, topk=0;
    std::vector<uint32_t> counts;
    uint64_t selected_bytes=0, transfer_bytes=0, runs=0, active=0;
};

inline features describe(const void * raw, size_t bytes, int64_t experts, int64_t tokens,
                         int64_t topk, size_t stride_slot, size_t stride_token, size_t expert_bytes) {
    if(experts<1 || experts>8192 || tokens<1 || tokens>65536 || topk<1 || topk>experts || topk>64 ||
       stride_slot<4 || stride_token<4 || !raw || !expert_bytes) throw std::runtime_error("unsupported route shape");
    if(stride_slot>std::numeric_limits<size_t>::max()/size_t(topk) ||
       stride_token>std::numeric_limits<size_t>::max()/size_t(tokens)) throw std::runtime_error("route stride overflow");
    const size_t last_slot=size_t(topk-1)*stride_slot, last_token=size_t(tokens-1)*stride_token;
    if(last_token>bytes || last_slot>bytes-last_token || bytes-last_token-last_slot<4) throw std::runtime_error("route storage too short");
    if(uint64_t(expert_bytes)>std::numeric_limits<uint64_t>::max()/uint64_t(experts)) throw std::runtime_error("weight bytes overflow");
    features f;f.experts=experts;f.tokens=tokens;f.topk=topk;f.counts.resize(experts);
    std::vector<int64_t> seen(experts,-1);
    for(int64_t t=0;t<tokens;++t)for(int64_t j=0;j<topk;++j){
        int32_t id;std::memcpy(&id,static_cast<const uint8_t*>(raw)+size_t(t)*stride_token+size_t(j)*stride_slot,4);
        if(id<0 || id>=experts) throw std::runtime_error("route ID outside model");
        if(seen[id]==t) throw std::runtime_error("duplicate expert within token is unsupported");
        seen[id]=t;
        ++f.counts[id];
    }
    for(int64_t e=0;e<experts;){
        if(!f.counts[e]){++e;continue;}
        int64_t last=e;while(last+1<experts && f.counts[last+1])++last;
        const uint64_t n=uint64_t(last-e+1);f.active+=n;++f.runs;f.selected_bytes+=n*expert_bytes;
        const uint64_t base=n*expert_bytes,padding=last<experts-1?std::min<size_t>(expert_bytes,512):0;
        if(base>std::numeric_limits<uint64_t>::max()-padding || f.transfer_bytes>std::numeric_limits<uint64_t>::max()-base-padding)throw std::runtime_error("transfer bytes overflow");
        f.transfer_bytes+=base+padding;e=last+1;
    }
    return f;
}
}
