// Functional audit only. Includes the supplied helper TU, not the operator/backend.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include "iq-abs.cpp"

extern "C" void ggml_abort(const char *, int, const char *, ...) {
    throw std::runtime_error("GGML_ASSERT rejected input");
}
extern "C" size_t ggml_type_size(enum ggml_type type) {
    if (type == GGML_TYPE_F32) return sizeof(float);
    if (type == GGML_TYPE_Q8_K) return sizeof(block_q8_K);
    throw std::runtime_error("audit stub type not supported");
}
static void require(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }
static size_t align_up(size_t value, size_t alignment) { return (value+alignment-1)&~(alignment-1); }
static uint32_t bits(float value) { uint32_t u; std::memcpy(&u,&value,4); return u; }
static bool input_safe(const block_q8_K * x, int blocks) {
    for (int b=0;b<blocks;++b) {
        if ((bits(x[b].d)&0x7f800000u)==0x7f800000u) return false;
        for (int8_t value:x[b].qs) if (value==-128) return false;
    }
    return true;
}

// Mathematical translation of ggml-cpu.c:3167-3198,3335-3336 for T=1 IQ tensors.
// Tiled bytes are supplied by its documented 64 + nth*512KiB formula or zero.
static size_t planned_size(int k,int inputs,int planned,int experts,int slots,bool quantize,bool tiled) {
    size_t cur=quantize?size_t(k/256)*sizeof(block_q8_K)*inputs+8:0;
    cur+=size_t(experts)*8+8;
    cur+=size_t(experts)*slots*8+8;
    cur+=size_t(experts)*64+64;
    cur=align_up(cur,64);
    cur+=tiled?64+size_t(planned)*512*1024:0;
    cur+=xeon_iq_abs_bytes(k,inputs,planned)+63;
    return cur+size_t(planned)*64;
}

int main() try {
    static_assert(sizeof(block_q8_K)==292,"Q8_K layout");
    size_t layout_checks=0,shape_checks=0,assert_checks=0,prepare_calls=0;
    size_t empty_ranges=0,mask_checks=0,sidecar_byte_checks=0,phase_cases=0;
    for(int k:{256,2304,5120})for(int inputs:{1,6,64})for(int planned:{1,6,12,24})
    for(int actual=1;actual<=planned;++actual)for(int experts:{8,384})for(bool quantize:{false,true})
    for(bool tiled:{false,true})for(size_t residue=0;residue<64;++residue) {
        const int slots=std::max(6,inputs);
        const size_t base=4096+residue;
        size_t cur=base;
        auto reserve=[&](size_t n,size_t a){cur=align_up(cur,a);size_t start=cur;cur+=n;return start;};
        if(quantize)reserve(size_t(k/256)*sizeof(block_q8_K)*inputs,8);
        reserve(size_t(experts)*8,8);
        reserve(size_t(experts)*slots*8,8);
        reserve(size_t(experts)*64,64);
        reserve(tiled?64+size_t(actual)*512*1024:0,64);
        size_t abs=reserve(xeon_iq_abs_bytes(k,inputs,actual),64);
        size_t masks=abs+size_t(k)*std::max(inputs,actual);
        require(abs%64==0&&masks%8==0,"alignment");
        require(abs+size_t(k)*inputs<=masks,"shared overlaps mask");
        require(abs+size_t(k)*actual<=masks,"private overlaps mask");
        require(masks+size_t(actual)*8==cur,"mask end");
        require(cur-base<=planned_size(k,inputs,planned,experts,slots,quantize,tiled),"planned capacity too small");
        ++layout_checks;
    }
    ggml_tensor w{},x{},ids{};
    w.type=GGML_TYPE_IQ2_S;x.type=GGML_TYPE_F32;x.ne[0]=256;x.ne[1]=6;x.ne[2]=x.ne[3]=1;x.nb[0]=4;ids.ne[1]=1;
    auto shape=[&](bool expected){require(xeon_iq_abs_shape(&w,&x,&ids)==expected,"shape");++shape_checks;};
    shape(true);w.type=GGML_TYPE_IQ3_S;shape(true);w.type=GGML_TYPE_IQ3_XXS;shape(false);w.type=GGML_TYPE_MXFP4;shape(false);w.type=GGML_TYPE_IQ2_S;
    x.type=GGML_TYPE_Q8_K;x.nb[0]=sizeof(block_q8_K);shape(true);
    x.ne[1]=64;shape(true);x.ne[1]=65;shape(false);x.ne[1]=0;shape(false);x.ne[1]=6;
    x.ne[2]=3;shape(false);x.ne[2]=1;x.ne[3]=2;shape(false);x.ne[3]=1;
    ids.ne[1]=3;shape(false);ids.ne[1]=1;x.ne[0]=255;shape(false);x.ne[0]=0;shape(false);
    x.ne[0]=int64_t(INT_MAX)+1;shape(false);x.ne[0]=INT_MAX-255;shape(true);x.ne[0]=256;
    x.nb[0]+=4;shape(false);x.nb[0]=sizeof(block_q8_K);
    for(auto bad:std::array<std::array<int64_t,3>,4>{{{{0,1,1}},{{256,0,1}},{{256,65,1}},{{256,1,0}}}}) {
        bool rejected=false;try{(void)xeon_iq_abs_bytes(bad[0],bad[1],int(bad[2]));}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"invalid size args accepted");++assert_checks;
    }

    // Reuse one aligned buffer across worker-count changes and safety phases.
    // No real threads: this verifies disjoint complete-block writes and publication values,
    // not the actual ggml/OpenMP barrier or scheduler.
    for(int k:{256,2304,5120})for(int inputs:{1,6,64}) {
        const int blocks=k/256,stride=blocks+1,planned=24;
        std::vector<block_q8_K> q(size_t(inputs)*stride);
        std::vector<uint8_t> storage(xeon_iq_abs_bytes(k,inputs,planned)+128,0xa5);
        const uintptr_t address=align_up(reinterpret_cast<uintptr_t>(storage.data()+16),64);
        auto *abs=reinterpret_cast<uint8_t*>(address);
        const size_t begin=abs-storage.data();
        for(int actual:{24,12,1,6,3,2})for(int phase=0;phase<6;++phase) {
            for(int input=0;input<inputs;++input)for(int b=0;b<stride;++b) {
                auto&v=q[size_t(input)*stride+b];v.d=.00390625f;
                for(int j=0;j<256;++j)v.qs[j]=int8_t((input*31+b*17+j*7+phase*13)%255-127);
            }
            if(phase==1)q[0].qs[0]=-128;
            if(phase==2)q[size_t(inputs-1)*stride+blocks-1].d=std::numeric_limits<float>::infinity();
            if(phase==3){uint32_t nan=0x7fc00001;std::memcpy(&q[size_t(inputs/2)*stride+blocks-1].d,&nan,4);}
            if(phase==4)for(int input=0;input<inputs;++input)if(input%3==1||input==63)q[size_t(input)*stride+blocks-1].qs[255]=-128;
            uint64_t combined=UINT64_MAX;
            for(int worker=actual-1;worker>=0;--worker) {
                const int first=worker*blocks/actual,end=(worker+1)*blocks/actual;
                if(first==end)++empty_ranges;
                uint64_t local=0;
                for(int input=0;input<inputs;++input) {
                    int safe=xeon_iq_abs_prepare(k,q.data()+size_t(input)*stride,first,end,abs+size_t(input)*k);
                    if(safe)local|=UINT64_C(1)<<input;
                    ++prepare_calls;
                }
                combined&=local;
            }
            uint64_t expected=0;
            for(int input=0;input<inputs;++input) {
                const auto *row=q.data()+size_t(input)*stride;
                if(input_safe(row,blocks))expected|=UINT64_C(1)<<input;
                for(int b=0;b<blocks;++b)for(int j=0;j<256;++j) {
                    int value=row[b].qs[j];if(value<0)value=-value;
                    require(abs[size_t(input)*k+256*b+j]==uint8_t(value),"prepared byte mismatch");++sidecar_byte_checks;
                }
            }
            require(combined==expected,"mask mismatch");++mask_checks;++phase_cases;
            require(std::all_of(storage.begin(),storage.begin()+begin,[](uint8_t v){return v==0xa5;}),"leading canary");
            require(std::all_of(storage.begin()+begin+size_t(inputs)*k,storage.end(),[](uint8_t v){return v==0xa5;}),"trailing canary");
        }
    }

    // Deliberate violation of iq-abs.h's no-in-flight-mode-change rule, staged without
    // racing threads. It shows why atomic mode alone cannot provide one operator snapshot.
    alignas(64) uint8_t prepared[1024]{};
    block_q8_K q[2]{};iq_compact_block weights[2]{};
    for(auto&v:q){v.d=1;for(auto&value:v.qs)value=1;}
    for(auto&v:weights){v.d=1;std::memset(v.qs,0,sizeof(v.qs));std::memset(v.scales,0,sizeof(v.scales));}
    xeon_iq_abs_set_mode(1);int worker0=xeon_iq_abs_get_mode();
    uint64_t mask0=xeon_iq_abs_prepare(512,q,0,1,prepared)?1:0;
    uint64_t stale_mask1=1; // A valid value left by a prior shared call.
    xeon_iq_abs_set_mode(2);int worker1=xeon_iq_abs_get_mode();
    (void)xeon_iq_abs_prepare(512,q,0,2,prepared+512);
    float old=iq_compact_dot<false>(512,weights,q),mixed=0;
    require((mask0&stale_mask1)==1,"staged shared selection");
    xeon_iq3s_abs_dot(512,&mixed,weights,q,prepared);
    require(old==512&&mixed==256,"mode counterexample changed");
    xeon_iq_abs_set_mode(0);
    std::printf("{\"passed\":true,\"scope\":\"Actual abs helper plus mathematical layout/partition checks; not full ggml backend\",\"layout_checks\":%zu,\"shape_checks\":%zu,\"invalid_argument_rejections\":%zu,\"phase_cases\":%zu,\"prepare_calls\":%zu,\"empty_worker_ranges\":%zu,\"mask_checks\":%zu,\"sidecar_byte_checks\":%zu,\"forbidden_mode_switch_demo\":{\"worker_modes\":[%d,%d],\"baseline\":%.1f,\"mixed\":%.1f,\"current_api_contract_violated\":true},\"performance_measured\":false}\n",layout_checks,shape_checks,assert_checks,phase_cases,prepare_calls,empty_ranges,mask_checks,sidecar_byte_checks,worker0,worker1,old,mixed);
    return 0;
}catch(const std::exception&error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
