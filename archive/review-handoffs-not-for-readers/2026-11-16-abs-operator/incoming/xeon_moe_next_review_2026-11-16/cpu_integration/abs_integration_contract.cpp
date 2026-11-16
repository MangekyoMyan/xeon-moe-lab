// Contract fixture for proposed shared and worker-private input-abs integration.
// It is not ggml-cpu.c, the ggml threadpool, a model benchmark, or a backend patch.
#define main original_abs_probe_main_not_called
#include "iq_abs_rhs_probe.cpp"
#undef main
#include <condition_variable>
#include <mutex>
#include <thread>
#include <numeric>

class publish_barrier {
    std::mutex mutex_;
    std::condition_variable cv_;
    const int count_;
    int remaining_;
    unsigned generation_=0;
public:
    explicit publish_barrier(int n):count_(n),remaining_(n) {}
    void arrive() {
        std::unique_lock<std::mutex> lock(mutex_);
        const unsigned generation=generation_;
        if (--remaining_==0) { remaining_=count_;++generation_;cv_.notify_all(); }
        else cv_.wait(lock,[&]{return generation_!=generation;});
    }
};

static uint64_t valid_inputs(int count) {
    need(count>=1&&count<=64,"input-count guard");
    return count==64?UINT64_MAX:(UINT64_C(1)<<count)-1;
}

struct shape {
    int64_t k,inputs,tokens,x2,x3;
    bool selected_type,source_supported;
};
static bool eligible_shape(shape s) {
    return s.selected_type && s.source_supported && s.tokens==1 && s.x2==1 && s.x3==1 &&
        s.inputs>=1 && s.inputs<=64 && s.k>0 && s.k<=INT32_MAX && s.k%256==0;
}
static size_t checked_add(size_t a,size_t b) {
    need(a<=SIZE_MAX-b,"size addition overflow");return a+b;
}
static size_t checked_mul(size_t a,size_t b) {
    need(!a||b<=SIZE_MAX/a,"size multiplication overflow");return a*b;
}
static size_t align_up(size_t n,size_t alignment) {
    return checked_add(n,alignment-1)&~(alignment-1);
}
struct layout {size_t sidecar, masks, end;};
static layout shared_layout(size_t absolute_base,size_t prefix,int k,int inputs,int workers) {
    // Actual absolute addresses are represented as integers solely for geometry checks.
    size_t cursor=checked_add(absolute_base,prefix);
    size_t sidecar=align_up(cursor,64);
    cursor=checked_add(sidecar,checked_mul(align_up(k,64),inputs));
    size_t masks=align_up(cursor,alignof(uint64_t));
    size_t end=checked_add(masks,checked_mul(workers,sizeof(uint64_t)));
    return {sidecar-absolute_base,masks-absolute_base,end-absolute_base};
}

struct counters {size_t outputs=0, fast=0, raw=0, fallback=0, prepared_input_calls=0, empty_ranges=0;};
struct totals {size_t cases=0, outputs=0, fast=0, raw=0, fallback=0, prepare_calls=0, empty_ranges=0, mask_checks=0, layout_checks=0, shape_checks=0;};

template<int T> static float raw_dot(int k,const source_type<T> *w,const block_q8_K *x) {
    float result;
    if constexpr(T==2)ggml_vec_dot_iq2_s_q8_K(k,&result,0,w,0,x,0,1);
    else ggml_vec_dot_iq3_s_q8_K(k,&result,0,w,0,x,0,1);
    return result;
}

template<int T> static void one_case(totals &total,int k,int inputs,int workers,bool prequantized,
                                    bool private_lazy,bool use_ref,int hit_pattern,int bad_kind) {
    const int blocks=k/256,slots=inputs==1?6:inputs,rows=17;
    std::mt19937 rng(202611160u+T+unsigned(k)+inputs*7+workers);
    const size_t float_stride=k+16;
    const size_t qstride=blocks+(prequantized?1:0);
    std::vector<float> f32(inputs*float_stride,0.f);
    std::vector<block_q8_K> q(inputs*qstride),expected_q(inputs*qstride);
    for(int i=0;i<inputs;++i) {
        for(int j=0;j<k;++j)f32[i*float_stride+j]=float(int(rng()%2001)-1000)*.0001f;
        quantize_row_q8_K(f32.data()+i*float_stride,expected_q.data()+i*qstride,k);
    }
    if(prequantized)q=expected_q;
    else std::memset(q.data(),0x5a,q.size()*sizeof(block_q8_K));
    if(bad_kind) {
        need(prequantized,"unsafe fixture must use prequantized input");
        block_q8_K &last=q[(inputs-1)*qstride+blocks-1];
        if(bad_kind==1)last.qs[255]=-128;
        else {uint32_t bits=bad_kind==2?0x7f800000u:0xffc00001u;std::memcpy(&last.d,&bits,4);}
        expected_q=q;
    }
    std::vector<source_type<T>> raw(size_t(slots)*rows*blocks);
    std::vector<weight_type<T>> packed(raw.size());
    for(size_t i=0;i<raw.size();++i) {
        auto *p=reinterpret_cast<uint8_t *>(&raw[i]);
        for(size_t b=0;b<sizeof(raw[i]);++b)p[b]=uint8_t(rng());
        raw[i].d=0x2400;
        iq_pack<T==2>(raw[i],packed[i]);
    }
    std::vector<int> order(slots);std::iota(order.begin(),order.end(),0);
    // Nontrivial expert order, while outputs/input indices retain original slots.
    std::shuffle(order.begin(),order.end(),rng);
    std::vector<bool> compact(slots);
    uint64_t required=0;
    for(int slot=0;slot<slots;++slot) {
        compact[slot]=!use_ref&&(hit_pattern==2||(hit_pattern==1&&slot%2==0));
        if(compact[slot])required|=UINT64_C(1)<<(slot%inputs);
    }
    std::vector<abs_rhs> shared(size_t(inputs)*blocks),private_data(size_t(workers)*blocks);
    std::memset(shared.data(),0x6b,shared.size()*sizeof(abs_rhs));
    std::memset(private_data.data(),0xa5,private_data.size()*sizeof(abs_rhs));
    std::vector<uint64_t> masks(workers,UINT64_C(0xdeadbeef)),observed_masks(workers,0);
    std::vector<float> output(size_t(slots)*rows);
    std::vector<counters> counts(workers);
    publish_barrier barrier(workers);
    std::vector<std::thread> threads;
    for(int worker=0;worker<workers;++worker)threads.emplace_back([&,worker] {
        const int first=worker*blocks/workers,end=(worker+1)*blocks/workers;
        if(first==end)++counts[worker].empty_ranges;
        if(!prequantized)for(int input=0;input<inputs;++input)
            quantize_row_q8_K(f32.data()+input*float_stride+first*256,
                             q.data()+input*qstride+first,(end-first)*256);
        uint64_t local=0;
        if(!private_lazy)for(int input=0;input<inputs;++input)if((required>>input)&1) {
            bool safe=prepare_abs((end-first)*256,q.data()+input*qstride+first,
                                  shared.data()+input*blocks+first);
            if(safe)local|=UINT64_C(1)<<input;
            ++counts[worker].prepared_input_calls;
        }
        masks[worker]=local; // Empty ranges publish true for every required input.
        if(worker%2==0)std::this_thread::yield();
        barrier.arrive(); // This fixture's CV barrier, not the ggml implementation.
        uint64_t safe_mask=valid_inputs(inputs);
        if(!private_lazy)for(uint64_t mask:masks)safe_mask&=mask;
        observed_masks[worker]=safe_mask;
        int last_input=-1;bool last_safe=false; // Reset on every operator invocation.
        for(int slot:order)for(int row=worker;row<rows;row+=workers) {
            const int input=slot%inputs;
            const auto *x=q.data()+input*qstride;
            const size_t offset=(size_t(slot)*rows+row)*blocks;
            bool safe=false;const abs_rhs *prepared=nullptr;
            if(compact[slot]) {
                if(private_lazy) {
                    if(last_input!=input) {
                        last_safe=prepare_abs(k,x,private_data.data()+worker*blocks);
                        last_input=input;++counts[worker].prepared_input_calls;
                    }
                    safe=last_safe;prepared=private_data.data()+worker*blocks;
                } else {safe=(safe_mask>>input)&1;prepared=shared.data()+input*blocks;}
            }
            float value;
            if(compact[slot]&&safe) {value=abs_dot<T>(k,packed.data()+offset,x,prepared);++counts[worker].fast;}
            else if(compact[slot]) {value=current_dot<T>(k,packed.data()+offset,x);++counts[worker].fallback;}
            else {value=raw_dot<T>(k,raw.data()+offset,x);++counts[worker].raw;}
            output[size_t(slot)*rows+row]=value;++counts[worker].outputs;
        }
    });
    for(auto &thread:threads)thread.join();
    for(int input=0;input<inputs;++input)
        need(std::memcmp(q.data()+input*qstride,expected_q.data()+input*qstride,blocks*sizeof(block_q8_K))==0,"partition quantization mismatch");
    if(!private_lazy) {
        std::vector<abs_rhs> exact(blocks);uint64_t expected_mask=0;
        for(int input=0;input<inputs;++input)if((required>>input)&1) {
            if(prepare_abs(k,q.data()+input*qstride,exact.data()))expected_mask|=UINT64_C(1)<<input;
            need(std::memcmp(exact.data(),shared.data()+input*blocks,k)==0,"partition abs mismatch");
        }
        for(uint64_t mask:observed_masks){need(mask==expected_mask,"worker mask mismatch");++total.mask_checks;}
    }
    for(int slot=0;slot<slots;++slot)for(int row=0;row<rows;++row) {
        const size_t offset=(size_t(slot)*rows+row)*blocks;
        const auto *x=q.data()+(slot%inputs)*qstride;
        float expected=compact[slot]?current_dot<T>(k,packed.data()+offset,x):raw_dot<T>(k,raw.data()+offset,x);
        need(fb(expected)==fb(output[size_t(slot)*rows+row]),"integrated branch bit mismatch");
    }
    for(auto c:counts){total.outputs+=c.outputs;total.fast+=c.fast;total.raw+=c.raw;total.fallback+=c.fallback;total.prepare_calls+=c.prepared_input_calls;total.empty_ranges+=c.empty_ranges;}
    if(hit_pattern==0||use_ref)for(auto c:counts)need(c.prepared_input_calls==0&&c.fast==0,"all-miss/use_ref prepared sidecar");
    ++total.cases;
}

int main() {
    try {
        for(int i=0;i<65536;++i)ggml_table_f32_f16[i]=GGML_COMPUTE_FP16_TO_FP32(ggml_fp16_t(i));
        totals t;
        for(int k:{256,2304,5120})for(int inputs:{1,6,64})for(int workers:{1,6,12})
            for(int preq:{0,1})for(int priv:{0,1}) {
                // All hit, mixed hit, all miss; skip large-64 all-hit repeats only by fixed matrix below.
                for(int hit:{0,1,2}) {
                    one_case<2>(t,k,inputs,workers,preq,priv,false,hit,0);
                    one_case<3>(t,k,inputs,workers,preq,priv,false,hit,0);
                }
            }
        for(int inputs:{1,6,64})for(int priv:{0,1})for(int bad:{1,2,3}) {
            one_case<2>(t,2304,inputs,12,true,priv,false,2,bad);
            one_case<3>(t,2304,inputs,12,true,priv,false,2,bad);
        }
        for(int priv:{0,1})for(int preq:{0,1}) {
            one_case<2>(t,256,6,12,preq,priv,true,2,0);
            one_case<3>(t,256,6,12,preq,priv,true,2,0);
        }
        for(size_t base=0;base<64;++base)for(size_t prefix:{0,1,7,64,12345})
            for(int k:{256,2304,5120})for(int inputs:{1,6,64})for(int workers:{1,6,12}) {
                const layout actual=shared_layout(base,prefix,k,inputs,workers);
                const layout planned=shared_layout(0,prefix,k,inputs,workers);
                need(actual.end<=checked_add(planned.end,63),"layout needs more than one max-alignment allowance");
                need((base+actual.sidecar)%64==0&&(base+actual.masks)%8==0,"alignment mismatch");
                need(actual.masks>=actual.sidecar+size_t(k)*inputs,"sidecar/mask overlap");++t.layout_checks;
            }
        shape good{2304,6,1,1,1,true,true};need(eligible_shape(good),"shape positive");++t.shape_checks;
        for(auto bad:std::vector<shape>{{0,6,1,1,1,true,true},{257,6,1,1,1,true,true},{2304,0,1,1,1,true,true},{2304,65,1,1,1,true,true},{2304,6,2,1,1,true,true},{2304,6,1,2,1,true,true},{2304,6,1,1,2,true,true},{2304,6,1,1,1,false,true},{2304,6,1,1,1,true,false}}){need(!eligible_shape(bad),"shape negative");++t.shape_checks;}
        bool rejected=false;try{align_up(SIZE_MAX,64);}catch(const std::runtime_error&){rejected=true;}need(rejected,"overflow guard");
        std::printf("{\"passed\":true,\"operator_contract_cases\":%zu,\"bit_exact_outputs\":%zu,\"abs_fast_outputs\":%zu,\"raw_outputs\":%zu,\"unsafe_compact_fallback_outputs\":%zu,\"prepared_input_calls\":%zu,\"empty_worker_ranges\":%zu,\"mask_checks\":%zu,\"layout_checks\":%zu,\"shape_checks\":%zu}\n",t.cases,t.outputs,t.fast,t.raw,t.fallback,t.prepare_calls,t.empty_ranges,t.mask_checks,t.layout_checks,t.shape_checks);
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
