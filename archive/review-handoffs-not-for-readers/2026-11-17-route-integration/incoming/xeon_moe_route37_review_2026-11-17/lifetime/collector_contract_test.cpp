#include "route37.h"
#include "ggml-backend.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

// Contract-only harness. The original collector translation unit is compiled
// unchanged. This is the only linked ggml backend symbol. No ggml graph,
// arithmetic, model, CUDA operation, service, or supplied build script runs.
// Fake weight buffers are never dereferenced by this collector.
extern "C" bool ggml_backend_buffer_is_host(ggml_backend_buffer_t b) { return b != nullptr; }

static void require(bool b, const char *s) { if (!b) throw std::runtime_error(s); }
static const char *current_case = "startup";
static int cases = 0;

struct Fixture {
    std::array<ggml_tensor,37> weights{};
    std::array<const ggml_tensor *,37> gates{};
    std::array<unsigned char,37> data{};
    int32_t values[6]{0,13,26,39,52,65};
    ggml_tensor ids{};
    Fixture() {
        for (size_t i=0;i<weights.size();++i) {
            auto &w=weights[i];
            w.type=GGML_TYPE_MXFP4; w.data=&data[i];
            w.buffer=reinterpret_cast<ggml_backend_buffer_t>(&data[i]);
            w.ne[0]=256; w.ne[1]=4; w.ne[2]=384; w.ne[3]=1;
            w.nb[0]=1; w.nb[1]=256; w.nb[2]=1024; w.nb[3]=393216;
            gates[i]=&w;
        }
        ids.type=GGML_TYPE_I32;ids.data=values;
        ids.ne[0]=6;ids.ne[1]=ids.ne[2]=ids.ne[3]=1;
        ids.nb[0]=4;ids.nb[1]=ids.nb[2]=ids.nb[3]=24;
    }
    void open(uint32_t r=4096,uint32_t a=512,uint32_t c=128,int observing=1) {
        require(xeon_route37_open_v1(this,1,observing,r,a,c),"open");
    }
    xeon_route37_label tag(uint32_t call=1,uint8_t phase=XR_DECODE) {
        xeon_route37_label t{}; t.request=7;t.sequence=5;t.decode_call=call;
        t.phase=phase;t.logical_token=call+100;return t;
    }
    void label(uint32_t call=1,uint8_t phase=XR_DECODE) {
        auto t=tag(call,phase);require(xeon_route37_label_v1(this,&t),"label");
    }
    void begin(bool rebuilt=false,uint32_t nt=1,uint32_t npos=1,uint32_t nseq=1,int seq=5,int64_t pos=10) {
        require(xeon_route37_begin_eval_v1(this,gates.data(),gates.size(),rebuilt,nt,npos,nseq,seq,pos,0),"begin");
    }
    void records(unsigned count=37) {
        std::thread producer([&] { for(unsigned i=0;i<count;++i)xeon_route37_record_v1(gates[i],&ids); });
        producer.join();
    }
    void success(uint32_t call=1,uint8_t phase=XR_DECODE,bool rebuilt=false) {
        label(call,phase);begin(rebuilt);records();
        require(xeon_route37_end_eval_v1(this,GGML_STATUS_SUCCESS),"success end");
        require(xeon_route37_complete_call_v1(this,0,1),"success complete");
    }
    xeon_route37_stats stats() {
        xeon_route37_stats s{};require(xeon_route37_stats_v1(&s),"stats readable");return s;
    }
    xeon_route37_attempt attempt(size_t i) {
        xeon_route37_attempt a{};require(xeon_route37_read_attempt_v1(i,&a),"read attempt");return a;
    }
    void close() { require(xeon_route37_close_v1(this),"close"); }
    void emit() {
        const auto s=stats();
        std::printf("{\"case\":\"%s\",\"passed\":true,\"records\":%llu,\"attempts\":%llu,\"calls\":%llu,\"dropped\":%llu,\"complete_decode_attempts\":%llu,\"flags\":%u}\n",
            current_case,(unsigned long long)s.records,(unsigned long long)s.attempts,
            (unsigned long long)s.calls,(unsigned long long)s.dropped,
            (unsigned long long)s.complete_decode_attempts,s.flags);
        ++cases;
    }
};

int main() try {
    {
        current_case="phase_and_rebuild_generation";Fixture f;f.open();
        const uint8_t phase[]{XR_WARMUP,XR_PREFILL,XR_UNKNOWN,XR_DECODE,XR_DECODE};
        const uint64_t generation[]{1,1,1,2,2};
        for(unsigned i=0;i<5;++i) {
            f.success(i+1,phase[i],i==0||i==3);
            const auto a=f.attempt(i);
            require(a.id==i+1&&a.generation==generation[i]&&a.label.phase==phase[i],"phase/generation");
            require(a.caller_completed&&a.graph_returned&&a.records==37,"completed attempt");
        }
        auto s=f.stats();require(s.records==185&&s.complete_decode_attempts==2&&!s.flags,"phase counts");
        f.emit();f.close();
    }
    {
        current_case="call_limit_exact_then_overflow";Fixture f;f.open(4096,512,1);f.success();
        auto t=f.tag(2);require(!xeon_route37_label_v1(&f,&t),"call overflow refused");
        auto s=f.stats();require(s.calls==1&&s.attempts==1&&s.records==37&&s.dropped==1&&s.flags==XR_OVERFLOW&&!s.call_open,"call overflow ledger");
        xeon_route37_call c{};require(!xeon_route37_read_call_v1(1,&c),"call out of bounds");
        f.emit();f.close();
    }
    {
        current_case="read_after_close_and_reopen_reset";Fixture f;f.open();f.success();f.close();
        auto s=f.stats();require(!s.enabled&&s.records==37,"closed session readable");
        xeon_route37_record r{};require(xeon_route37_read_record_v1(36,&r),"closed record readable");
        f.open();s=f.stats();require(!s.records&&!s.attempts&&!s.calls&&!s.flags&&!s.dropped,"reopen resets");
        require(!xeon_route37_read_record_v1(0,&r),"stale index refused after reset");
        f.success(2);auto a=f.attempt(0);require(a.id==1&&a.generation==1&&a.label.decode_call==2,"new session IDs reset");
        f.emit();f.close();
    }
    {
        current_case="invalid_registration_duplicate_tensor";Fixture f;f.open();f.label();f.gates[1]=f.gates[0];f.begin();f.records();
        require(!xeon_route37_end_eval_v1(&f,0),"bad mapping end");
        require(!xeon_route37_complete_call_v1(&f,0,1),"bad mapping complete");
        auto s=f.stats();require(!s.records&&!s.complete_decode_attempts&&s.flags==(XR_MAPPING|XR_COVERAGE),"invalid mapping remains inactive");
        f.emit();f.close();
    }
    {
        current_case="invalid_registration_duplicate_data";Fixture f;f.open();f.label();f.weights[1].data=f.weights[0].data;f.begin();f.records();
        require(!xeon_route37_end_eval_v1(&f,0)&&!xeon_route37_complete_call_v1(&f,0,1),"data alias failure");
        auto s=f.stats();require(!s.records&&s.flags==(XR_MAPPING|XR_COVERAGE),"data alias remains inactive");f.emit();f.close();
    }
    {
        current_case="zero_attempt_success_status_is_invalid";Fixture f;f.open();f.label();
        require(!xeon_route37_complete_call_v1(&f,0,1),"no eval refused");
        auto s=f.stats();require(!s.attempts&&!s.records&&!s.complete_decode_attempts&&s.flags==XR_NO_EVAL,"no eval flags");f.emit();f.close();
    }
    {
        current_case="record_limit_overflow_preserves_old_records";Fixture f;f.open(37);f.success();
        xeon_route37_record old{};require(xeon_route37_read_record_v1(0,&old),"old record");
        f.label(2);f.begin();f.records();require(!xeon_route37_end_eval_v1(&f,0)&&!xeon_route37_complete_call_v1(&f,0,1),"record overflow fails");
        xeon_route37_record now{};require(xeon_route37_read_record_v1(0,&now)&&!std::memcmp(&old,&now,sizeof(old)),"old record retained");
        auto a=f.attempt(1);auto s=f.stats();
        require(a.records==0&&a.layers==((UINT64_C(1)<<37)-1)&&s.dropped==37&&s.flags==(XR_OVERFLOW|XR_COVERAGE)&&s.complete_decode_attempts==1,"overflow accounting");f.emit();f.close();
    }
    {
        current_case="attempt_limit_overflow_preserves_prior_call";Fixture f;f.open(4096,1);f.success();f.label(2);
        require(!xeon_route37_begin_eval_v1(&f,f.gates.data(),37,0,1,1,1,5,11,0),"attempt overflow refused");
        require(!xeon_route37_complete_call_v1(&f,0,1),"attempt overflow incomplete");
        auto s=f.stats();require(s.records==37&&s.attempts==1&&s.calls==2&&s.dropped==1&&s.flags==(XR_OVERFLOW|XR_NO_EVAL)&&s.complete_decode_attempts==1,"attempt overflow accounting");f.emit();f.close();
    }
    {
        current_case="failure_then_success_same_call_failclosed_then_retry_call";Fixture f;f.open();f.label();
        f.begin();f.records();require(!xeon_route37_end_eval_v1(&f,GGML_STATUS_ABORTED),"first attempt fails");
        f.begin();f.records();require(xeon_route37_end_eval_v1(&f,0),"second attempt succeeds");
        require(!xeon_route37_complete_call_v1(&f,0,1),"whole call remains failed");
        require(!f.attempt(0).caller_completed&&!f.attempt(1).caller_completed&&!f.stats().complete_decode_attempts,"no partial call counts");
        f.success(2);require(f.attempt(2).caller_completed&&f.stats().complete_decode_attempts==1,"retry new call counts despite historical flags");f.emit();f.close();
    }
    {
        current_case="unsynchronized_call_is_final_failure";Fixture f;f.open();f.label();f.begin();f.records();
        require(xeon_route37_end_eval_v1(&f,0),"CPU complete");
        require(!xeon_route37_complete_call_v1(&f,0,0),"unsynchronized failure");
        require(!xeon_route37_complete_call_v1(&f,0,1),"completion cannot repair closed call");
        require(!f.attempt(0).caller_completed&&f.stats().flags==XR_UNSYNCHRONIZED,"unsynchronized flags");f.emit();f.close();
    }
    {
        current_case="decode_multitoken_is_rejected";Fixture f;f.open();f.label();f.begin(false,2);f.records();
        require(!xeon_route37_end_eval_v1(&f,0)&&!xeon_route37_complete_call_v1(&f,0,1),"decode multi excluded");
        auto s=f.stats();require(!s.records&&!s.complete_decode_attempts&&s.flags==XR_GEOMETRY,"multitoken decode flags");f.emit();f.close();
    }
    {
        current_case="prefill_multitoken_excluded_without_decode_counts";Fixture f;f.open();f.label(1,XR_PREFILL);f.begin(false,2);f.records();
        require(xeon_route37_end_eval_v1(&f,0)&&xeon_route37_complete_call_v1(&f,0,1),"prefill out of scope is a completed call");
        auto s=f.stats();require(!s.records&&!s.complete_decode_attempts&&!s.flags&&!f.attempt(0).scope_t1,"prefill multi exclusion");f.emit();f.close();
    }
    std::printf("{\"summary\":true,\"passed\":true,\"cases\":%d,\"scope\":\"unchanged collector plus host-buffer stub and direct hooks; no ggml graph or model\"}\n",cases);
    return 0;
} catch(const std::exception &e) {
    std::fprintf(stderr,"FAIL %s: %s\n",current_case,e.what());return 1;
}
