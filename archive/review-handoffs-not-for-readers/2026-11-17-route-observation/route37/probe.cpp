#include "route37.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
static void need(bool p,const char*s){if(!p)throw std::runtime_error(s);}
struct API {
    decltype(&xeon_route37_open_v1) open;
    decltype(&xeon_route37_label_v1) label;
    decltype(&xeon_route37_begin_eval_v1) begin;
    decltype(&xeon_route37_end_eval_v1) end;
    decltype(&xeon_route37_complete_call_v1) finish;
    decltype(&xeon_route37_close_v1) close;
    decltype(&xeon_route37_record_v1) record;
    decltype(&xeon_route37_stats_v1) stats;
    decltype(&xeon_route37_read_record_v1) read;
    decltype(&xeon_route37_read_attempt_v1) attempt;
    decltype(&xeon_route37_read_call_v1) call;
    API(){
#define LOAD(member,name) member=reinterpret_cast<decltype(member)>(dlsym(RTLD_DEFAULT,name));need(member,name)
        LOAD(open,"xeon_route37_open_v1");LOAD(label,"xeon_route37_label_v1");LOAD(begin,"xeon_route37_begin_eval_v1");
        LOAD(end,"xeon_route37_end_eval_v1");LOAD(finish,"xeon_route37_complete_call_v1");LOAD(close,"xeon_route37_close_v1");
        LOAD(record,"xeon_route37_record_v1");LOAD(stats,"xeon_route37_stats_v1");LOAD(read,"xeon_route37_read_record_v1");
        LOAD(attempt,"xeon_route37_read_attempt_v1");LOAD(call,"xeon_route37_read_call_v1");
#undef LOAD
    }
};
struct Fixture {
    ggml_backend_t cpu=ggml_backend_cpu_init();ggml_context *ctx=nullptr;ggml_cgraph *graph=nullptr;
    ggml_backend_buffer_t buffer=nullptr;ggml_threadpool_t pool=nullptr;
    std::array<const ggml_tensor*,37> gates{};std::array<ggml_tensor*,37> ids{},out{};ggml_tensor *input=nullptr;
    Fixture(){
        need(cpu,"cpu");ctx=ggml_init({4*1024*1024,nullptr,true});need(ctx,"ctx");
        graph=ggml_new_graph_custom(ctx,256,false);input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,1,1);
        const ggml_type types[]={GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4};
        for(int i=0;i<37;++i){auto *w=ggml_new_tensor_3d(ctx,types[i%4],256,4,384);gates[i]=w;
            ids[i]=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,1);out[i]=ggml_mul_mat_id(ctx,w,input,ids[i]);ggml_build_forward_expand(graph,out[i]);}
        buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buffer,"buffer");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 rng(20261117);
        for(auto *w:gates){std::vector<uint8_t>v(ggml_nbytes(w));for(auto &b:v)b=uint8_t(rng());
            for(size_t j=0;j<v.size();j+=ggml_type_size(w->type)){if(w->type==GGML_TYPE_MXFP4)v[j]=127;else{uint16_t scale=0x2000;std::memcpy(v.data()+j,&scale,2);}}
            ggml_backend_tensor_set(const_cast<ggml_tensor*>(w),v.data(),0,v.size());}
        auto p=ggml_threadpool_params_default(12);p.poll=0;p.strict_cpu=true;for(int i=0;i<12;++i)p.cpumask[i]=true;
        pool=ggml_threadpool_new(&p);need(pool,"pool");ggml_backend_cpu_set_threadpool(cpu,pool);ggml_backend_cpu_set_n_threads(cpu,12);
    }
    ~Fixture(){ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(cpu);}
    void inputs(int cycle){
        float x[256];for(int i=0;i<256;++i)x[i]=float((i*17+cycle*19)%1000-500)/1024;
        ggml_backend_tensor_set(input,x,0,sizeof(x));
        for(int layer=0;layer<37;++layer){int32_t r[6];for(int j=0;j<6;++j)r[j]=(layer*7+cycle*11+j*13)%384;ggml_backend_tensor_set(ids[layer],r,0,sizeof(r));}
    }
    std::vector<float> run(){
        int status=-999;std::thread worker([&]{status=ggml_backend_graph_compute(cpu,graph);});worker.join();need(status==GGML_STATUS_SUCCESS,"graph");
        std::vector<float>v(37*24);for(int i=0;i<37;++i)ggml_backend_tensor_get(out[i],v.data()+i*24,0,24*sizeof(float));
        for(float x:v)need(std::isfinite(x),"nonfinite fixture output");
        return v;
    }
};
static xeon_route37_label tag(int n,int phase=XR_DECODE){xeon_route37_label t{};t.request=7;t.sequence=5;t.decode_call=42+n;t.phase=uint8_t(phase);t.logical_token=100+n;return t;}
static void open(API&a,Fixture&f,int observed=1,int limit=4096,int attempts=512){need(a.open(&f,1,observed,limit,attempts,128),"open");}
static void begin(API&a,Fixture&f,int n=0,int phase=XR_DECODE,int nt=1){auto t=tag(n,phase);need(a.label(&f,&t),"label");need(a.begin(&f,f.gates.data(),37,n==0,nt,1,1,5,10+n,0),"begin");}
static void close_check(API&a,Fixture&f,uint32_t required,uint64_t records,uint64_t decoded){
    xeon_route37_stats s{};need(a.stats(&s),"stats");need(s.records==records&&s.complete_decode_attempts==decoded&&(s.flags&required)==required,"stats values");
    if(!required)need(s.flags==0,"unexpected flags");
    need(a.close(&f),"close");
}
int main(int argc,char **argv)try{
    need(argc==4,"golden|test GOLDEN_FILE EXPECTED_CPU_LIBRARY");Dl_info info{};need(dladdr(reinterpret_cast<void*>(ggml_backend_cpu_init),&info),"provider");
    need(std::filesystem::canonical(info.dli_fname)==std::filesystem::canonical(argv[3]),"CPU provider mismatch");Fixture f;
    const bool golden=std::strcmp(argv[1],"golden")==0;need(golden||std::strcmp(argv[1],"test")==0,"mode");
    FILE*file=std::fopen(argv[2],golden?"wb":"rb");need(file,"golden file");std::vector<std::vector<float>>expected;
    for(int i=0;i<5;++i){f.inputs(i);auto out=f.run();if(golden){need(std::fwrite(out.data(),4,out.size(),file)==out.size(),"write");}else{
        std::vector<float>v(out.size());need(std::fread(v.data(),4,v.size(),file)==v.size(),"read");need(!std::memcmp(v.data(),out.data(),v.size()*4),"default-off output mismatch");expected.push_back(v);}}
    if(!golden)need(std::fgetc(file)==EOF,"trailing golden");
    std::fclose(file);
    if(golden){std::puts("{\"passed\":true,\"scope\":\"synthetic frozen37-node graph\",\"graphs\":5,\"values\":4440}");return 0;}
    API a;const int phases[]={XR_WARMUP,XR_PREFILL,XR_UNKNOWN,XR_DECODE,XR_DECODE};
    for(int observe:{0,1}){
        open(a,f,observe);
        for(int i=0;i<5;++i){f.inputs(i);begin(a,f,i,phases[i]);auto replacement=tag(99);need(!a.label(&f,&replacement),"label changed mid-eval");
            need(!a.close(&f),"closed while eval active");auto out=f.run();need(a.end(&f,GGML_STATUS_SUCCESS),"end");need(a.finish(&f,0,1),"finish");
            need(!std::memcmp(out.data(),expected[i].data(),out.size()*4),"observe output mismatch");
            if(observe)for(int l=0;l<37;++l){xeon_route37_record r{};need(a.read(i*37+l,&r),"record");
                need(r.ordinal==uint64_t(i*37+l)&&r.attempt==uint64_t(i+1)&&r.layer==l&&r.phase==phases[i]&&r.graph_generation==1&&r.position==10+i&&r.decode_call==42u+i&&r.logical_token==100u+i&&r.flags==0,"metadata");
                for(int j=0;j<6;++j)need(r.ids[j]==(l*7+i*11+j*13)%384,"ordered IDs");
                for(auto b:r.reserved)need(b==0,"reserved bytes");}
        }
        close_check(a,f,0,observe?185:0,observe?2:0);
    }
    // Failure fixtures are explicit contract tests, not model runs or timings.
    f.inputs(0);
    for(int which=0;which<7;++which){
        open(a,f,1,which==3?36:4096);begin(a,f);
        for(int l=0;l<(which==0?36:37);++l){
            if(which==2&&l==0){auto alias=*f.gates[0];a.record(&alias,f.ids[0]);continue;}
            if(which==1&&l==0)a.record(f.gates[0],f.ids[0]);
            if(which==4&&l==0){int32_t bad[6]={1,1,3,4,5,-1};auto ids=*f.ids[0];ids.data=bad;a.record(f.gates[0],&ids);continue;}
            a.record(f.gates[l],f.ids[l]);
        }
        const uint32_t want[]={XR_COVERAGE,XR_DUPLICATE,XR_MAPPING,XR_OVERFLOW,XR_IDS,XR_GRAPH_FAILED,XR_UNSYNCHRONIZED};
        const int end=a.end(&f,which==5?GGML_STATUS_FAILED:GGML_STATUS_SUCCESS);
        need(bool(end)==(which==6),"failure end result");need(!a.finish(&f,0,which!=6),"failure complete accepted");
        close_check(a,f,want[which],which==0||which==2||which==3?36:which==1?38:37,0);
    }
    open(a,f);begin(a,f,0,XR_PREFILL,2);f.run();need(a.end(&f,GGML_STATUS_SUCCESS)&&a.finish(&f,0,1),"multi prefill scope");close_check(a,f,0,0,0);
    open(a,f,1,4096,1);begin(a,f);for(int l=0;l<37;++l)a.record(f.gates[l],f.ids[l]);need(a.end(&f,0)&&a.finish(&f,0,1),"first attempt");
    auto t=tag(1);need(a.label(&f,&t),"second label");need(!a.begin(&f,f.gates.data(),37,0,1,1,1,5,11,0),"attempt overflow accepted");need(!a.finish(&f,0,1),"overflow finish");close_check(a,f,XR_OVERFLOW,37,1);
    open(a,f);begin(a,f);
    for(int l=0;l<37;++l)a.record(f.gates[l],f.ids[l]);
    need(!a.end(&f,GGML_STATUS_ABORTED)&&!a.finish(&f,-1,1),"failed attempt accepted");
    t=tag(1);t.logical_token=100;need(a.label(&f,&t),"retry label");need(a.begin(&f,f.gates.data(),37,0,1,1,1,5,10,0),"retry begin");
    int32_t strided[12]={0,999,13,999,26,999,39,999,52,999,65,999};
    for(int l=0;l<37;++l){auto ids=*f.ids[l];if(!l){ids.data=strided;ids.nb[0]=8;}a.record(f.gates[l],&ids);}
    need(a.end(&f,0)&&a.finish(&f,0,1),"retry completion");xeon_route37_attempt first{},second{};xeon_route37_call call{};
    need(a.attempt(0,&first)&&a.attempt(1,&second)&&a.call(0,&call),"attempt/call read");
    need(first.id==1&&second.id==2&&first.position==second.position&&first.label.logical_token==second.label.logical_token&&!first.caller_completed&&second.caller_completed&&call.status==-1,"retry ledger");
    xeon_route37_record stride_record{};need(a.read(37,&stride_record),"stride record");for(int i=0;i<6;++i)need(stride_record.ids[i]==13*i,"stride identity");
    close_check(a,f,XR_GRAPH_FAILED|XR_CALL_FAILED,74,1);
    std::puts("{\"passed\":true,\"scope\":\"synthetic actual-ggml plus contract tests; no model routes captured\",\"graph_executions\":16,\"golden_comparisons\":13320,\"complete_observed_decode_attempts_in_five_graph_sequence\":2,\"nondecode_T1_phases_checked\":3,\"negative_contract_cases\":9,\"failed_retry_ledger_preserved\":true,\"strided_ids_checked\":true,\"cross_thread_execution\":true}");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
