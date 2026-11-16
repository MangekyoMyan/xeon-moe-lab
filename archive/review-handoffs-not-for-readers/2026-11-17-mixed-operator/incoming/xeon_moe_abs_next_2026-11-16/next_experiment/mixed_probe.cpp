// Review candidate for one isolated IQ2_S up operator; not a production switch.
// Build with the pinned ggml headers and generated mixed_case_data.h.
// XEON_MIXED_SPEC_ONLY excludes all ggml/model code and tests only the protocol.
#include "mixed_case_data.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

namespace spec = xeon_mixed_case;
static constexpr int THREADS=12, BLOCKS=16, REPEATS=32;
static constexpr size_t PER_EXPERT=6635520, EXPECTED_VALUES=13824;
static constexpr const char * CASES[3]={"C0","C6","Cplan"};
static constexpr char LABELS[8]={'F','E','E','F','E','F','F','E'};
static void need(bool ok,const char * message) { if(!ok) throw std::runtime_error(message); }
static std::string quote(const std::string &s) {
    std::ostringstream out; out << '"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\') out << '\\' << char(c);
        else if(c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << char(c);
    }
    return out.str()+'"';
}
template<class Range> static void array_json(std::ostream &s,const Range &r) {
    s << '['; bool first=true; for(const auto &x:r) { if(!first) s << ','; first=false; s << x; } s << ']';
}
[[maybe_unused]] static int integer(const char *arg) {
    size_t n=0; const std::string s(arg); int x=std::stoi(s,&n);
    need(n==s.size(),"invalid integer argument"); return x;
}
static int rotation(int process) { need(process>=0&&process<8,"process index must be 0..7"); return (process/2)%3; }
static int case_index(int process,int order) { return (order+rotation(process))%3; }
static char logical_mode(int block,int arm) {
    const bool reverse=((block%8)+(block/8))%2;
    const bool middle=(arm==1||arm==2);
    return middle!=reverse?'S':'O';
}
[[maybe_unused]] static std::vector<uint32_t> selected(int c) {
    if(c==0) return {};
    if(c==1) return {std::begin(spec::routes),std::end(spec::routes)};
    need(c==2,"invalid case index"); return {std::begin(spec::cache_ids),std::end(spec::cache_ids)};
}
[[maybe_unused]] static int mask(int c,int group) { return c==0?0:c==1?63:spec::masks[group]; }
static int popcount(int v) { int n=0; for(;v;v&=v-1)++n; return n; }
static void verify_protocol() {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE binary32 required");
    const uint16_t one=1; need(*reinterpret_cast<const unsigned char*>(&one)==1,"little endian required");
    need(spec::K==5120&&spec::M==2304&&spec::EXPERTS==384&&spec::USED==6&&spec::GROUPS==8&&spec::INPUTS==1,"unexpected geometry");
    need(std::size(spec::routes)==48&&std::size(spec::cache_ids)==39&&std::size(spec::masks)==8,"unexpected fixture sizes");
    std::vector<int> r(std::begin(spec::routes),std::end(spec::routes));
    std::sort(r.begin(),r.end()); need(r.front()>=0&&r.back()<384&&std::adjacent_find(r.begin(),r.end())==r.end(),"routes must be 48 distinct expert IDs");
    std::vector<int> s(std::begin(spec::cache_ids),std::end(spec::cache_ids));
    need(std::is_sorted(s.begin(),s.end())&&std::adjacent_find(s.begin(),s.end())==s.end(),"Cplan IDs must be sorted unique");
    for(int id:s) need(std::binary_search(r.begin(),r.end(),id),"Cplan ID outside fixture routes");
    const int expected[8]={5,5,5,5,3,5,6,5};
    for(int g=0;g<8;++g) {
        int m=0; for(int j=0;j<6;++j) if(std::binary_search(s.begin(),s.end(),spec::routes[g*6+j])) m|=1<<j;
        need(m==spec::masks[g]&&popcount(m)==expected[g],"Cplan membership mismatch");
    }
    for(int p=0;p<8;++p) for(int g=0;g<8;++g) for(int a=0;a<4;++a)
        need(logical_mode(g,a)!=logical_mode(g+8,a),"each route must reverse its arm order");
}
static uint64_t monotonic_ns() {
    timespec t{}; need(clock_gettime(CLOCK_MONOTONIC,&t)==0,"CLOCK_MONOTONIC failed");
    return uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec);
}

#ifdef XEON_MIXED_SPEC_ONLY
int main() try {
    verify_protocol();
    need(monotonic_ns()>0,"clock");
    std::cout << "{\"kind\":\"protocol_only\",\"passed\":true,\"model_executed\":false,\"ggml_linked\":false,\"processes\":8,\"chunks\":1536,\"timed_graph_calls_if_run\":49152,\"case_orders\":[";
    for(int p=0;p<8;++p) { if(p) std::cout<<','; std::cout<<'['; for(int i=0;i<3;++i) { if(i)std::cout<<',';std::cout<<quote(CASES[case_index(p,i)]); } std::cout<<']'; }
    std::cout << "],\"Cplan_masks\":"; array_json(std::cout,spec::masks); std::cout << "}\n"; return 0;
} catch(const std::exception&e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
#else
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "iq3-cache.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>

static void (*set_mode)(int)=nullptr;
static void (*diagnostics)(int)=nullptr;
static void (*counts)(uint64_t*)=nullptr;
static void mode(int m) { if(set_mode)set_mode(m);else need(m==0,"shared mode missing"); }
static std::string provider(void *symbol) {
    Dl_info info{}; need(symbol&&dladdr(symbol,&info)&&info.dli_fname,"symbol provider unavailable");
    return std::filesystem::canonical(info.dli_fname).string();
}
struct Sample {
    int block,group,arm,actual,hit_mask; char logical;
    uint64_t begin,end; int mismatches;
};
struct CaseRecord {
    int index=0; size_t bytes=0; std::array<int,12> pins{};
    bool warmed=false; std::vector<Sample> samples;
};
struct Report {
    int process=-1; std::filesystem::path directory; std::string metadata;
    std::vector<CaseRecord> cases; bool passed=false; std::string error;
    void save() const {
        if(directory.empty())return;
        const auto path=directory/"raw.jsonl";
        need(!std::filesystem::exists(path),"raw.jsonl already exists");
        std::ofstream out(path); need(bool(out),"cannot create raw.jsonl"); out<<std::setprecision(17);
        if(!metadata.empty())out<<metadata<<'\n';
        size_t n=0,output_bytes=0;
        for(const auto &c:cases) {
            if(c.warmed) {
                out<<"{\"kind\":\"mixed_case\",\"case\":"<<quote(CASES[c.index])<<",\"cache_bytes\":"<<c.bytes<<",\"selected_ids\":";
                array_json(out,selected(c.index)); out<<",\"masks\":[";
                for(int g=0;g<8;++g){if(g)out<<',';out<<mask(c.index,g);}
                out<<"],\"pins\":";array_json(out,c.pins);
                out<<",\"expected_file\":"<<quote(std::string("expected-")+CASES[c.index]+".bin")
                   <<",\"expected_groups\":8,\"expected_values_per_group\":13824,\"warmup_arm_calls_per_group\":4,\"bit_mismatches\":0,\"dispatch_checked\":"<<(LABELS[process]=='E'?"true":"false")<<"}\n";
                output_bytes+=8*EXPECTED_VALUES*sizeof(float);
            }
            for(const auto &s:c.samples) {
                out<<"{\"kind\":\"mixed_chunk\",\"case\":"<<quote(CASES[c.index])
                   <<",\"block\":"<<s.block<<",\"route_group\":"<<s.group<<",\"arm\":"<<s.arm
                   <<",\"logical_mode\":"<<quote(std::string(1,s.logical))<<",\"actual_mode\":"<<s.actual
                   <<",\"cached_of_six\":"<<popcount(s.hit_mask)<<",\"hit_mask\":"<<s.hit_mask<<",\"ids\":[";
                for(int j=0;j<6;++j){if(j)out<<',';out<<spec::routes[s.group*6+j];}
                out<<"],\"begin_ns\":"<<s.begin<<",\"end_ns\":"<<s.end<<",\"elapsed_ns\":"<<(s.end-s.begin)
                   <<",\"repeats\":32,\"ms_per_graph\":"<<double(s.end-s.begin)/REPEATS/1e6
                   <<",\"bit_mismatches\":"<<s.mismatches<<"}\n"; ++n;
            }
        }
        out<<"{\"kind\":\"mixed_complete\",\"passed\":"<<(passed?"true":"false")<<",\"chunks\":"<<n
           <<",\"timed_graph_calls\":"<<n*REPEATS<<",\"outputs_bytes\":"<<output_bytes
           <<",\"expected_output_scope\":\"mode0 once per case/group; final output of each timed chunk compared in memory\",\"error\":"<<(error.empty()?"null":quote(error))<<"}\n";
        out.close(); need(!out.fail(),"raw.jsonl write failed");
    }
};
struct Resources {
    ggml_backend_t cpu=nullptr; gguf_context *meta=nullptr; ggml_context *ctx=nullptr;
    ggml_backend_buffer_t weights=nullptr,activations=nullptr; ggml_threadpool_t pool=nullptr;
    void *mapping=MAP_FAILED; size_t mapped_size=0;
    ~Resources() {
        if(cpu)ggml_backend_cpu_set_threadpool(cpu,nullptr);
        if(pool)ggml_threadpool_free(pool);
        if(activations)ggml_backend_buffer_free(activations);
        if(weights)ggml_backend_buffer_free(weights);
        if(ctx)ggml_free(ctx);
        if(mapping!=MAP_FAILED)munmap(mapping,mapped_size);
        if(meta)gguf_free(meta);
        if(cpu)ggml_backend_free(cpu);
    }
};
struct CacheOwner {
    void *value=nullptr;
    ~CacheOwner(){if(value)xeon_iq3_cache_free_v1(value);}
};
static void write_expected(const std::filesystem::path &path,const std::array<std::vector<float>,8> &expected) {
    need(!std::filesystem::exists(path),"expected output file already exists");
    std::ofstream file(path,std::ios::binary);need(bool(file),"expected output create failed");
    for(const auto&v:expected)file.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));
    file.close();need(!file.fail(),"expected output write failed");
}
static void run(Report &report,int argc,char **argv) {
    need(argc==8&&std::string(argv[1])=="bench","usage: bench SHARD INPUTS_BIN OUTPUT_DIR PROCESS_INDEX EXPECTED_CPU_LIBRARY EXPECTED_BASE_LIBRARY");
    verify_protocol(); report.process=integer(argv[5]); rotation(report.process);
    report.directory=std::filesystem::canonical(argv[4]);need(std::filesystem::is_directory(report.directory),"output directory required");
    need(!std::filesystem::exists(report.directory/"raw.jsonl"),"existing raw.jsonl");
    for(const char *name:CASES)need(!std::filesystem::exists(report.directory/(std::string("expected-")+name+".bin")),"existing expected output");
    const bool experimental=LABELS[report.process]=='E';
    const auto cpu_path=provider(reinterpret_cast<void*>(ggml_backend_cpu_init));
    const auto base_path=provider(reinterpret_cast<void*>(ggml_init));
    const auto cache_path=provider(reinterpret_cast<void*>(xeon_iq3_cache_add_selected_v2));
    need(cpu_path==std::filesystem::canonical(argv[6]).string(),"wrong CPU library provider");
    need(base_path==std::filesystem::canonical(argv[7]).string(),"wrong base library provider");
    need(cache_path==cpu_path,"cache API came from another library");
    set_mode=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"xeon_iq_abs_set_mode"));
    diagnostics=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"xeon_iq_abs_set_diagnostics"));
    counts=reinterpret_cast<void(*)(uint64_t*)>(dlsym(RTLD_DEFAULT,"xeon_iq_abs_counts"));
    need(experimental?(set_mode&&diagnostics&&counts):(!set_mode&&!diagnostics&&!counts),"wrong frozen/experimental API identity");
    const std::string mode_path=experimental?provider(reinterpret_cast<void*>(set_mode)):"";
    if(experimental) {
        need(mode_path==cpu_path&&provider(reinterpret_cast<void*>(diagnostics))==cpu_path&&provider(reinterpret_cast<void*>(counts))==cpu_path,"abs API provider mismatch");
        diagnostics(0);mode(0);
    }
    need(xeon_iq3_cache_total_v1()==0,"cache not initially empty");
    std::vector<float> inputs(size_t(spec::GROUPS)*spec::K);
    std::ifstream input_file(argv[3],std::ios::binary);need(bool(input_file),"input open failed");
    input_file.read(reinterpret_cast<char*>(inputs.data()),inputs.size()*sizeof(float));
    need(input_file.gcount()==std::streamsize(inputs.size()*sizeof(float))&&input_file.peek()==std::char_traits<char>::eof(),"input byte length mismatch");
    need(std::all_of(inputs.begin(),inputs.end(),[](float v){return std::isfinite(v);}),"nonfinite input fixture");
    Resources r; r.cpu=ggml_backend_cpu_init();need(r.cpu,"CPU backend init failed");
    need(!ggml_is_numa(),"unexpected ggml NUMA mode");
    std::ostringstream meta;meta<<"{\"kind\":\"mixed_meta\",\"schema\":1,\"process_index\":"<<report.process
       <<",\"library_label\":"<<quote(std::string(1,LABELS[report.process]))<<",\"commit\":"<<quote(spec::commit)
       <<",\"tensor\":"<<quote(spec::tensor_name)<<",\"threads\":12,\"blocks\":16,\"repeats\":32,\"routes\":";
    array_json(meta,spec::routes);meta<<",\"case_order\":[";
    for(int i=0;i<3;++i){if(i)meta<<',';meta<<quote(CASES[case_index(report.process,i)]);}
    meta<<"],\"inputs_sha256\":"<<quote(spec::inputs_sha256)<<",\"plan_sha256\":"<<quote(spec::plan_sha256)
        <<",\"hashes_verified_by_harness\":false,\"cpu_provider\":"<<quote(cpu_path)<<",\"base_provider\":"<<quote(base_path)
        <<",\"cache_provider\":"<<quote(cache_path)<<",\"mode_provider\":"<<quote(mode_path)
        <<",\"ggml_numa\":false,\"clock\":\"CLOCK_MONOTONIC\",\"shard\":"<<quote(std::filesystem::canonical(argv[2]).string())<<'}';
    report.metadata=meta.str();report.cases.reserve(3);
    // Preserve the provider map for the launcher's post-run library identity checks.
    { std::ifstream maps("/proc/self/maps");std::ofstream copy(report.directory/"maps.txt");need(bool(maps)&&bool(copy),"maps receipt failed");copy<<maps.rdbuf();need(bool(copy),"maps write failed"); }
    r.meta=gguf_init_from_file(argv[2],{true,nullptr});need(r.meta,"GGUF metadata failed");
    const auto ti=gguf_find_tensor(r.meta,spec::tensor_name);need(ti>=0,"required tensor missing");
    const auto ne=gguf_get_tensor_ne(r.meta,ti);const auto type=gguf_get_tensor_type(r.meta,ti);
    need(type==GGML_TYPE_IQ2_S&&ne[0]==spec::K&&ne[1]==spec::M&&ne[2]==spec::EXPERTS&&ne[3]==1,"only pinned IQ2_S up geometry is supported");
    const int fd=open(argv[2],O_RDONLY|O_CLOEXEC);need(fd>=0,"shard open failed");
    struct stat st{};if(fstat(fd,&st)!=0){close(fd);throw std::runtime_error("shard stat failed");}
    need(st.st_size>0,"empty shard");r.mapped_size=size_t(st.st_size);
    const size_t data_offset=gguf_get_data_offset(r.meta),tensor_offset=gguf_get_tensor_offset(r.meta,ti);
    need(data_offset<=r.mapped_size&&tensor_offset<=r.mapped_size-data_offset,"tensor offset out of bounds");
    const size_t offset=data_offset+tensor_offset,bytes=ggml_row_size(type,spec::K)*spec::M*spec::EXPERTS;
    need(bytes<=r.mapped_size-offset,"weight tensor out of bounds");
    r.mapping=mmap(nullptr,r.mapped_size,PROT_READ,MAP_SHARED,fd,0);close(fd);need(r.mapping!=MAP_FAILED,"mmap failed");
    r.ctx=ggml_init({8*1024*1024,nullptr,true});need(r.ctx,"ggml context failed");
    auto w=ggml_new_tensor_3d(r.ctx,type,spec::K,spec::M,spec::EXPERTS);ggml_set_name(w,spec::tensor_name);
    r.weights=ggml_backend_cpu_buffer_from_ptr(r.mapping,r.mapped_size);need(r.weights,"weight buffer failed");
    w->buffer=r.weights;w->data=static_cast<char*>(r.mapping)+offset;ggml_backend_buffer_set_usage(r.weights,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto x=ggml_new_tensor_3d(r.ctx,GGML_TYPE_F32,spec::K,1,1);
    auto ids=ggml_new_tensor_2d(r.ctx,GGML_TYPE_I32,6,1);
    auto y=ggml_mul_mat_id(r.ctx,w,x,ids);
    need(size_t(ggml_nelements(y))==EXPECTED_VALUES,"unexpected output shape");
    auto graph=ggml_new_graph_custom(r.ctx,32,false);ggml_build_forward_expand(graph,y);
    r.activations=ggml_backend_alloc_ctx_tensors(r.ctx,r.cpu);need(r.activations,"activation allocation failed");
    auto pool_params=ggml_threadpool_params_default(THREADS);pool_params.poll=0;pool_params.strict_cpu=true;
    for(int i=0;i<THREADS;++i)pool_params.cpumask[i]=true;
    r.pool=ggml_threadpool_new(&pool_params);need(r.pool,"threadpool failed");
    ggml_backend_cpu_set_n_threads(r.cpu,THREADS);ggml_backend_cpu_set_threadpool(r.cpu,r.pool);ggml_backend_cpu_set_use_ref(r.cpu,false);
    auto compute=[&](){need(ggml_backend_graph_compute(r.cpu,graph)==GGML_STATUS_SUCCESS,"operator compute failed");};
    auto set_group=[&](int group){
        int32_t route[6];for(int j=0;j<6;++j)route[j]=spec::routes[group*6+j];
        ggml_backend_tensor_set(ids,route,0,sizeof(route));
        ggml_backend_tensor_set(x,inputs.data()+size_t(group)*spec::K,0,size_t(spec::K)*sizeof(float));
    };
    std::array<std::vector<float>,8> expected;for(auto &v:expected)v.resize(EXPECTED_VALUES);
    std::vector<float> output(EXPECTED_VALUES);
    for(int order=0;order<3;++order) {
        need(xeon_iq3_cache_total_v1()==0,"previous cache not released");
        const int ci=case_index(report.process,order);const auto selection=selected(ci);
        report.cases.emplace_back();auto &record=report.cases.back();record.index=ci;record.pins.fill(-1);record.samples.reserve(BLOCKS*4);
        CacheOwner owner;owner.value=xeon_iq3_cache_create_v1(304ULL*1024*1024,64ULL*1024*1024*1024,THREADS);need(owner.value,"cache owner failed");
        need(xeon_iq3_cache_add_selected_v2(owner.value,w,selection.data(),selection.size())==(selection.empty()?0:1),"selected cache build failed");
        record.bytes=xeon_iq3_cache_used_v1(owner.value);need(record.bytes==selection.size()*PER_EXPERT,"cache payload differs from specification");
        const void *const *lookup=xeon_iq3_cache_lookup_experts_v2(w);
        for(int e=0;e<384;++e)need(bool(lookup&&lookup[e])==(std::find(selection.begin(),selection.end(),uint32_t(e))!=selection.end()),"cache membership mismatch");
        for(int group=0;group<8;++group) {
            set_group(group);mode(0);if(diagnostics)diagnostics(0);compute();
            ggml_backend_tensor_get(y,expected[group].data(),0,EXPECTED_VALUES*sizeof(float));
            need(std::all_of(expected[group].begin(),expected[group].end(),[](float v){return std::isfinite(v);}),"nonfinite mode0 expected output");
            for(int arm=0;arm<4;++arm) {
                const int actual=experimental&&logical_mode(group,arm)=='S'?1:0;
                mode(actual);uint64_t before[3]{},after[3]{};
                if(diagnostics){diagnostics(1);counts(before);}compute();if(counts)counts(after);
                ggml_backend_tensor_get(y,output.data(),0,EXPECTED_VALUES*sizeof(float));
                need(std::memcmp(output.data(),expected[group].data(),EXPECTED_VALUES*sizeof(float))==0,"warmup output mismatch");
                if(counts)need(after[1]-before[1]==uint64_t(actual?popcount(mask(ci,group))*spec::M:0),"unexpected warmup abs rows");
            }
            if(diagnostics)diagnostics(0);
        }
        int actual_team=0;
        #pragma omp parallel num_threads(12)
        {
            #pragma omp single
            actual_team=omp_get_num_threads();
            record.pins[omp_get_thread_num()]=sched_getcpu();
        }
        need(actual_team==THREADS,"unexpected actual OpenMP team size");
        auto sorted=record.pins;std::sort(sorted.begin(),sorted.end());
        for(int i=0;i<THREADS;++i)need(sorted[i]==i,"worker CPU sample differs from required 0..11");
        write_expected(report.directory/(std::string("expected-")+CASES[ci]+".bin"),expected);record.warmed=true;
        // No JSON formatting, printing, allocation of records, or file I/O in this loop.
        for(int block=0;block<BLOCKS;++block) {
            const int group=block%8;set_group(group);
            for(int arm=0;arm<4;++arm) {
                const char logical=logical_mode(block,arm);const int actual=experimental&&logical=='S'?1:0;
                // Setter is used only after the previous synchronous graph call returned.
                mode(actual);compute();
                const uint64_t begin=monotonic_ns();for(int n=0;n<REPEATS;++n)compute();const uint64_t end=monotonic_ns();
                need(end>begin,"nonpositive wall duration");
                ggml_backend_tensor_get(y,output.data(),0,EXPECTED_VALUES*sizeof(float));
                const int mismatch=std::memcmp(output.data(),expected[group].data(),EXPECTED_VALUES*sizeof(float))!=0;
                record.samples.push_back({block,group,arm,actual,mask(ci,group),logical,begin,end,mismatch});
                need(!mismatch,"timed chunk final output mismatch");
            }
        }
    }
    need(xeon_iq3_cache_total_v1()==0,"final cache not released");
    mode(0);if(diagnostics)diagnostics(0);report.passed=true;
}
int main(int argc,char **argv) {
    Report report;
    try {run(report,argc,argv);report.save();std::puts("{\"passed\":true,\"scope\":\"isolated IQ2 up operator\"}");return 0;}
    catch(const std::exception&e) {
        report.error=e.what();report.passed=false;
        try {report.save();}catch(const std::exception&save_error){std::fprintf(stderr,"receipt save failed: %s\n",save_error.what());}
        std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;
    }
}
#endif
