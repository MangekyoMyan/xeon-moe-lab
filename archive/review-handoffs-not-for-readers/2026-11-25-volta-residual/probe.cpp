#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "json.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;
constexpr int K = 5120, M = 2304, Slots = 6, Tokens = 96;
extern "C" cudaError_t residual_input(int, bool, const float *, float *, int, cudaStream_t);
bool ggml_cuda_should_use_mmvq(ggml_type, int, int64_t);
bool ggml_cuda_should_use_mmq(ggml_type, int, int64_t, int64_t);
bool ggml_cuda_should_use_mmf(ggml_type, int, int, const int64_t *, const size_t *, int, bool);

static void require(bool ok, const std::string &what) { if (!ok) throw std::runtime_error(what); }
static void cuda_ok(cudaError_t c) { require(c == cudaSuccess, cudaGetErrorString(c)); }
static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static json read_json(const fs::path &p) { std::ifstream f(p); require(bool(f),p.string()); json j; f>>j; return j; }
static void save(const fs::path &p, const json &j) { std::ofstream f(p); f<<j.dump(2)<<'\n'; f.close(); require(!f.fail(),p.string()); }
template<class T> static std::vector<T> read_binary(const fs::path &p, size_t count) {
    require(fs::file_size(p) == count*sizeof(T),p.string());
    std::vector<T> v(count); std::ifstream f(p,std::ios::binary);
    f.read(reinterpret_cast<char *>(v.data()),count*sizeof(T)); require(bool(f),p.string()); return v;
}
static void provider(const void *symbol,const std::string &wanted) {
    Dl_info info{}; require(dladdr(symbol,&info) && info.dli_fname && fs::canonical(info.dli_fname)==fs::canonical(wanted),"loaded provider "+wanted);
}
static const char *mode_name(int mode) { return std::array<const char *,3>{"MMVQ","MMQ","FP16"}[mode]; }
static int route(int n, ggml_tensor *w) {
    constexpr int cc=700;
    require(!ggml_cuda_should_use_mmf(w->type,cc,32,w->ne,w->nb,n,false),"unexpected floating-point native path");
    if (ggml_cuda_should_use_mmvq(w->type,cc,n)) return 0;
    if (ggml_cuda_should_use_mmq(w->type,cc,n,0)) return 1;
    return 2;
}
static size_t available() { size_t f,t; cuda_ok(cudaMemGetInfo(&f,&t)); return f; }

struct Backend {
    ggml_backend_t gpu=nullptr;
    cudaStream_t prepare=nullptr;
    ggml_backend_event_t ready=nullptr;
    Backend() {
        gpu=ggml_backend_cuda_init(0);require(gpu && ggml_backend_is_cuda(gpu),"CUDA backend");
        cuda_ok(cudaStreamCreateWithFlags(&prepare,cudaStreamNonBlocking));
        ready=ggml_backend_event_new(ggml_backend_get_device(gpu));
    }
    void publish(){
        if(ready) {
            // The pinned CUDA backend stores a cudaEvent_t in this ggml event.
            cuda_ok(cudaEventRecord(static_cast<cudaEvent_t>(ready->context),prepare));
            ggml_backend_event_wait(gpu,ready);
        } else cuda_ok(cudaStreamSynchronize(prepare));
    }
    ~Backend(){
        if(ready)ggml_backend_event_free(ready);
        if(prepare)cudaStreamDestroy(prepare);
        if(gpu)ggml_backend_free(gpu);
    }
};

struct Shape {
    Backend &backend;
    const int n;
    int regular_mode,paired_mode;
    ggml_context *ctx=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_tensor *w=nullptr,*x=nullptr,*r=nullptr,*joined=nullptr,*paired_product=nullptr;
    std::array<ggml_tensor*,3> out{};
    std::array<ggml_cgraph*,3> graph{};

    Shape(Backend &owner,int count): backend(owner),n(count) {
        ctx=ggml_init({1024*1024,nullptr,true});require(ctx,"ggml context");
        w=ggml_new_tensor_2d(ctx,GGML_TYPE_IQ2_S,K,M);
        x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,n);
        r=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,n);
        joined=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,2*n);
        out[0]=ggml_mul_mat(ctx,w,x);
        // The CUDA backend keys its graph cache by the first node. Give each
        // timing arm its own key while reusing exactly the same weights and x.
        auto split_primary=ggml_mul_mat(ctx,w,x);
        auto correction=ggml_mul_mat(ctx,w,r);
        out[1]=ggml_add(ctx,split_primary,ggml_scale(ctx,correction,1.f/128));
        paired_product=ggml_mul_mat(ctx,w,joined);
        auto first=ggml_view_2d(ctx,paired_product,M,n,paired_product->nb[1],0);
        auto second=ggml_view_2d(ctx,paired_product,M,n,paired_product->nb[1],size_t(n)*paired_product->nb[1]);
        out[2]=ggml_add(ctx,first,ggml_scale(ctx,second,1.f/128));
        for(int i=0;i<3;i++){
            graph[i]=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph[i],out[i]);
            require(ggml_backend_supports_op(backend.gpu,out[i]),"unsupported operation");
        }
        require(ggml_graph_node(graph[0],0)!=ggml_graph_node(graph[1],0) &&
                ggml_graph_node(graph[0],0)!=ggml_graph_node(graph[2],0) &&
                ggml_graph_node(graph[1],0)!=ggml_graph_node(graph[2],0),"distinct graph-cache keys");
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend.gpu);require(buffer,"GPU buffer");
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(buffer,0);
        regular_mode=route(n,w);paired_mode=route(2*n,w);
        require(regular_mode==(n<=8?0:n<64?1:2) && paired_mode==(2*n<=8?0:2*n<64?1:2),"installed dispatch predicates");
    }
    ~Shape(){ if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx); }
    void run(int arm) {
        if(arm) {
            bool paired=arm==2;
            cuda_ok(residual_input(paired?paired_mode:regular_mode,paired,static_cast<const float*>(x->data),
                    static_cast<float*>((paired?joined:r)->data),K*n,backend.prepare));
            backend.publish();
        }
        require(ggml_backend_graph_compute(backend.gpu,graph[arm])==GGML_STATUS_SUCCESS,"native graph failed");
    }
    std::vector<float> read(ggml_tensor *t,size_t elements) {
        std::vector<float> v(elements);ggml_backend_tensor_get(t,v.data(),0,v.size()*4);
        require(std::all_of(v.begin(),v.end(),[](float a){return std::isfinite(a);}),"nonfinite score");
        return v;
    }
};

struct Ref { int row; double full,observed; };
struct Fixture {
    std::string id,projection;
    std::vector<float> x,old;
    std::vector<int> ids;
    std::map<std::pair<int,int>,std::vector<Ref>> refs;
};
struct Job {int fixture,expert;std::vector<int> positions;};

static Fixture fixture(const fs::path &root,const std::string &name,const std::string &role) {
    const auto d=root/(name+"-early");const auto key="ffn_moe_"+role+"-0";
    const auto capture=read_json(d/"CAPTURE.json");
    require(capture.at("prefix_logits_byte_equal_to_no_callback"),"original prefix validation");
    Fixture f{name,role,read_binary<float>(d/(key+"-input.bin"),K*Tokens),read_binary<float>(d/(key+".bin"),M*Slots*Tokens),
              read_binary<int>(d/(key+"-ids.bin"),Slots*Tokens),{}};
    for(const auto &s:capture.at("F64_samples")){
        if(s[0]!=key)continue;
        int t=s[1],slot=s[2],e=s[3],row=s[4];
        require(f.ids.at(t*Slots+slot)==e && row>=0 && row<M,"reference coordinates");
        f.refs[{t,slot}].push_back({row,s[5],s[7]});
    }
    require(f.refs.size()==36,"sample coverage");
    return f;
}

int main(int argc,char **argv) try {
    require(argc==4,"usage: probe design.json new-output-directory forward|reverse");
    const std::string visit_order=argv[3];
    require(visit_order=="forward" || visit_order=="reverse","job order");
    const auto spec=read_json(argv[1]);const fs::path dest=argv[2];
    require(fs::is_directory(dest) && !fs::exists(dest/"RUN.json") && !fs::exists(dest/"groups.tsv"),"fresh output directory");
    provider(reinterpret_cast<void*>(ggml_backend_cuda_init),spec["cuda_library"]);
    provider(reinterpret_cast<void*>(ggml_row_size),spec["base_library"]);
    provider(reinterpret_cast<void*>(ggml_cpu_init),spec["cpu_library"]);
    ggml_cpu_init();
    cuda_ok(cudaSetDevice(0));cudaDeviceProp device{};cuda_ok(cudaGetDeviceProperties(&device,0));
    require(device.major==7 && device.minor==0 && device.warpSize==32,"Volta SM70 required");
    const size_t start_free=available(), reserve=1024ULL<<20;
    require(start_free>reserve+(150ULL<<20),"insufficient room for a small operator probe");
    std::vector<Fixture> data;std::vector<Job> jobs;
    for(auto name:{"JA96","PY96"}) for(auto role:{"gate","up"}){
        auto f=fixture(spec.at("capture_root").get<std::string>(),name,role);
        std::map<int,std::vector<int>> all;std::set<int> selected;
        for(int t=0;t<Tokens;t++){
            std::set<int> unique;
            for(int slot=0;slot<Slots;slot++){
                int e=f.ids[t*Slots+slot];require(e>=0&&e<384&&unique.insert(e).second,"genuine distinct routing IDs");
                all[e].push_back(t*Slots+slot);
                if(f.refs.count({t,slot}))selected.insert(e);
            }
        }
        for(int e:selected)jobs.push_back({int(data.size()),e,all[e]});
        data.push_back(std::move(f));
    }
    std::stable_sort(jobs.begin(),jobs.end(),[](const auto&a,const auto&b){return a.positions.size()<b.positions.size();});
    if(visit_order=="reverse")std::reverse(jobs.begin(),jobs.end());
    const std::string shard=spec.at("shard");std::ifstream weights(shard,std::ios::binary);require(bool(weights),"shard open");
    auto meta=gguf_init_from_file(shard.c_str(),{true,nullptr});require(meta,"shard metadata");
    std::map<std::string,size_t> offsets;
    for(auto role:{"gate","up"}){
        const auto name=std::string("blk.0.ffn_")+role+"_exps.weight";auto i=gguf_find_tensor(meta,name.c_str());
        require(i>=0 && gguf_get_tensor_type(meta,i)==GGML_TYPE_IQ2_S,"expected IQ2_S tensor");
        auto dims=gguf_get_tensor_ne(meta,i);
        require(dims[0]==K && dims[1]==M && dims[2]==384 && dims[3]==1,"expert geometry");
        offsets[role]=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,i);
    }
    gguf_free(meta);
    auto dequant=ggml_get_type_traits(GGML_TYPE_IQ2_S)->to_float;require(dequant,"IQ2_S dequant");
    const size_t rowbytes=ggml_row_size(GGML_TYPE_IQ2_S,K),expertbytes=M*rowbytes;
    std::vector<uint8_t> packed(expertbytes);std::vector<float> row(K);
    std::ofstream timings(dest/"timings.tsv"),dots(dest/"dots.tsv"),groups(dest/"groups.tsv");
    std::ofstream steady(dest/"steady.tsv");
    require(bool(timings)&&bool(dots)&&bool(groups)&&bool(steady),"evidence output");
    timings<<"job\trepetition\torder\tarm\tns\n";
    steady<<"job\tarm\trepetition\tns\n";
    groups<<"job\tfixture\tprojection\texpert\tinputs\tmode\tpaired_mode\tcompared\tbase_bit_mismatches\tmax_base_diff\tpaired_first_bit_mismatches\n";
    dots<<"job\tfixture\tprojection\ttoken\tslot\texpert\trow\tinputs\tmode\tpaired_mode\tfull\tarchived\tbase\tsplit\tpaired\tpaired_first\tweight_half_ideal\n"<<std::setprecision(17);
    groups<<std::setprecision(12);
    json report={{"passed",false},{"start_free_bytes_after_cuda_init",start_free},{"selected_jobs",jobs.size()},{"device",device.name},
                 {"mode_counts_rechecked",true},{"GPU_input_residual_scale",128},{"GGUF_expert_bytes",expertbytes}};
    report["visit_order"]=visit_order;report["distinct_graph_keys_checked"]=true;
    size_t minimum=start_free, mismatches=0, compared=0, sample_count=0;
    int permutations[6][3]={{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
    {
        Backend gpu;report["preparation_join"]=gpu.ready?"CUDA event into ggml stream":"preparation stream synchronization";
        std::unique_ptr<Shape> op;
        for(size_t ji=0;ji<jobs.size();ji++){
            const auto &job=jobs[ji];const auto &f=data[job.fixture];const int n=job.positions.size();
            if(!op || op->n!=n){op.reset();op=std::make_unique<Shape>(gpu,n);}
            minimum=std::min(minimum,available());require(minimum>=reserve,"insufficient remaining VRAM");
            weights.seekg(offsets.at(f.projection)+size_t(job.expert)*expertbytes);
            weights.read(reinterpret_cast<char*>(packed.data()),expertbytes);require(bool(weights),"expert read failed");
            std::vector<float> x(size_t(n)*K);
            for(int j=0;j<n;j++)std::memcpy(x.data()+size_t(j)*K,f.x.data()+size_t(job.positions[j]/Slots)*K,K*4);
            ggml_backend_tensor_set(op->w,packed.data(),0,packed.size());
            ggml_backend_tensor_set(op->x,x.data(),0,x.size()*4);
            // All allocations, uploads, warming and readbacks stay outside measured intervals.
            for(int rep=0;rep<2;rep++)for(int arm=0;arm<3;arm++)op->run(arm);
            std::array<std::vector<float>,3> y;
            for(int arm=0;arm<3;arm++){op->run(arm);y[arm]=op->read(op->out[arm],size_t(n)*M);}
            auto first=op->read(op->paired_product,size_t(n)*M);
            int diffs=0,pdiffs=0;double maxerr=0;
            for(int j=0;j<n;j++){
                int pos=job.positions[j],t=pos/Slots,slot=pos%Slots;
                const auto *old=f.old.data()+size_t(pos)*M;
                const auto *v=y[0].data()+size_t(j)*M;
                for(int m=0;m<M;m++){
                    diffs+=std::memcmp(v+m,old+m,4)!=0;
                    pdiffs+=std::memcmp(v+m,&first[size_t(j)*M+m],4)!=0;
                    maxerr=std::max(maxerr,std::abs(double(v[m])-old[m]));
                }
                auto range=f.refs.find({t,slot});if(range==f.refs.end())continue;
                for(auto &ref:range->second){
                    const int m=ref.row;dequant(packed.data()+size_t(m)*rowbytes,row.data(),K);
                    double full=0,wh=0;
                    for(int k=0;k<K;k++){
                        float a=x[size_t(j)*K+k];full+=double(row[k])*a;
                        if(op->regular_mode==2) wh+=double(ggml_fp16_to_fp32(ggml_fp32_to_fp16(row[k])))*a;
                    }
                    require(std::abs(full-ref.full)<1.e-8 && float(ref.observed)==old[m],"archived reference cross-check failed");
                    size_t at=size_t(j)*M+m;
                    dots<<ji<<'\t'<<f.id<<'\t'<<f.projection<<'\t'<<t<<'\t'<<slot<<'\t'<<job.expert<<'\t'<<m<<'\t'<<n<<'\t'
                        <<mode_name(op->regular_mode)<<'\t'<<mode_name(op->paired_mode)<<'\t'<<full<<'\t'<<old[m]<<'\t'<<v[m]<<'\t'
                        <<y[1][at]<<'\t'<<y[2][at]<<'\t'<<first[at]<<'\t';
                    if(op->regular_mode==2)dots<<wh;
                    dots<<'\n';++sample_count;
                }
            }
            groups<<ji<<'\t'<<f.id<<'\t'<<f.projection<<'\t'<<job.expert<<'\t'<<n<<'\t'<<mode_name(op->regular_mode)<<'\t'
                  <<mode_name(op->paired_mode)<<'\t'<<size_t(n)*M<<'\t'<<diffs<<'\t'<<maxerr<<'\t'<<pdiffs<<'\n';
            mismatches+=diffs;compared+=size_t(n)*M;
            for(int rep=0;rep<6;rep++){
                for(int order=0;order<3;order++){
                    const int arm=permutations[(rep+ji)%6][order];
                    const auto a=now_ns();op->run(arm);const auto b=now_ns();
                    timings<<ji<<'\t'<<rep<<'\t'<<order<<'\t'<<arm<<'\t'<<(b-a)<<'\n';
                }
            }
            for(int order=0;order<3;order++){
                const int arm=(order+int(ji))%3;
                op->run(arm);
                for(int rep=0;rep<5;rep++){
                    const auto a=now_ns();op->run(arm);const auto b=now_ns();
                    steady<<ji<<'\t'<<arm<<'\t'<<rep<<'\t'<<(b-a)<<'\n';
                }
            }
            for(int arm=0;arm<3;arm++){
                op->run(arm);auto again=op->read(op->out[arm],size_t(n)*M);
                require(!std::memcmp(again.data(),y[arm].data(),again.size()*4),"operator output not repeatable");
            }
            minimum=std::min(minimum,available());require(minimum>=reserve,"VRAM reserve during replay");
        }
    }
    timings.close();dots.close();groups.close();steady.close();require(!timings.fail()&&!dots.fail()&&!groups.fail()&&!steady.fail(),"save outputs");
    require(sample_count==4608 && (int)jobs.size()==114,"complete predeclared sample");
    report.update({{"passed",true},{"full_outputs_compared_to_prior",compared},{"baseline_bit_mismatches",mismatches},
                   {"sample_count",sample_count},{"timed_intervals",jobs.size()*18},{"steady_intervals",jobs.size()*15},{"all_three_arms_repeated_bitwise",true},
                   {"minimum_cuda_free_bytes_sampled",minimum},{"end_free_bytes",available()}});
    save(dest/"RUN.json",report);
    std::cout<<report.dump()<<'\n';return 0;
} catch(const std::exception &e){std::cerr<<"probe failed: "<<e.what()<<'\n';return 1;}
