// Read-only real GGUF weights + previously captured activation; CPU FFN only.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "iq3-cache.h"
#include "json.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sched.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
using json=nlohmann::ordered_json;
namespace fs=std::filesystem;
static void need(bool b,const std::string &s){if(!b)throw std::runtime_error(s);}
static int64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static json readj(const fs::path &p){std::ifstream f(p);need(bool(f),"read json");return json::parse(f);}
static uint64_t fnv(const void *p,size_t n){auto x=static_cast<const uint8_t*>(p);uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<n;++i)h=(h^x[i])*UINT64_C(1099511628211);return h;}
static std::vector<char> readbytes(const fs::path&p){std::ifstream f(p,std::ios::binary);need(bool(f),"read bytes "+p.string());return {std::istreambuf_iterator<char>(f),{}};}
static std::string provider(void *f){Dl_info d{};need(f&&dladdr(f,&d),"provider");return fs::canonical(d.dli_fname);}
struct Shard{
    std::string path;gguf_context *meta=nullptr;void *data=MAP_FAILED;size_t bytes=0;ggml_backend_buffer_t buffer=nullptr;
    explicit Shard(const std::string &p):path(p){meta=gguf_init_from_file(p.c_str(),{true,nullptr});need(meta,"GGUF metadata");}
    void map(){if(data!=MAP_FAILED)return;int fd=open(path.c_str(),O_RDONLY);need(fd>=0,"GGUF open");struct stat st{};need(fstat(fd,&st)==0,"GGUF stat");bytes=st.st_size;data=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);need(data!=MAP_FAILED,"GGUF mmap");buffer=ggml_backend_cpu_buffer_from_ptr(data,bytes);need(buffer,"CPU mmap buffer");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);}
    ~Shard(){if(buffer)ggml_backend_buffer_free(buffer);if(data!=MAP_FAILED)munmap(data,bytes);if(meta)gguf_free(meta);}
};
struct Fixture{
    ggml_context *wc=nullptr,*ctx=nullptr;ggml_backend_t cpu=nullptr;ggml_backend_buffer_t actbuf=nullptr;void *cache=nullptr;
    ggml_tensor *x=nullptr,*ids=nullptr,*y=nullptr;ggml_cgraph *graph=nullptr;std::vector<float> actual;std::vector<char> captured;json record;
    Fixture(ggml_backend_t b,std::vector<std::unique_ptr<Shard>>&shards,const json &cell,const fs::path &capture,const json &cpuplan):cpu(b){
        wc=ggml_init({512*1024,nullptr,true});ctx=ggml_init({512*1024,nullptr,true});need(wc&&ctx,"contexts");
        cache=xeon_iq3_cache_create_v1(size_t(512)<<20,size_t(64)<<30,12);need(cache,"cache owner");
        std::array<ggml_tensor*,3>w{};record["weights"]=json::array();
        auto routes=cell["ids"].get<std::vector<int32_t>>();need(routes.size()==6,"top6");
        for(int i=0;i<3;++i){
            const auto &pj=cell["projections"][i];const std::string name=pj["name"];bool found=false;
            for(auto &sp:shards){auto &s=*sp;const int64_t id=gguf_find_tensor(s.meta,name.c_str());if(id<0)continue;need(!found,"duplicate tensor");found=true;s.map();
                const auto ne=gguf_get_tensor_ne(s.meta,id);const auto type=gguf_get_tensor_type(s.meta,id);need(std::string(ggml_type_name(type))==pj["type"],"tensor type");
                for(int k=0;k<4;++k)need(ne[k]==pj["ne"][k].get<int64_t>(),"tensor shape");
                w[i]=ggml_new_tensor_3d(wc,type,ne[0],ne[1],ne[2]);ggml_set_name(w[i],name.c_str());const size_t offset=gguf_get_data_offset(s.meta)+gguf_get_tensor_offset(s.meta,id);
                need(offset<=s.bytes&&ggml_nbytes(w[i])<=s.bytes-offset,"tensor bounds");need(ggml_backend_tensor_alloc(s.buffer,w[i],static_cast<char*>(s.data)+offset)==GGML_STATUS_SUCCESS,"tensor attach");
                std::vector<uint32_t> compact;json native=json::array();
                for(size_t slot=0;slot<6;++slot){const auto e=routes[slot];need(e>=0&&e<384,"route bound");
                    const bool cached=pj["compact_mask"][slot];bool planned=false;
                    if(cpuplan.contains(name)){auto vv=cpuplan[name].get<std::vector<uint32_t>>();planned=std::find(vv.begin(),vv.end(),uint32_t(e))!=vv.end();}
                    need(cached==planned,"captured/live plan membership");if(cached)compact.push_back(e);
                    native.push_back({{"expert",e},{"bytes",w[i]->nb[2]},{"FNV64",fnv(static_cast<char*>(w[i]->data)+size_t(e)*w[i]->nb[2],w[i]->nb[2])}});
                }
                if(!compact.empty())need(xeon_iq3_cache_add_selected_v2(cache,w[i],compact.data(),compact.size())==1,"cache registration");
                auto table=xeon_iq3_cache_lookup_experts_v2(w[i]);for(size_t slot=0;slot<6;++slot)need(bool(table&&table[routes[slot]])==pj["compact_mask"][slot].get<bool>(),"actual compact lookup");
                record["weights"].push_back({{"name",name},{"file",s.path},{"file_bytes",s.bytes},{"offset",offset},{"type",ggml_type_name(type)},{"ne",pj["ne"]},{"compact_experts",compact},{"selected_native",native}});
            }need(found,"tensor missing "+name);
        }
        x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,5120,1,1);ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,1);
        auto up=ggml_mul_mat_id(ctx,w[1],x,ids);auto gate=ggml_mul_mat_id(ctx,w[0],x,ids);
        auto act=ggml_swiglu_clamp(ctx,gate,up,cell["clamp"].get<float>());y=ggml_mul_mat_id(ctx,w[2],act,ids);
        graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)need(ggml_backend_supports_op(cpu,ggml_graph_node(graph,i)),"unsupported CPU graph");
        actbuf=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(actbuf,"activation allocation");ggml_backend_buffer_clear(actbuf,0);
        auto input=readbytes(capture/cell["x_file"].get<std::string>());need(input.size()==5120*4,"captured x size");
        ggml_backend_tensor_set(x,input.data(),0,input.size());ggml_backend_tensor_set(ids,routes.data(),0,24);
        captured=readbytes(capture/cell["down_file"].get<std::string>());need(captured.size()==5120*6*4,"captured output");actual.resize(5120*6);
        record["cell"]=cell["cell"];record["layer"]=cell["layer"];record["ids"]=routes;record["x_FNV64"]=fnv(input.data(),input.size());record["compact_bytes"]=xeon_iq3_cache_used_v1(cache);record["activation_allocation"]=ggml_backend_buffer_get_size(actbuf);
    }
    void compute(){need(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"graph compute");}
    void output(){ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);for(float v:actual)need(std::isfinite(v),"nonfinite output");need(std::memcmp(actual.data(),captured.data(),captured.size())==0,"output differs from captured baseline");}
    ~Fixture(){if(cache)xeon_iq3_cache_free_v1(cache);if(actbuf)ggml_backend_buffer_free(actbuf);if(ctx)ggml_free(ctx);if(wc)ggml_free(wc);}
};
int main(int argc,char **argv) try{
    need(argc==3,"SPEC OUTPUT_DIR");auto spec=readj(argv[1]);fs::path out=argv[2];need(!fs::exists(out/"RUN.json"),"output exists");fs::create_directories(out);
    json result;result["spec"]=spec;result["providers"]={{"from_float",provider(reinterpret_cast<void*>(ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float))},{"CPU",provider(reinterpret_cast<void*>(ggml_backend_cpu_init))},{"cache",provider(reinterpret_cast<void*>(xeon_iq3_cache_add_selected_v2))},{"base",provider(reinterpret_cast<void*>(ggml_init))}};
    for(auto key:{"CPU","cache","from_float"}) {
        need(result["providers"][key]==fs::canonical(spec["CPU"].get<std::string>()).string(),"CPU provider mismatch");
    }
    need(result["providers"]["base"]==fs::canonical(spec["base"].get<std::string>()).string(),"base provider mismatch");
    std::vector<std::unique_ptr<Shard>> shards;for(auto &p:spec["shards"])shards.push_back(std::make_unique<Shard>(p.get<std::string>()));
    auto cpu=ggml_backend_cpu_init();need(cpu,"backend");auto capture=readj(spec["capture_json"].get<std::string>());result["blocks"]=json::array();
    for(int ci:spec["cells"].get<std::vector<int>>()){
        auto cell=capture["cells"].at(ci);need(cell["cell"]==ci,"cell index");Fixture fixture(cpu,shards,cell,fs::path(spec["capture_json"].get<std::string>()).parent_path(),spec["cpu_plan"]);
        for(int threads:spec["threads"].get<std::vector<int>>()){
            ggml_backend_cpu_set_n_threads(cpu,threads);auto p=ggml_threadpool_params_default(threads);p.poll=0;p.strict_cpu=true;for(int i=0;i<threads;++i)p.cpumask[i]=true;
            auto pool=ggml_threadpool_new(&p);need(pool,"threadpool");ggml_backend_cpu_set_threadpool(cpu,pool);
            for(int i=0;i<3;++i) {fixture.compute();}
            fixture.output();
            json block=fixture.record;block["threads"]=threads;block["output_matches_capture"]=true;block["samples"]=json::array();
            const int loops=spec["loops"],samples=spec["samples"];need(loops>0&&samples>0,"loop count");
            for(int sample=0;sample<samples;++sample){const int64_t begin=ns();for(int i=0;i<loops;++i)fixture.compute();const int64_t end=ns();fixture.output();block["samples"].push_back({{"sample",sample},{"begin_ns",begin},{"end_ns",end},{"calls",loops},{"output_FNV64",fnv(fixture.actual.data(),fixture.actual.size()*4)}});}
            std::vector<int> cpus(threads);std::vector<long> tids(threads);
#pragma omp parallel num_threads(threads)
            {const int i=omp_get_thread_num();cpus.at(i)=sched_getcpu();tids.at(i)=syscall(SYS_gettid);}
            block["worker_receipt_after_timing"]={{"CPU",cpus},{"TID",tids}};
            const std::string name="cell"+std::to_string(ci)+"-t"+std::to_string(threads)+".f32";std::ofstream f(out/name,std::ios::binary);f.write(reinterpret_cast<const char*>(fixture.actual.data()),fixture.actual.size()*4);f.close();need(bool(f),"output write");block["output_file"]=name;result["blocks"].push_back(block);
            ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);
        }
    }
    ggml_backend_free(cpu);result["passed"]=true;result["scope"]="Isolated CPU FFN using actual saved x/top6/clamp and read-only original GGUF weights, actual subset cache. No main-model scheduler, GPU, API or token throughput measurement.";
    std::ofstream f(out/"RUN.json");f<<result.dump(2)<<'\n';f.close();need(bool(f),"RUN write");std::cout<<"real FFN passed\n";return 0;
}catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
