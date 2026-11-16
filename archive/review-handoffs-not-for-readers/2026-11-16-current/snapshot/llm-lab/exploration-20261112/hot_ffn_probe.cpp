// Isolated real-weight T=1 FFN residency experiment. Not a production dispatcher.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using Clock=std::chrono::steady_clock;
constexpr int D=5120,F=2304,E=384,TOP=6;
static void need(bool value,const char * message){if(!value)throw std::runtime_error(message);}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
int get_mmvq_mmid_max_batch(ggml_type,int);
struct Info{std::string path,name;ggml_type type;int64_t k,m,e;size_t offset,expert_bytes;};
struct Meta{std::array<Info,3>w;float clamp=0;};
static Meta metadata(const std::string & directory,int layer){
    Meta out;std::array<bool,3>found{};bool clamp_found=false;
    std::vector<std::filesystem::path>paths;
    for(auto & p:std::filesystem::directory_iterator(directory))if(p.path().extension()==".gguf")paths.push_back(p.path());
    std::sort(paths.begin(),paths.end());const char * parts[]={"gate","up","down"};
    for(auto & path:paths){
        auto m=gguf_init_from_file(path.c_str(),{true,nullptr});need(m,"metadata");
        for(int64_t i=0;i<gguf_get_n_kv(m);++i)if(std::string(gguf_get_key(m,i)).ends_with(".swiglu_clamp_exp")){
            if(gguf_get_kv_type(m,i)==GGUF_TYPE_ARRAY){
                need(gguf_get_arr_type(m,i)==GGUF_TYPE_FLOAT32&&size_t(layer)<gguf_get_arr_n(m,i),"clamp array");
                out.clamp=static_cast<const float*>(gguf_get_arr_data(m,i))[layer];
            }else{need(gguf_get_kv_type(m,i)==GGUF_TYPE_FLOAT32,"clamp type");out.clamp=gguf_get_val_f32(m,i);}
            clamp_found=true;
        }
        for(int j=0;j<3;++j){
            auto name="blk."+std::to_string(layer)+".ffn_"+parts[j]+"_exps.weight";
            auto id=gguf_find_tensor(m,name.c_str());if(id<0)continue;need(!found[j],"duplicate tensor");
            const auto * n=gguf_get_tensor_ne(m,id);auto q=gguf_get_tensor_type(m,id);
            need(n[3]==1&&n[2]==E,"expert shape");
            out.w[j]={path.string(),name,q,n[0],n[1],n[2],gguf_get_data_offset(m)+gguf_get_tensor_offset(m,id),ggml_row_size(q,n[0])*size_t(n[1])};found[j]=true;
        }
        gguf_free(m);
    }
    need(std::all_of(found.begin(),found.end(),[](bool v){return v;})&&clamp_found,"missing tensor/clamp");
    need(std::isfinite(out.clamp)&&out.clamp>0,"finite positive clamp required");
    for(int j=0;j<3;++j)need(out.w[j].k==(j==2?F:D)&&out.w[j].m==(j==2?D:F),"model dimensions");
    return out;
}
struct Mapping{
    Info info;void * base=nullptr;size_t length=0;uint8_t * raw=nullptr;ggml_backend_buffer_t buffer=nullptr;
    explicit Mapping(const Info & i):info(i){
        size_t page=sysconf(_SC_PAGESIZE),start=i.offset/page*page,skip=i.offset-start;length=skip+i.expert_bytes*i.e;
        int fd=open(i.path.c_str(),O_RDONLY);need(fd>=0,"open model");struct stat st{};
        bool valid=fstat(fd,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>0&&start<=size_t(st.st_size)&&length<=size_t(st.st_size)-start;
        if(!valid){close(fd);throw std::runtime_error("model tensor range");}
        base=mmap(nullptr,length,PROT_READ,MAP_SHARED,fd,start);close(fd);need(base!=MAP_FAILED,"mmap");
        raw=static_cast<uint8_t*>(base)+skip;
        buffer=ggml_backend_cpu_buffer_from_ptr(raw,i.expert_bytes*i.e);need(buffer,"CPU weight wrapper");
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ~Mapping(){if(buffer)ggml_backend_buffer_free(buffer);if(base&&base!=MAP_FAILED)munmap(base,length);}
};
struct CacheAPI{
    void * (*create)(size_t,size_t,int)=nullptr;
    int (*add)(void*,const ggml_tensor*,const uint32_t*,size_t)=nullptr;
    void (*free)(void*)=nullptr;uint64_t (*hits)()=nullptr;size_t (*used)(const void*)=nullptr;
    CacheAPI(){
        create=reinterpret_cast<decltype(create)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_create_v1"));
        add=reinterpret_cast<decltype(add)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_add_selected_v2"));
        free=reinterpret_cast<decltype(free)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_free_v1"));
        hits=reinterpret_cast<decltype(hits)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_hits_v1"));
        used=reinterpret_cast<decltype(used)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_used_v1"));
        need(create&&add&&free&&hits&&used,"selected cache ABI");
    }
};
struct Banks{
    ggml_context * ctx=nullptr;ggml_backend_buffer_t gpu_buffer=nullptr;
    std::array<std::unique_ptr<Mapping>,3>maps;std::array<ggml_tensor*,3>cpu{},gpu{};
    CacheAPI api;void * cache=nullptr;size_t payload_bytes=0;int cache_tensors=0;
    Banks(const Meta & m,ggml_backend_t backend,const std::array<int32_t,TOP>&routes,const std::array<int,TOP>&order){
        ctx=ggml_init({1024*1024,nullptr,true});need(ctx,"weights metadata");
        for(int j=0;j<3;++j){auto & i=m.w[j];maps[j]=std::make_unique<Mapping>(i);
            cpu[j]=ggml_new_tensor_3d(ctx,i.type,i.k,i.m,i.e);
            need(ggml_backend_tensor_alloc(maps[j]->buffer,cpu[j],maps[j]->raw)==GGML_STATUS_SUCCESS,"CPU tensor bind");
            gpu[j]=ggml_new_tensor_3d(ctx,i.type,i.k,i.m,TOP);payload_bytes+=i.expert_bytes*TOP;
            need(get_mmvq_mmid_max_batch(i.type,700)>=1,"T1 MMVQ support");
        }
        gpu_buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);need(gpu_buffer,"GPU weight allocation");
        ggml_backend_buffer_set_usage(gpu_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_buffer_clear(gpu_buffer,0);
        for(int j=0;j<3;++j)for(int local=0;local<TOP;++local){auto & map=*maps[j];
            ggml_backend_tensor_set(gpu[j],map.raw+size_t(routes[order[local]])*map.info.expert_bytes,size_t(local)*map.info.expert_bytes,map.info.expert_bytes);
        }
        ggml_backend_synchronize(backend);
    }
    void add_cache(const std::array<int32_t,TOP>&routes){
        cache=api.create(size_t(128)*1024*1024,size_t(64)*1024*1024*1024,12);need(cache,"cache owner");
        std::array<uint32_t,TOP>selected{};std::copy(routes.begin(),routes.end(),selected.begin());
        for(auto w:cpu){const bool supported=w->type==GGML_TYPE_IQ2_S||w->type==GGML_TYPE_IQ3_XXS||w->type==GGML_TYPE_IQ3_S;
            if(supported){need(api.add(cache,w,selected.data(),selected.size())==1,"cache registration");++cache_tensors;}
        }
    }
    ~Banks(){if(cache)api.free(cache);if(gpu_buffer)ggml_backend_buffer_free(gpu_buffer);if(ctx)ggml_free(ctx);}
};
struct Graph{
    ggml_context * ctx=nullptr;ggml_backend_buffer_t buf=nullptr;ggml_backend_t backend;
    ggml_tensor * x=nullptr,* ids=nullptr;std::array<ggml_tensor*,4>nodes{};ggml_cgraph * graph=nullptr;
    Graph(ggml_backend_t b,const std::array<ggml_tensor*,3>&w,int slots,float clamp):backend(b){
        ctx=ggml_init({2*1024*1024,nullptr,true});need(ctx,"graph context");
        x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,1,1);ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,slots,1);
        nodes[0]=ggml_mul_mat_id(ctx,w[0],x,ids);nodes[1]=ggml_mul_mat_id(ctx,w[1],x,ids);
        nodes[2]=ggml_swiglu_clamp(ctx,nodes[0],nodes[1],clamp);nodes[3]=ggml_mul_mat_id(ctx,w[2],nodes[2],ids);
        graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,nodes[3]);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)need(ggml_backend_supports_op(backend,ggml_graph_node(graph,i)),"unsupported FFN node");
        buf=ggml_backend_alloc_ctx_tensors(ctx,backend);need(buf,"activation allocation");ggml_backend_buffer_clear(buf,0);
    }
    void compute(){need(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"CPU compute");}
    std::array<std::vector<float>,4> values(){
        std::array<std::vector<float>,4>result;
        for(int j=0;j<4;++j){result[j].resize(ggml_nelements(nodes[j]));ggml_backend_tensor_get(nodes[j],result[j].data(),0,ggml_nbytes(nodes[j]));}
        return result;
    }
    ~Graph(){if(buf)ggml_backend_buffer_free(buf);if(ctx)ggml_free(ctx);}
};
// FP64 accumulation of dequantized real weights, rounding each projection to F32.
static std::array<std::vector<float>,4> reference(const Banks & bank,const std::array<int32_t,TOP>&routes,const std::vector<float>&x,float limit){
    std::array<std::vector<float>,4>out;
    for(int j=0;j<4;++j)out[j].resize(size_t(j==3?D:F)*TOP);
    std::vector<float>row(D);
    for(int slot=0;slot<TOP;++slot){
        for(int p=0;p<2;++p){const auto & map=*bank.maps[p];auto fn=ggml_get_type_traits(map.info.type)->to_float;need(fn,"dequantize trait");
            const auto * raw=map.raw+size_t(routes[slot])*map.info.expert_bytes;const auto stride=ggml_row_size(map.info.type,D);
            for(int m=0;m<F;++m){fn(raw+size_t(m)*stride,row.data(),D);double value=0;for(int k=0;k<D;++k)value+=double(row[k])*x[k];out[p][slot*F+m]=float(value);}
        }
        for(int m=0;m<F;++m){float g=std::min(out[0][slot*F+m],limit),u=std::clamp(out[1][slot*F+m],-limit,limit);out[2][slot*F+m]=(g/(1.f+std::exp(-g)))*u;}
        const auto & map=*bank.maps[2];auto fn=ggml_get_type_traits(map.info.type)->to_float;need(fn,"down dequantize trait");
        const auto * raw=map.raw+size_t(routes[slot])*map.info.expert_bytes;const auto stride=ggml_row_size(map.info.type,F);
        for(int m=0;m<D;++m){fn(raw+size_t(m)*stride,row.data(),F);double value=0;for(int k=0;k<F;++k)value+=double(row[k])*out[2][slot*F+k];out[3][slot*D+m]=float(value);}
    }
    return out;
}
struct Error{double nmse=0,rmse=0,max_abs=0;size_t n=0;};
static Error error(const std::vector<float>&got,const std::vector<float>&ref){
    need(got.size()==ref.size()&&!got.empty(),"error shape");double s=0,norm=0,maxabs=0;
    for(size_t i=0;i<got.size();++i){need(std::isfinite(got[i])&&std::isfinite(ref[i]),"nonfinite output");double d=double(got[i])-ref[i];s+=d*d;norm+=double(ref[i])*ref[i];maxabs=std::max(maxabs,std::abs(d));}
    need(norm>0,"zero reference norm");return {s/norm,std::sqrt(s/got.size()),maxabs,got.size()};
}
static void print_error(const Error&e){std::printf("{\"nmse\":%.12g,\"rmse\":%.12g,\"max_abs\":%.12g,\"n\":%zu}",e.nmse,e.rmse,e.max_abs,e.n);}
struct Timing{double total=0,enqueue=0,cpu=0,wait=0,scatter=0;};
struct Plan{
    Banks & banks;Graph & cpu_graph;ggml_backend_t gpu_backend;int hot;
    std::array<int32_t,TOP>routes;std::array<int,TOP>order;std::unique_ptr<Graph>gpu_graph;
    float * pinned_x=nullptr,*pinned_y=nullptr;int32_t * pinned_ids=nullptr;std::vector<float>result;
    Plan(Banks & b,Graph & c,ggml_backend_t g,int h,const std::array<int32_t,TOP>&r,const std::array<int,TOP>&o,float clamp):banks(b),cpu_graph(c),gpu_backend(g),hot(h),routes(r),order(o),result(D*TOP){
        ck(cudaHostAlloc(&pinned_x,D*sizeof(float),cudaHostAllocPortable));
        ck(cudaHostAlloc(&pinned_y,D*TOP*sizeof(float),cudaHostAllocPortable));
        ck(cudaHostAlloc(&pinned_ids,TOP*sizeof(int32_t),cudaHostAllocPortable));
        if(hot)gpu_graph=std::make_unique<Graph>(gpu_backend,banks.gpu,hot,clamp);
    }
    Timing run(const std::vector<float>&input,bool concurrent){
        Timing t;auto start=Clock::now();
        std::array<int32_t,TOP>cpu_ids=routes;
        for(int j=0;j<hot;++j){cpu_ids[order[j]]=-1;pinned_ids[j]=j;}
        if(hot<TOP){ggml_backend_tensor_set(cpu_graph.x,input.data(),0,D*sizeof(float));ggml_backend_tensor_set(cpu_graph.ids,cpu_ids.data(),0,TOP*sizeof(int32_t));}
        if(hot){
            std::memcpy(pinned_x,input.data(),D*sizeof(float));
            ggml_backend_tensor_set_async(gpu_backend,gpu_graph->x,pinned_x,0,D*sizeof(float));
            ggml_backend_tensor_set_async(gpu_backend,gpu_graph->ids,pinned_ids,0,hot*sizeof(int32_t));
            need(ggml_backend_graph_compute_async(gpu_backend,gpu_graph->graph)==GGML_STATUS_SUCCESS,"GPU async compute");
            ggml_backend_tensor_get_async(gpu_backend,gpu_graph->nodes[3],pinned_y,0,size_t(hot)*D*sizeof(float));
        }
        t.enqueue=ms(start);
        if(hot&&!concurrent){auto tick=Clock::now();ggml_backend_synchronize(gpu_backend);t.wait+=ms(tick);}
        if(hot<TOP){auto tick=Clock::now();cpu_graph.compute();ggml_backend_tensor_get(cpu_graph.nodes[3],result.data(),0,result.size()*sizeof(float));t.cpu=ms(tick);}
        if(hot&&concurrent){auto tick=Clock::now();ggml_backend_synchronize(gpu_backend);t.wait+=ms(tick);}
        auto tick=Clock::now();for(int j=0;j<hot;++j)std::memcpy(result.data()+order[j]*D,pinned_y+j*D,D*sizeof(float));
        t.scatter=ms(tick);t.total=ms(start);return t;
    }
    ~Plan(){ggml_backend_synchronize(gpu_backend);gpu_graph.reset();if(pinned_ids)cudaFreeHost(pinned_ids);if(pinned_y)cudaFreeHost(pinned_y);if(pinned_x)cudaFreeHost(pinned_x);}
};
static void times(const char * label,const std::vector<Timing>&v,double Timing::*member){
    std::printf(",\"%s\":[",label);for(size_t i=0;i<v.size();++i)std::printf("%s%.9g",i?",":"",v[i].*member);std::printf("]");
}
static void provider(void * fn,const char * expected){Dl_info info{};need(dladdr(fn,&info),"provider lookup");need(std::filesystem::canonical(info.dli_fname)==std::filesystem::canonical(expected),"wrong loaded provider");}
int main(int argc,char ** argv)try{
    need(argc==7,"MODEL_DIRECTORY LAYER SEED REPEATS EXPECTED_CPU_LIBRARY EXPECTED_CUDA_LIBRARY");
    const int layer=std::stoi(argv[2]),seed=std::stoi(argv[3]),repeats=std::stoi(argv[4]);need((layer==4||layer==24)&&repeats>=10&&repeats<=300,"bounded arguments");
    provider(reinterpret_cast<void*>(ggml_backend_cpu_init),argv[5]);provider(reinterpret_cast<void*>(ggml_backend_cuda_init),argv[6]);
    need(!prctl(PR_SET_THP_DISABLE,1,0,0,0),"THP disable");
    auto meta=metadata(argv[1],layer);ck(cudaSetDevice(0));cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));need(prop.major==7&&prop.minor==0,"Volta target");
    size_t before,total;ck(cudaMemGetInfo(&before,&total));need(before>size_t(700)*1024*1024,"GPU headroom");
    auto cpu=ggml_backend_cpu_init(),gpu=ggml_backend_cuda_init(0);need(cpu&&gpu,"backends");
    auto params=ggml_threadpool_params_default(12);params.poll=0;params.strict_cpu=true;for(int i=0;i<12;++i)params.cpumask[i]=true;
    auto pool=ggml_threadpool_new(&params);need(pool,"threadpool");ggml_backend_cpu_set_n_threads(cpu,12);ggml_backend_cpu_set_threadpool(cpu,pool);
    std::mt19937 rng(seed);std::uniform_real_distribution<float>dist(-.1f,.1f);std::vector<float>x(D);for(auto & v:x)v=dist(rng);
    std::array<int32_t,TOP>routes={3,41,92,151,244,370};std::shuffle(routes.begin(),routes.end(),rng);
    const std::array<int,TOP>hot_order={4,1,5,0,3,2};
    {
        Banks banks(meta,gpu,routes,hot_order);Graph cg(cpu,banks.cpu,TOP,meta.clamp);
        ggml_backend_tensor_set(cg.x,x.data(),0,x.size()*4);ggml_backend_tensor_set(cg.ids,routes.data(),0,TOP*4);cg.compute();auto raw_cpu=cg.values();
        banks.add_cache(routes);cg.compute();auto base=cg.values();
        for(int j=0;j<4;++j)need(std::memcmp(raw_cpu[j].data(),base[j].data(),base[j].size()*4)==0,"real-weight compact cache changed output");
        auto ref=reference(banks,routes,x,meta.clamp);
        size_t after;ck(cudaMemGetInfo(&after,&total));
        std::printf("{\"schema\":1,\"layer\":%d,\"seed\":%d,\"repeats\":%d,\"cpu_threads\":12,\"input_amplitude\":0.1,\"clamp\":%.9g,\"routes\":[",layer,seed,repeats,meta.clamp);
        for(int j=0;j<TOP;++j)std::printf("%s%d",j?",":"",routes[j]);
        std::printf("],\"hot_slot_order\":[4,1,5,0,3,2],\"GPU_bank_payload_bytes\":%zu,\"GPU_memory_delta_setup_bytes\":%zu,\"CPU_compact_bytes\":%zu,\"CPU_compact_tensors\":%d,\"weight_upload_timed\":false,\"whole_model_benchmark\":false,\"boundary\":\"host input to canonical host unweighted6-slot output; partition,input copies,launch,CPU,GPU wait,scatter included\",\"CPU_error_vs_FP64_dequant_reference\":[",banks.payload_bytes,before-after,banks.api.used(banks.cache),banks.cache_tensors);
        for(int j=0;j<4;++j){if(j)std::printf(",");print_error(error(base[j],ref[j]));}
        std::printf("],\"measurements\":[");bool first=true;
        const int schedule[]={0,1,3,6,3,1,0};
        for(int h:schedule){
            Plan plan(banks,cg,gpu,h,routes,hot_order,meta.clamp);
            for(int i=0;i<4;++i)plan.run(x,true);
            auto hits=banks.api.hits();plan.run(x,false);auto serial_result=plan.result;plan.run(x,true);
            need(std::memcmp(serial_result.data(),plan.result.data(),plan.result.size()*4)==0,"serial/concurrent output mismatch");
            if(h<TOP&&banks.cache_tensors)need(banks.api.hits()>hits,"compact cache path not observed");
            for(int slot=0;slot<TOP;++slot)if(std::find(hot_order.begin(),hot_order.begin()+h,slot)==hot_order.begin()+h)
                need(std::memcmp(plan.result.data()+slot*D,base[3].data()+slot*D,D*4)==0,"cold slot changed");
            auto mixed_error=error(plan.result,ref[3]);auto delta_cpu=error(plan.result,base[3]);
            // Same-precision accumulation/activation paths differ; preserve actual error rather than claiming bit equivalence.
            std::vector<Timing>serial,concurrent;
            for(int i=0;i<repeats;++i){
                if(i%2){concurrent.push_back(plan.run(x,true));serial.push_back(plan.run(x,false));}
                else{serial.push_back(plan.run(x,false));concurrent.push_back(plan.run(x,true));}
            }
            need(std::memcmp(serial_result.data(),plan.result.data(),plan.result.size()*4)==0,"nonrepeatable final output");
            std::printf("%s{\"hot\":%d,\"cold\":%d,\"GPU_active_payload_bytes\":%zu,\"mixed_error_vs_FP64_reference\":",first?"":",",h,TOP-h,banks.payload_bytes/TOP*h);first=false;print_error(mixed_error);
            std::printf(",\"mixed_error_vs_CPU\":");print_error(delta_cpu);
            // CUDA can fuse gate/up/GLU and leave the gate/up buffers unwritten.
            // Inspect only materialized activation/down; do not disable fusion in timings.
            bool numeric_gate=mixed_error.nmse<1e-2;
            std::printf(",\"GPU_materialized_stages\":[\"activation\",\"down\"],\"GPU_stage_error_vs_FP64_reference\":[");
            if(h){auto gv=plan.gpu_graph->values();
                for(int j=0;j<h;++j)need(std::memcmp(plan.result.data()+hot_order[j]*D,gv[3].data()+j*D,D*4)==0,"GPU hot-slot scatter mismatch");
                for(int p=2;p<4;++p){
                int width=p==3?D:F;std::vector<float>selected(size_t(width)*h);
                for(int j=0;j<h;++j)std::memcpy(selected.data()+j*width,ref[p].data()+hot_order[j]*width,width*4);
                auto stage_error=error(gv[p],selected);numeric_gate=numeric_gate&&stage_error.nmse<1e-2;
                if(p>2)std::printf(",");print_error(stage_error);
            }}std::printf("]");
            times("serial_ms",serial,&Timing::total);times("concurrent_ms",concurrent,&Timing::total);
            times("concurrent_enqueue_ms",concurrent,&Timing::enqueue);times("concurrent_cpu_ms",concurrent,&Timing::cpu);
            times("concurrent_wait_ms",concurrent,&Timing::wait);times("concurrent_scatter_ms",concurrent,&Timing::scatter);
            std::printf(",\"serial_concurrent_byte_identical\":true,\"cold_slots_byte_identical\":true,\"GPU_hot_slots_byte_identical\":true,\"numeric_diagnostic_gate_nmse_limit\":0.01,\"numeric_diagnostic_gate_passed\":%s}",numeric_gate?"true":"false");std::fflush(stdout);
        }
        std::puts("],\"pass\":true}");
    }
    ggml_backend_synchronize(gpu);ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
