#include "route_features.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Graph {
    ggml_context *ctx;ggml_backend_t backend;ggml_backend_buffer_t buffer;
    ggml_tensor *weights,*input,*ids,*output;ggml_cgraph *gf;
    Graph(ggml_backend_t b,ggml_type q,int64_t K,int64_t M,int E,int T,bool up):backend(b){
        ctx=ggml_init({4*1024*1024,nullptr,true});if(!ctx)throw std::runtime_error("metadata allocation");
        weights=ggml_new_tensor_3d(ctx,q,K,M,E);input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,K,up?1:6,T);
        ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,T);output=ggml_mul_mat_id(ctx,weights,input,ids);
        gf=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(gf,output);
        if(!ggml_backend_supports_op(b,output))throw std::runtime_error("unsupported operator");
        buffer=ggml_backend_alloc_ctx_tensors(ctx,b);if(!buffer)throw std::runtime_error("buffer allocation");
        ggml_backend_buffer_clear(buffer,0);
    }
    void compute(){if(ggml_backend_graph_compute(backend,gf)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute");ggml_backend_synchronize(backend);}
    ~Graph(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
};

static void array(const char *key,const std::vector<double> &v){std::printf(",\"%s\":[",key);for(size_t i=0;i<v.size();++i)std::printf("%s%.9g",i?",":"",v[i]);std::printf("]");}

int main(int argc,char**argv) try {
    if(argc!=8)throw std::runtime_error("shard tensor T E routes.bin seed repeats");
    int T=std::stoi(argv[3]),E=std::stoi(argv[4]),seed=std::stoi(argv[6]),reps=std::stoi(argv[7]);
    if(T<16 || T>1024 || E<6 || E>384 || reps<3 || reps>7)throw std::runtime_error("test bounds");
    bool up=std::string(argv[2]).find("down_exps")==std::string::npos;
    auto meta=gguf_init_from_file(argv[1],{true,nullptr});if(!meta)throw std::runtime_error("GGUF metadata");
    auto ti=gguf_find_tensor(meta,argv[2]);if(ti<0)throw std::runtime_error("tensor missing");
    auto q=gguf_get_tensor_type(meta,ti);auto ne=gguf_get_tensor_ne(meta,ti);int64_t K=ne[0],M=ne[1];
    if(K>5120 || M>5120 || E>ne[2] || ne[3]!=1)throw std::runtime_error("shape mismatch");
    size_t rowbytes=ggml_row_size(q,K),expertbytes=rowbytes*M,bytes=expertbytes*E;
    size_t offset=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,ti);gguf_free(meta);
    size_t page=sysconf(_SC_PAGESIZE),aligned=offset/page*page,skip=offset-aligned;
    int fd=open(argv[1],O_RDONLY);if(fd<0)throw std::runtime_error("weight file");
    struct stat st;if(fstat(fd,&st)||offset>size_t(st.st_size)||bytes>size_t(st.st_size)-offset)throw std::runtime_error("weight range");
    void *mapped=mmap(nullptr,skip+bytes,PROT_READ,MAP_SHARED,fd,aligned);close(fd);if(mapped==MAP_FAILED)throw std::runtime_error("mmap");
    auto raw=static_cast<const uint8_t *>(mapped)+skip;
    std::vector<int32_t> routes(size_t(T)*6);std::ifstream rf(argv[5],std::ios::binary);
    if(!rf.read(reinterpret_cast<char*>(routes.data()),routes.size()*4)||rf.peek()!=EOF)throw std::runtime_error("route fixture length");
    auto features=expert_shadow::describe(routes.data(),routes.size()*4,E,T,6,4,24,expertbytes);
    ck(cudaSetDevice(0));size_t free,total;ck(cudaMemGetInfo(&free,&total));
    const size_t activations=size_t(K*(up?1:6)+M*6)*T*4;
    const size_t workspace=size_t(K+M)*6*T*4+size_t(K)*6*T*2+size_t(K)*M*6+128*1024*1024;
    const size_t reserve=bytes+activations+workspace;
    if(free<reserve+400*1024*1024)throw std::runtime_error("conservative GPU headroom check failed");
    auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_cuda_init(0);if(!cpu||!gpu)throw std::runtime_error("backend init");
    auto pp=ggml_threadpool_params_default(24);pp.poll=0;pp.strict_cpu=true;for(int i=0;i<24;++i)pp.cpumask[i]=true;
    auto pool=ggml_threadpool_new(&pp);ggml_backend_cpu_set_n_threads(cpu,24);ggml_backend_cpu_set_threadpool(cpu,pool);
    bool registered=ggml_backend_cuda_register_host_buffer(mapped,skip+bytes);if(!registered)throw std::runtime_error("host registration failed");
    {
        Graph c(cpu,q,K,M,E,T,up),g(gpu,q,K,M,E,T,up);
        ggml_backend_buffer_set_usage(g.buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        std::mt19937 rng(seed);std::uniform_real_distribution<float> dist(-.5f,.5f);
        std::vector<float>x(size_t(K)*(up?1:6)*T);for(auto &v:x)v=dist(rng);
        for(Graph *v:{&c,&g}){
            ggml_backend_tensor_set(v->input,x.data(),0,x.size()*4);ggml_backend_tensor_set(v->ids,routes.data(),0,routes.size()*4);
        }
        ggml_backend_tensor_set(c.weights,raw,0,bytes);
        auto upload=[&]{
            for(int e=0;e<E;){
                if(!features.counts[e]){++e;continue;}int last=e;while(last+1<E && features.counts[last+1])++last;
                size_t n=size_t(last+1-e)*expertbytes+(last<E-1?std::min<size_t>(expertbytes,512):0);
                ggml_backend_tensor_set_async(gpu,g.weights,raw+size_t(e)*expertbytes,size_t(e)*expertbytes,n);e=last+1;
            }
        };
        upload();g.compute();c.compute();
        std::vector<float> yc(ggml_nelements(c.output)),yg(yc.size());
        ggml_backend_tensor_get(c.output,yc.data(),0,yc.size()*4);ggml_backend_tensor_get(g.output,yg.data(),0,yg.size()*4);
        double norm=0,error=0,refnorm=0,ce=0,ge=0;int nonfinite=0;
        for(size_t i=0;i<yc.size();++i){nonfinite+=!std::isfinite(yc[i])||!std::isfinite(yg[i]);norm+=double(yc[i])*yc[i];error+=std::pow(double(yc[i])-yg[i],2);}
        auto to_float=ggml_get_type_traits(q)->to_float;std::vector<float> row(K);
        for(int t=0;t<T;t+=std::max(1,T/11))for(int j=0;j<6;++j)for(int s=0;s<7;++s){
            int64_t m=(M-1)*s/6;to_float(raw+routes[t*6+j]*expertbytes+m*rowbytes,row.data(),K);
            const float *xx=x.data()+(t*(up?1:6)+(up?0:j))*K;double ref=0;for(int64_t k=0;k<K;++k)ref+=double(row[k])*xx[k];
            refnorm+=ref*ref;size_t out=(t*6+j)*M+m;ce+=std::pow(ref-yc[out],2);ge+=std::pow(ref-yg[out],2);
        }
        bool pass=nonfinite==0 && norm>0 && refnorm>0 && error/norm<5e-4 && ce/refnorm<5e-4 && ge/refnorm<5e-4;
        if(!pass)throw std::runtime_error("independent numeric validation failed");
        auto cpu_end=[&]{
            ggml_backend_tensor_get_async(gpu,g.input,c.input->data,0,x.size()*4);
            // Both choices start after the existing shared routing fetch. Host IDs are already available.
            ggml_backend_synchronize(gpu);
            c.compute();ggml_backend_tensor_set_async(gpu,g.output,c.output->data,0,yc.size()*4);ggml_backend_synchronize(gpu);
        };
        auto stream=[&]{ggml_backend_buffer_set_usage(g.buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);upload();g.compute();};
        auto resident=[&]{ggml_backend_buffer_set_usage(g.buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);g.compute();};
        cpu_end();stream();resident();
        ggml_backend_tensor_get(g.output,yg.data(),0,yg.size()*4);
        double resident_error=0,resident_ref_error=0;
        for(size_t i=0;i<yg.size();++i){if(!std::isfinite(yg[i]))throw std::runtime_error("nonfinite resident output");resident_error+=std::pow(double(yc[i])-yg[i],2);}
        for(int t=0;t<T;t+=std::max(1,T/11))for(int j=0;j<6;++j)for(int s=0;s<7;++s){
            int64_t m=(M-1)*s/6;to_float(raw+routes[t*6+j]*expertbytes+m*rowbytes,row.data(),K);
            const float *xx=x.data()+(t*(up?1:6)+(up?0:j))*K;double ref=0;for(int64_t k=0;k<K;++k)ref+=double(row[k])*xx[k];
            resident_ref_error+=std::pow(ref-yg[(t*6+j)*M+m],2);
        }
        if(resident_error/norm>=5e-4 || resident_ref_error/refnorm>=5e-4)throw std::runtime_error("resident independent validation failed");
        std::vector<double> tc,te,tg,tr;
        std::vector<int> order;
        auto timed=[](auto &&fn,std::vector<double>&out){auto a=Clock::now();fn();out.push_back(ms(a));};
        for(int r=0;r<reps;++r){
            // Four balanced rotations; each arm appears in every position.
            for(int i=0;i<4;++i){int arm=(r+i)%4;order.push_back(arm);
                if(arm==0)timed(cpu_end,te);else if(arm==1)timed(stream,tg);else if(arm==2)timed(resident,tr);else timed([&]{c.compute();},tc);
            }
        }
        std::printf("{\"schema\":1,\"quant\":\"%s\",\"direction\":\"%s\",\"K\":%lld,\"M\":%lld,\"experts\":%d,\"tokens\":%d,\"topk\":6,\"threads\":24,\"active_experts\":%llu,\"transfer_runs\":%llu,\"expert_bytes\":%zu,\"selected_weight_bytes\":%llu,\"submitted_weight_bytes\":%llu,\"cpu_gpu_nmse\":%.12g,\"cpu_ref_nmse\":%.12g,\"gpu_ref_nmse\":%.12g,\"pass\":true,\"histogram\":[",ggml_type_name(q),up?"up":"down",(long long)K,(long long)M,E,T,(unsigned long long)features.active,(unsigned long long)features.runs,expertbytes,(unsigned long long)features.selected_bytes,(unsigned long long)features.transfer_bytes,error/norm,ce/refnorm,ge/refnorm);
        for(int e=0;e<E;++e)std::printf("%s%u",e?",":"",features.counts[e]);std::printf("]");
        std::printf(",\"resident_cpu_nmse\":%.12g,\"resident_ref_nmse\":%.12g,\"arm_order\":[",resident_error/norm,resident_ref_error/refnorm);
        for(size_t i=0;i<order.size();++i)std::printf("%s%d",i?",":"",order[i]);std::printf("]");
        array("cpu_compute_ms",tc);array("cpu_gpu_boundary_ms",te);array("gpu_stream_ms",tg);array("gpu_resident_ms",tr);std::puts("}");
    }
    ggml_backend_cuda_unregister_host_buffer(mapped);ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(gpu);ggml_backend_free(cpu);munmap(mapped,skip+bytes);return 0;
}catch(const std::exception &e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
