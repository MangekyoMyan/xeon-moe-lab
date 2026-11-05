#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef ENABLE_PMU
#include "range_profiling.h"
#endif

static void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a) { return std::chrono::duration<double,std::milli>(Clock::now()-a).count(); }
static double median(std::vector<double> x) { std::sort(x.begin(),x.end());return x[x.size()/2]; }

int main(int argc,char **argv) try {
    if(argc != 8) throw std::runtime_error("shard tensor tokens experts uniform|skew timing|trace|pmu mode");
    const int T=std::stoi(argv[3]), E=std::stoi(argv[4]), selected=std::stoi(argv[7]);
    if(T<1 || T>1024 || E<6 || E>96 || (selected!=0 && selected!=1)) throw std::runtime_error("bounded test range");
    const std::string method=argv[6];
    const bool skew=std::string(argv[5])=="skew";
    const bool up=std::string(argv[2]).find("down_exps")==std::string::npos;
    auto set_mode=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"xeon_grouped_mode"));
    auto forced_calls=reinterpret_cast<uint64_t(*)()>(dlsym(RTLD_DEFAULT,"xeon_grouped_forced_calls"));
    if(!set_mode || !forced_calls) throw std::runtime_error("experimental shim missing");
    auto meta=gguf_init_from_file(argv[1],{true,nullptr});if(!meta)throw std::runtime_error("GGUF metadata");
    const auto ti=gguf_find_tensor(meta,argv[2]);if(ti<0)throw std::runtime_error("tensor not found");
    const auto type=gguf_get_tensor_type(meta,ti);const auto ne=gguf_get_tensor_ne(meta,ti);
    const int64_t K=ne[0],M=ne[1];if(ne[2]<E || K>5120 || M>5120)throw std::runtime_error("shape");
    const size_t rb=ggml_row_size(type,K), eb=rb*M, bytes=eb*E;
    const size_t offset=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,ti);
    const size_t page=sysconf(_SC_PAGESIZE), aligned=offset/page*page, skip=offset-aligned;
    int fd=open(argv[1],O_RDONLY);if(fd<0)throw std::runtime_error("open model");
    auto rawbase=mmap(nullptr,skip+bytes,PROT_READ,MAP_SHARED,fd,aligned);close(fd);
    if(rawbase==MAP_FAILED)throw std::runtime_error("mmap");
    const auto raw=static_cast<const uint8_t *>(rawbase)+skip;gguf_free(meta);
    ck(cudaSetDevice(0));size_t free,total;ck(cudaMemGetInfo(&free,&total));
    const size_t expected=bytes+eb+size_t(K+M)*6*T*4+256*1024*1024;
    if(free<expected+400*1024*1024)throw std::runtime_error("insufficient GPU headroom for bounded probe");
    auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_cuda_init(0);
    if(!cpu || !gpu)throw std::runtime_error("backend init");
    auto ctx=ggml_init({4*1024*1024,nullptr,true});
    auto bank=ggml_new_tensor_3d(ctx,type,K,M,E+1);
    auto weights=ggml_view_3d(ctx,bank,K,M,E,bank->nb[1],bank->nb[2],0);
    auto input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,K,up?1:6,T);
    auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,T);
    auto out=ggml_mul_mat_id(ctx,weights,input,ids);
    auto gf=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(gf,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);if(!buffer)throw std::runtime_error("GPU allocation");
    ggml_backend_buffer_clear(buffer,0);
    std::mt19937 rng(42);std::uniform_real_distribution<float> dist(-.5f,.5f);
    std::vector<float> x(ggml_nelements(input));for(auto &v:x)v=dist(rng);
    std::vector<int32_t> ix(6*T);
    for(int t=0;t<T;++t)for(int j=0;j<6;++j)ix[t*6+j]=skew?(t%4==0?(t*6+j)%E:j):(t*6+j)%E;
    ggml_backend_tensor_set(weights,raw,0,bytes);
    ggml_backend_tensor_set(input,x.data(),0,x.size()*4);ggml_backend_tensor_set(ids,ix.data(),0,ix.size()*4);
    auto compute=[&]{if(ggml_backend_graph_compute(gpu,gf)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute failed");ggml_backend_synchronize(gpu);};
    std::vector<float> ys(ggml_nelements(out)),yc(ys.size());
    set_mode(0);compute();compute();ggml_backend_tensor_get(out,ys.data(),0,ys.size()*4);
    set_mode(1);compute();compute();ggml_backend_tensor_get(out,yc.data(),0,yc.size()*4);
    double sq=0,norm=0,maxerr=0;size_t nonfinite=0;
    for(size_t i=0;i<ys.size();++i){nonfinite+=!std::isfinite(ys[i])||!std::isfinite(yc[i]);double d=double(ys[i])-yc[i];sq+=d*d;norm+=double(ys[i])*ys[i];maxerr=std::max(maxerr,std::abs(d));}
    std::vector<float> row(K);auto to_float=ggml_get_type_traits(type)->to_float;
    double refnorm=0,se=0,ce=0;
    for(int t=0;t<T;t+=std::max(1,T/13))for(int j=0;j<6;++j)for(int s=0;s<9;++s){
        int64_t m=(M-1)*s/8;to_float(raw+ix[6*t+j]*eb+m*rb,row.data(),K);
        const float *xx=x.data()+(t*(up?1:6)+(up?0:j))*K;double ref=0;
        for(int64_t k=0;k<K;++k)ref+=double(row[k])*xx[k];
        size_t oi=(t*6+j)*M+m;refnorm+=ref*ref;se+=std::pow(ref-ys[oi],2);ce+=std::pow(ref-yc[oi],2);
    }
    const double nmse=sq/std::max(norm,1e-30), sn=se/std::max(refnorm,1e-30), cn=ce/std::max(refnorm,1e-30);
    const bool pass=nonfinite==0 && nmse<5e-4 && sn<5e-4 && cn<5e-4;
    std::printf("VALIDATION {\"type\":\"%s\",\"K\":%lld,\"M\":%lld,\"tokens\":%d,\"experts\":%d,\"skew\":%s,\"weight_bytes\":%zu,\"nmse\":%.12g,\"stock_ref_nmse\":%.12g,\"candidate_ref_nmse\":%.12g,\"nonfinite\":%zu,\"max_abs\":%.9g,\"pass\":%s,\"forced_calls\":%llu}\n",ggml_type_name(type),(long long)K,(long long)M,T,E,skew?"true":"false",bytes,nmse,sn,cn,nonfinite,maxerr,pass?"true":"false",(unsigned long long)forced_calls());std::fflush(stdout);
    if(!pass)throw std::runtime_error("numerical check failed");
    if(method=="timing"){
        std::vector<double> times[2];
        for(int round=0;round<5;++round)for(int mode:{0,1,1,0}){
            set_mode(mode);auto start=Clock::now();for(int rep=0;rep<8;++rep)compute();times[mode].push_back(ms(start)/8);
        }
        std::printf("TIMING {\"stock_ms\":%.9g,\"candidate_ms\":%.9g,\"speedup\":%.9g,\"samples_each\":%zu,\"graphs_disabled\":true}\n",median(times[0]),median(times[1]),median(times[0])/median(times[1]),times[0].size());
    }else if(method=="trace"){
        set_mode(selected);
        auto begin=reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT,"xeon_trace_begin"));
        auto end=reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT,"xeon_trace_end"));
        if(!begin||!end)throw std::runtime_error("CUPTI trace missing");
        begin();for(int n=0;n<3;++n)compute();end();
    }
#ifdef ENABLE_PMU
    else if(method=="pmu"){
        set_mode(selected);CUcontext cu=nullptr;if(cuCtxGetCurrent(&cu)!=CUDA_SUCCESS||!cu)throw std::runtime_error("CUDA context");
        RangeProfilerConfig cfg;cfg.maxNumOfRanges=4;cfg.minNestingLevel=1;cfg.numOfNestingLevel=1;
        RangeProfilerTarget target(cu,cfg);std::string chip;CUPTI_API_CALL(target.GetChipName(0,chip));
        std::vector<uint8_t> available,config,data;CUPTI_API_CALL(target.GetCounterAvailabilityImage(cu,available));
        CuptiProfilerHost host;host.SetUp(chip,available);
        std::vector<const char*> metrics={"gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","dram__throughput.avg.pct_of_peak_sustained_elapsed","sm__throughput.avg.pct_of_peak_sustained_elapsed","sm__warps_active.avg.pct_of_peak_sustained_active"};
        CUPTI_API_CALL(host.CreateConfigImage(metrics,config));CUPTI_API_CALL(target.EnableRangeProfiler());
        CUPTI_API_CALL(target.CreateCounterDataImage(metrics,data));CUPTI_API_CALL(target.SetConfig(CUPTI_UserRange,CUPTI_UserReplay,config,data));
        int passes=0;
        do {if(++passes>32)throw std::runtime_error("PMU replay bound");CUPTI_API_CALL(target.StartRangeProfiler());CUPTI_API_CALL(target.PushRange(selected?"candidate":"stock"));for(int n=0;n<3;++n)compute();CUPTI_API_CALL(target.PopRange());CUPTI_API_CALL(target.StopRangeProfiler());}while(!target.IsAllPassSubmitted());
        CUPTI_API_CALL(target.DecodeCounterData());size_t nrange=0;CUPTI_API_CALL(host.GetNumOfRanges(data,nrange));
        for(size_t i=0;i<nrange;++i)CUPTI_API_CALL(host.EvaluateCounterData(i,metrics,data));host.PrintProfilerRanges();
        CUPTI_API_CALL(target.DisableRangeProfiler());host.TearDown();std::printf("PMU_PASSES %d\n",passes);
    }
#endif
    else throw std::runtime_error("unknown mode");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(gpu);ggml_backend_free(cpu);munmap(rawbase,skip+bytes);
    return 0;
} catch(const std::exception &e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
