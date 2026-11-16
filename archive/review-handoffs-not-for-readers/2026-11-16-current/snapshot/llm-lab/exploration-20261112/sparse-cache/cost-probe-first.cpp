#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "iq3-cache.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>

static void need(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static volatile uint64_t evict_sink=0;

int main(int argc,char ** argv)try {
    need(argc==3,"shard tensor");
    auto meta=gguf_init_from_file(argv[1],{true,nullptr});need(meta,"GGUF metadata");
    const auto ti=gguf_find_tensor(meta,argv[2]);need(ti>=0,"tensor missing");
    const auto type=gguf_get_tensor_type(meta,ti);const auto ne=gguf_get_tensor_ne(meta,ti);
    need(type==GGML_TYPE_IQ3_XXS||type==GGML_TYPE_IQ2_S||type==GGML_TYPE_IQ3_S,"IQ tensor required");
    const int k=ne[0],m=ne[1],experts=ne[2],used=6,rounds=8;
    need((k==5120&&m==2304)||(k==2304&&m==5120),"unexpected real dimensions");
    need(ne[0]==k&&ne[1]==m&&ne[2]==384&&ne[3]==1,"384 expert bank required");
    const size_t offset=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,ti);
    const size_t weight_bytes=ggml_row_size(type,k)*m*experts;
    const int fd=open(argv[1],O_RDONLY|O_CLOEXEC);need(fd>=0,"open");
    struct stat st{};need(!fstat(fd,&st)&&S_ISREG(st.st_mode)&&st.st_size>0,"file stat");
    const size_t file_bytes=st.st_size;need(offset<=file_bytes&&weight_bytes<=file_bytes-offset,"tensor file bounds");
    void * mapping=mmap(nullptr,file_bytes,PROT_READ,MAP_SHARED,fd,0);close(fd);need(mapping!=MAP_FAILED,"read-only mmap");
    auto cpu=ggml_backend_cpu_init();need(cpu,"CPU");ggml_backend_cpu_set_n_threads(cpu,12);
    auto params=ggml_threadpool_params_default(12);params.poll=0;params.strict_cpu=true;
    for(int i=0;i<12;++i)params.cpumask[i]=true;
    auto pool=ggml_threadpool_new(&params);ggml_backend_cpu_set_threadpool(cpu,pool);
    auto ctx=ggml_init({8*1024*1024,nullptr,true});need(ctx,"context");
    auto w=ggml_new_tensor_3d(ctx,type,k,m,experts);ggml_set_name(w,argv[2]);
    auto weights_buffer=ggml_backend_cpu_buffer_from_ptr(mapping,file_bytes);need(weights_buffer,"external weights");
    ggml_backend_buffer_set_usage(weights_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    w->buffer=weights_buffer;w->data=static_cast<char *>(mapping)+offset;
    const bool shared=k==5120;
    auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,shared?1:used,1);
    auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,1);
    auto y=ggml_mul_mat_id(ctx,w,x,ids);
    auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,y);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buffer,"activation buffer");
    std::mt19937 rng(49153);
    std::vector<uint32_t> permutation(experts);std::iota(permutation.begin(),permutation.end(),0);std::shuffle(permutation.begin(),permutation.end(),rng);
    std::vector<std::vector<float>> inputs(rounds,std::vector<float>(ggml_nelements(x)));
    for(auto & input:inputs)for(auto & value:input)value=float(int(rng()%2001)-1000)/1024;
    std::vector<float> expected(ggml_nelements(y)),out(expected.size());
    std::vector<uint64_t> eviction(128*1024*1024/sizeof(uint64_t),1);
    auto evict=[&]() {
        uint64_t sum=0;
        #pragma omp parallel for schedule(static) num_threads(12) reduction(+:sum)
        for(size_t i=0;i<eviction.size();i+=8) {
            if(i==0 && omp_get_num_threads()!=12)std::abort();
            sum+=*reinterpret_cast<volatile uint64_t *>(&eviction[i]);
        }
        evict_sink=sum;
    };
    auto compute=[&](std::vector<float>& dst){
        need(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"compute");
        ggml_backend_tensor_get(y,dst.data(),0,dst.size()*4);
    };
    const size_t per_expert=size_t(k/256)*m*144;
    for(int hits=0;hits<=used;++hits) {
        std::vector<uint32_t> selected;
        for(int r=0;r<rounds;++r)for(int i=0;i<hits;++i)selected.push_back(permutation[r*used+i]);
        std::sort(selected.begin(),selected.end());
        const size_t bytes=(selected.size()*per_expert+4095)/4096*4096;
        void * owner=xeon_iq3_cache_create_v1(std::max(size_t(1),bytes),0,12);need(owner,"owner");
        need(xeon_iq3_cache_add_selected_v2(owner,w,selected.data(),selected.size())==(hits?1:0),"selected add");
        std::vector<double> raw,mixed,ratios;
        std::printf("{\"type\":\"%s\",\"tensor\":\"%s\",\"k\":%d,\"m\":%d,\"experts\":384,\"threads\":12,\"ggml_numa\":%s,\"eviction_MiB\":128,\"cached_of_six\":%d,\"samples\":[",ggml_type_name(type),argv[2],k,m,ggml_is_numa()?"true":"false",hits);
        bool first=true;
        for(int r=0;r<rounds;++r) {
            int32_t routes[used];for(int i=0;i<used;++i)routes[i]=permutation[r*used+i];
            ggml_backend_tensor_set(ids,routes,0,sizeof(routes));ggml_backend_tensor_set(x,inputs[r].data(),0,inputs[r].size()*4);
            ggml_backend_cpu_set_use_ref(cpu,true);compute(expected);
            ggml_backend_cpu_set_use_ref(cpu,false);compute(out);need(!std::memcmp(expected.data(),out.data(),out.size()*4),"warmup bit mismatch");
            double group_raw=0,group_mixed=0;
            for(int arm:{0,1,1,0}) {
                ggml_backend_cpu_set_use_ref(cpu,arm==0);
                evict();
                const double start=now();compute(out);const double elapsed=now()-start;
                need(!std::memcmp(expected.data(),out.data(),out.size()*4),"timed bit mismatch");
                if(arm){mixed.push_back(elapsed);group_mixed+=elapsed;}else{raw.push_back(elapsed);group_raw+=elapsed;}
                std::printf("%s{\"round\":%d,\"arm\":%d,\"ms\":%.6f}",first?"":",",r,arm,elapsed);first=false;
            }
            ratios.push_back(group_raw/group_mixed);
        }
        std::printf("],\"raw_median_ms\":%.6f,\"mixed_median_ms\":%.6f,\"paired_ratio_median\":%.6f,\"saved_ms_per_selected_expert\":%.9f,\"bit_mismatches\":0}\n",median(raw),median(mixed),median(ratios),hits?(median(raw)-median(mixed))/hits:0);
        std::fflush(stdout);ggml_backend_cpu_set_use_ref(cpu,false);xeon_iq3_cache_free_v1(owner);
        need(xeon_iq3_cache_total_v1()==0,"cache leak");
    }
    ggml_backend_buffer_free(buffer);ggml_backend_buffer_free(weights_buffer);ggml_free(ctx);
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);
    munmap(mapping,file_bytes);gguf_free(meta);std::puts("{\"pass\":true}");
}catch(const std::exception & error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
