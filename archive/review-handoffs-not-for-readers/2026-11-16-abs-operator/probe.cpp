#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "gguf.h"
#include "iq3-cache.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>

static void need(bool ok,const char *msg){if(!ok)throw std::runtime_error(msg);}
static void (*set_mode)(int)=nullptr;static void (*diag)(int)=nullptr;static void (*counts)(uint64_t*)=nullptr;
static void mode(int value){if(set_mode)set_mode(value);else need(value==0,"mode unavailable");}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static ggml_threadpool_t pool(int threads){auto p=ggml_threadpool_params_default(threads);p.poll=0;p.strict_cpu=true;for(int i=0;i<threads;++i)p.cpumask[i]=true;auto result=ggml_threadpool_new(&p);need(result,"threadpool");return result;}
static void write(FILE *f,const void*p,size_t n){need(std::fwrite(p,1,n,f)==n,"write");}
static size_t cases=0,values=0;

static void validate_case(FILE *gold,bool read_gold,ggml_backend_t cpu,ggml_type type,int k,int inputs,int tokens,bool strided,bool f32,bool duplicate){
    const int used=std::max(6,inputs),experts=std::max(8,used+1),M=19;
    auto ctx=ggml_init({4*1024*1024,nullptr,true});need(ctx,"ctx");
    auto w=ggml_new_tensor_3d(ctx,type,k,M,experts);ggml_set_name(w,"abs_fixture.weight");
    auto xt=f32?GGML_TYPE_F32:GGML_TYPE_Q8_K;size_t qrow=ggml_row_size(xt,k);
    auto backing=ggml_new_tensor_3d(ctx,xt,k,strided?inputs*2:inputs,tokens);
    auto x=strided?ggml_view_3d(ctx,backing,k,inputs,tokens,qrow*2,qrow*inputs*2,0):backing;
    auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,used,tokens);auto y=ggml_mul_mat_id(ctx,w,x,ids);
    auto g=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(g,y);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buffer,"buffer");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(76391+type*37+k+inputs*11+tokens+strided+f32+duplicate);
    std::vector<uint8_t>weights(ggml_nbytes(w));for(auto&v:weights)v=rng();
    for(size_t b=0;b<weights.size();b+=ggml_type_size(type)){uint16_t d=0x2400;if(type==GGML_TYPE_MXFP4)weights[b]=127;else std::memcpy(weights.data()+b,&d,2);}
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());
    std::vector<int32_t>routes(used*tokens);for(int t=0;t<tokens;++t)for(int i=0;i<used;++i)routes[t*used+i]=(i+t+1)%experts;
    if(duplicate){routes[1]=routes[0];routes[5]=-1;}ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*4);
    std::vector<uint32_t>all(experts);std::iota(all.begin(),all.end(),0);
    for(int selection=0;selection<(type==GGML_TYPE_MXFP4?1:3);++selection){
        const auto selected=selection==0?std::vector<uint32_t>{}:selection==1?all:std::vector<uint32_t>{1,3,5};
        void*owner=nullptr;
        if(!selected.empty()){owner=xeon_iq3_cache_create_v1(32*1024*1024,0,12);need(owner,"owner");need(xeon_iq3_cache_add_selected_v2(owner,w,selected.data(),selected.size())==1,"cache add");}
        for(int threads:{12,1,24}){
            auto tp=pool(threads);ggml_backend_cpu_set_n_threads(cpu,threads);ggml_backend_cpu_set_threadpool(cpu,tp);
            auto plan=ggml_graph_plan(g,threads,tp);
            std::vector<uint8_t>workspace(plan.work_size+256,0xa5);
            auto address=(reinterpret_cast<uintptr_t>(workspace.data())+63)&~uintptr_t(63);address+=16;
            plan.work_data=reinterpret_cast<uint8_t*>(address);size_t begin=plan.work_data-workspace.data(),end=begin+plan.work_size;
            need(end<=workspace.size(),"workspace sizing");
            for(int phase=0;phase<5;++phase){
                std::vector<uint8_t>data(ggml_nbytes(backing));std::vector<bool>unsafe(inputs,false);
                for(size_t offset=0;offset<data.size();offset+=qrow){
                    if(f32){auto *v=reinterpret_cast<float*>(data.data()+offset);for(int i=0;i<k;++i)v[i]=float(int(rng()%2001)-1000)/2048;}
                    else{auto *v=reinterpret_cast<block_q8_K*>(data.data()+offset);for(int b=0;b<k/256;++b){v[b].d=.00390625f;for(auto&q:v[b].qs)q=int(rng()%255)-127;}
                        int input=int(offset/qrow)/(strided?2:1)%inputs;
                        if(phase==1||(phase==3&&(input%3==1||input==63))){unsafe[input]=true;if(input%2)v[k/256-1].d=std::numeric_limits<float>::infinity();else v[k/256-1].qs[255]=-128;}
                        for(int b=0;b<k/256;++b)for(int j=0;j<16;++j){int sum=0;for(int i=0;i<16;++i)sum+=v[b].qs[j*16+i];v[b].bsums[j]=sum;}
                    }
                }
                ggml_backend_tensor_set(backing,data.data(),0,data.size());
                for(bool reference:{false,true}){
                    plan.use_ref=reference;std::vector<float>expected(ggml_nelements(y)),actual(expected.size());
                    uint64_t header[]={uint64_t(type),uint64_t(k),uint64_t(inputs),uint64_t(tokens),uint64_t(strided),uint64_t(f32),uint64_t(duplicate),uint64_t(selection),uint64_t(threads),uint64_t(phase),uint64_t(reference),expected.size()};
                    if(read_gold){uint64_t stored[12];need(std::fread(stored,sizeof(stored),1,gold)==1&&!std::memcmp(stored,header,sizeof(header)),"golden metadata");need(std::fread(expected.data(),4,expected.size(),gold)==expected.size(),"golden bytes");}
                    for(int arm=0;arm<(set_mode?3:1);++arm){
                        mode(arm);if(diag)diag(1);uint64_t before[3]{},after[3]{};if(counts)counts(before);
                        need(ggml_graph_compute(g,&plan)==GGML_STATUS_SUCCESS,"graph compute");if(counts)counts(after);
                        ggml_backend_tensor_get(y,actual.data(),0,ggml_nbytes(y));
                        if(!read_gold&&arm==0){expected=actual;write(gold,header,sizeof(header));write(gold,expected.data(),expected.size()*4);}
                        need(!std::memcmp(expected.data(),actual.data(),expected.size()*4),"bit mismatch versus frozen final");
                        need(std::all_of(workspace.begin(),workspace.begin()+begin,[](uint8_t v){return v==0xa5;})&&std::all_of(workspace.begin()+end,workspace.end(),[](uint8_t v){return v==0xa5;}),"workspace canary");
                        bool active=arm!=0&&!reference&&tokens==1&&inputs<=64&&(type==GGML_TYPE_IQ2_S||type==GGML_TYPE_IQ3_S);
                        uint64_t expected_rows=0;for(int id=0;id<used;++id)if(routes[id]>=0&&std::find(selected.begin(),selected.end(),uint32_t(routes[id]))!=selected.end()&&!unsafe[id%inputs])expected_rows+=M;
                        if(counts){need(after[1]-before[1]==(active?expected_rows:0),"dispatch row count");if(!active||(arm==2&&selected.empty()))need(after[0]==before[0],"unexpected prepare");if(active&&arm==1)need(after[0]-before[0]==uint64_t(k/256)*inputs,"shared partition/empty-worker count");}
                        ++cases;values+=actual.size();
                    }
                }
            }
            ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(tp);
        }
        if(owner){xeon_iq3_cache_free_v1(owner);}
        need(xeon_iq3_cache_total_v1()==0,"cache leak");
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}

static void benchmark(ggml_backend_t cpu,const char*shard,const char*tensor,const char*directory){
    need(set_mode&&diag&&counts,"experimental API required");
    auto meta=gguf_init_from_file(shard,{true,nullptr});need(meta,"metadata");auto ti=gguf_find_tensor(meta,tensor);need(ti>=0,"tensor");auto ne=gguf_get_tensor_ne(meta,ti);auto type=gguf_get_tensor_type(meta,ti);
    int K=ne[0],M=ne[1],E=ne[2];need((type==GGML_TYPE_IQ2_S||type==GGML_TYPE_IQ3_S)&&E==384&&ne[3]==1&&((K==5120&&M==2304)||(K==2304&&M==5120)),"real shape/type");
    int fd=open(shard,O_RDONLY|O_CLOEXEC);need(fd>=0,"open shard");struct stat st{};need(!fstat(fd,&st),"stat");size_t offset=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,ti),bytes=ggml_row_size(type,K)*M*E;need(offset<=size_t(st.st_size)&&bytes<=size_t(st.st_size)-offset,"bounds");
    void*map=mmap(nullptr,st.st_size,PROT_READ,MAP_SHARED,fd,0);close(fd);need(map!=MAP_FAILED,"mmap");
    auto ctx=ggml_init({8*1024*1024,nullptr,true});auto w=ggml_new_tensor_3d(ctx,type,K,M,E);ggml_set_name(w,tensor);auto wb=ggml_backend_cpu_buffer_from_ptr(map,st.st_size);need(wb,"weights");w->buffer=wb;w->data=static_cast<char*>(map)+offset;ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,K,K==5120?1:6,1);auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,1);auto y=ggml_mul_mat_id(ctx,w,x,ids);auto g=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(g,y);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(buffer,"activations");
    auto tp=pool(12);ggml_backend_cpu_set_n_threads(cpu,12);ggml_backend_cpu_set_threadpool(cpu,tp);ggml_backend_cpu_set_use_ref(cpu,false);
    std::mt19937 rng(20261116);std::vector<uint32_t>permutation(E);std::iota(permutation.begin(),permutation.end(),0);std::shuffle(permutation.begin(),permutation.end(),rng);
    std::vector<std::vector<float>>input(8,std::vector<float>(ggml_nelements(x)));for(auto&r:input)for(auto&v:r)v=float(int(rng()%2001)-1000)/1024;
    std::filesystem::create_directories(directory);FILE*f=std::fopen((std::string(directory)+"/inputs.bin").c_str(),"wb");need(f,"inputs");for(auto&r:input)write(f,r.data(),r.size()*4);std::fclose(f);
    std::vector<float>expected(ggml_nelements(y)),out(expected.size());
    std::vector<int>pins(12,-1);
    for(int hits:{6,0}){
        std::vector<uint32_t>selected;if(hits)selected.assign(permutation.begin(),permutation.begin()+48);
        void*owner=xeon_iq3_cache_create_v1(304ULL*1024*1024,64ULL*1024*1024*1024,12);need(owner,"cache owner");need(xeon_iq3_cache_add_selected_v2(owner,w,selected.data(),selected.size())==(hits?1:0),"pack selected");
        auto compute=[&](){need(ggml_backend_graph_compute(cpu,g)==GGML_STATUS_SUCCESS,"operator compute");};
        for(int r=0;r<8;++r){int32_t route[6];for(int j=0;j<6;++j)route[j]=permutation[6*r+j];ggml_backend_tensor_set(ids,route,0,sizeof(route));ggml_backend_tensor_set(x,input[r].data(),0,input[r].size()*4);
            mode(0);diag(0);compute();ggml_backend_tensor_get(y,expected.data(),0,ggml_nbytes(y));
            for(int a=0;a<3;++a){mode(a);diag(1);uint64_t before[3],after[3];counts(before);compute();counts(after);ggml_backend_tensor_get(y,out.data(),0,ggml_nbytes(y));need(!std::memcmp(out.data(),expected.data(),out.size()*4),"warm bit mismatch");need(after[1]-before[1]==uint64_t(a&&hits?6*M:0),"warm dispatch");}diag(0);
        }
        // Inspect the retained OpenMP worker affinities after ggml warmup.
        #pragma omp parallel num_threads(12)
        { pins[omp_get_thread_num()]=sched_getcpu(); }
        auto sorted=pins;std::sort(sorted.begin(),sorted.end());for(int i=0;i<12;++i)need(sorted[i]==i,"physical worker affinity");
        std::printf("{\"kind\":\"operator\",\"tensor\":\"%s\",\"type\":\"%s\",\"K\":%d,\"M\":%d,\"inputs\":%lld,\"threads\":12,\"ggml_numa\":%s,\"cached_of_six\":%d,\"cache_bytes\":%zu,\"pairs\":[",tensor,ggml_type_name(type),K,M,(long long)x->ne[1],ggml_is_numa()?"true":"false",hits,xeon_iq3_cache_used_v1(owner));
        bool first=true;
        for(int block=0;block<16;++block){int r=block%8;int32_t route[6];for(int j=0;j<6;++j)route[j]=permutation[r*6+j];ggml_backend_tensor_set(ids,route,0,sizeof(route));ggml_backend_tensor_set(x,input[r].data(),0,input[r].size()*4);
            mode(0);compute();ggml_backend_tensor_get(y,expected.data(),0,ggml_nbytes(y));
            const bool reverse=(block%8+block/8)%2;
            for(int which=0;which<2;++which){int candidate=reverse?2-which:1+which;int order[4]={0,candidate,candidate,0};if(reverse){order[0]=candidate;order[1]=0;order[2]=0;order[3]=candidate;}
                for(int index=0;index<4;++index){int a=order[index];mode(a);compute();const double begin=now();for(int repeat=0;repeat<8;++repeat)compute();double elapsed=(now()-begin)/8;
                    ggml_backend_tensor_get(y,out.data(),0,ggml_nbytes(y));need(!std::memcmp(out.data(),expected.data(),out.size()*4),"timed output mismatch");need(std::isfinite(elapsed)&&elapsed>0,"time");
                    std::printf("%s{\"block\":%d,\"route_group\":%d,\"candidate\":%d,\"order\":%d,\"mode\":%d,\"ms\":%.9f}",first?"":",",block,r,candidate,index,a,elapsed);first=false;
                }
            }
        }
        std::printf("],\"routes\":[");for(int i=0;i<48;++i)std::printf("%s%u",i?",":"",permutation[i]);std::printf("],\"pins\":[");for(int i=0;i<12;++i)std::printf("%s%d",i?",":"",pins[i]);std::puts("],\"bit_mismatches\":0}");std::fflush(stdout);
        xeon_iq3_cache_free_v1(owner);need(xeon_iq3_cache_total_v1()==0,"cache leak");
    }
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(tp);ggml_backend_buffer_free(buffer);ggml_backend_buffer_free(wb);ggml_free(ctx);munmap(map,st.st_size);gguf_free(meta);
}
int main(int argc,char**argv)try{
    need(argc>=4,"validate GOLD LIBRARY | bench SHARD TENSOR DIR LIBRARY");std::setvbuf(stdout,nullptr,_IOLBF,0);
    Dl_info info{};need(dladdr(reinterpret_cast<void*>(ggml_backend_cpu_init),&info),"provider");need(std::filesystem::canonical(info.dli_fname)==std::filesystem::canonical(argv[argc-1]),"wrong CPU library");
    set_mode=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"xeon_iq_abs_set_mode"));diag=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"xeon_iq_abs_set_diagnostics"));counts=reinterpret_cast<void(*)(uint64_t*)>(dlsym(RTLD_DEFAULT,"xeon_iq_abs_counts"));
    auto cpu=ggml_backend_cpu_init();need(cpu,"CPU");
    if(std::string(argv[1])=="validate"){
        need(argc==4,"validate args");bool read_gold=set_mode!=nullptr;FILE*f=std::fopen(argv[2],read_gold?"rb":"wb");need(f,"gold file");
        for(auto type:{GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_S})for(int k:{256,2304,5120})for(int inputs:{1,6})for(bool strided:{false,true})for(bool fp32:{false,true})validate_case(f,read_gold,cpu,type,k,inputs,1,strided,fp32,false);
        for(auto type:{GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_S}){
            for(int inputs:{64,65})for(bool fp32:{false,true})validate_case(f,read_gold,cpu,type,256,inputs,1,true,fp32,true);
            validate_case(f,read_gold,cpu,type,256,6,3,true,false,true);
            validate_case(f,read_gold,cpu,type,2304,6,1,true,false,true);
        }
        validate_case(f,read_gold,cpu,GGML_TYPE_IQ3_XXS,256,6,1,true,false,true);
        for(int k:{256,2304})for(int inputs:{1,6})for(int tokens:{1,3})validate_case(f,read_gold,cpu,GGML_TYPE_MXFP4,k,inputs,tokens,true,true,true);
        if(read_gold){need(std::fgetc(f)==EOF,"extra golden data");}
        std::fclose(f);
        std::printf("{\"passed\":true,\"graph_executions\":%zu,\"checked_outputs\":%zu,\"experimental\":%s,\"work_data_mod64\":16}\n",cases,values,read_gold?"true":"false");
    }else{need(std::string(argv[1])=="bench"&&argc==6,"bench args");benchmark(cpu,argv[2],argv[3],argv[4]);std::puts("{\"passed\":true}");}
    ggml_backend_free(cpu);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
