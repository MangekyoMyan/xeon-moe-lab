#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <omp.h>
#include <sched.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

static constexpr int D=512,I=128,K=512,W=256,H=64;
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
struct Graph {
 ggml_context*ctx=nullptr;ggml_backend_buffer_t buf=nullptr;ggml_cgraph*graph=nullptr;
 ggml_tensor *history=nullptr,*ids=nullptr,*q=nullptr,*raw=nullptr,*selected=nullptr,*mask=nullptr,*sinks=nullptr,*out=nullptr;
 Graph(ggml_backend_t backend,int rows,int mode,ggml_tensor*host=nullptr) {
  ctx=ggml_init({4*1024*1024,nullptr,true});need(ctx,"context");
  if(mode==2){
   ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,K);
   auto view=ggml_view_2d(ctx,host,D,rows,host->nb[1],0);
   out=ggml_cast(ctx,ggml_get_rows(ctx,view,ids),GGML_TYPE_F16);
   graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
   buf=ggml_backend_alloc_ctx_tensors_from_buft(ctx,ggml_backend_cuda_host_buffer_type());need(buf,"CPU pinned outputs");return;
  }
  q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,D,1,H,1);raw=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,D,W);
  mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,W+K,1);sinks=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,H);
  if(mode==0){
   history=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,D+I,rows);ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,K);
   auto view=ggml_view_2d(ctx,history,D,rows,history->nb[1],0);
   selected=ggml_cast(ctx,ggml_get_rows(ctx,view,ids),GGML_TYPE_F16);
  }else selected=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,D,K);
  auto all=ggml_concat(ctx,raw,selected,1);
  out=ggml_flash_attn_ext(ctx,q,all,all,mask,1.f/std::sqrt(float(D)),0,0);
  ggml_prec_set_acc(out,GGML_PREC_F32);ggml_flash_attn_ext_add_sinks(out,sinks);
  need(ggml_backend_supports_op(backend,out),"FlashAttention shape unsupported");
  graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
  buf=ggml_backend_alloc_ctx_tensors(ctx,backend);need(buf,"GPU graph buffer");
 }
 ~Graph(){if(buf)ggml_backend_buffer_free(buf);if(ctx)ggml_free(ctx);}
};

int main()try{
 auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_cuda_init(0);need(cpu&&gpu,"backend");
 ggml_backend_cpu_set_n_threads(cpu,12);auto pp=ggml_threadpool_params_default(12);pp.poll=0;pp.strict_cpu=true;for(int i=0;i<12;++i)pp.cpumask[i]=true;
 auto pool=ggml_threadpool_new(&pp);ggml_backend_cpu_set_threadpool(cpu,pool);
 omp_set_dynamic(0);omp_set_num_threads(12);int errors=0;
 #pragma omp parallel reduction(+:errors)
 {cpu_set_t set;CPU_ZERO(&set);CPU_SET(omp_get_thread_num(),&set);errors+=sched_setaffinity(0,sizeof(set),&set)!=0;}
 need(!errors,"CPU affinity");
 std::mt19937 rng(8721);std::vector<ggml_fp16_t>pattern(1024*(D+I));
 for(auto &x:pattern)x=ggml_fp32_to_fp16(float(int(rng()%2001)-1000)/2048);
 std::vector<float>q(D*H),sinks(H);for(auto &x:q)x=float(int(rng()%2001)-1000)/2048;
 std::vector<ggml_fp16_t>raw(D*W),mask(K+W,ggml_fp32_to_fp16(0));
 for(auto &x:raw)x=ggml_fp32_to_fp16(float(int(rng()%2001)-1000)/2048);
 for(int j=128;j<W;++j)mask[j]=ggml_fp32_to_fp16(-INFINITY);
 for(int j=K+W-8;j<K+W;++j)mask[j]=ggml_fp32_to_fp16(-INFINITY);
 for(int rows:{4096,65536,131072,524288,1048576}) {
  auto ctx=ggml_init({1024*1024,nullptr,true});auto history=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,D+I,rows);
  auto host_buf=ggml_backend_alloc_ctx_tensors(ctx,cpu);need(host_buf,"RAM KV");auto *host=static_cast<ggml_fp16_t*>(history->data);
  for(int row=0;row<rows;++row) {
   auto dst=host+size_t(row)*(D+I);std::memcpy(dst,pattern.data()+size_t(row%1024)*(D+I),(D+I)*sizeof(ggml_fp16_t));
   dst[0]=ggml_fp32_to_fp16(float(row&1023)/1024);dst[1]=ggml_fp32_to_fp16(float((row>>10)&1023)/1024);
  }
  {
   Graph resident(gpu,rows,0),tiered(gpu,rows,1),gather(cpu,rows,2,history);
   ggml_backend_tensor_set(resident.history,host,0,ggml_nbytes(history));
   for(auto *g:{&resident,&tiered}){
    ggml_backend_tensor_set(g->q,q.data(),0,q.size()*4);ggml_backend_tensor_set(g->raw,raw.data(),0,raw.size()*2);
    ggml_backend_tensor_set(g->mask,mask.data(),0,mask.size()*2);ggml_backend_tensor_set(g->sinks,sinks.data(),0,sinks.size()*4);
   }
   std::vector<int32_t>ids(K);std::vector<float>reference(D*H),candidate(D*H);
   std::vector<ggml_fp16_t>expected(K*D),actual(K*D);
   std::vector<double>total[4],idtime[4],gathertime[4],copytime[4],attentiontime[4];
   auto run=[&](int method,bool check){
    double a=now(),b=a,c=a,d=a;
    if(method){
     ggml_backend_tensor_get(resident.ids,gather.ids->data,0,K*4);b=now();
     if(method==1)need(ggml_backend_graph_compute(cpu,gather.graph)==GGML_STATUS_SUCCESS,"CPU gather graph");
     else{
      auto *src_ids=static_cast<const int32_t*>(gather.ids->data);auto *dst=static_cast<ggml_fp16_t*>(gather.out->data);
      if(method==3){
       #pragma omp parallel for schedule(static)
       for(int j=0;j<K;++j)std::memcpy(dst+size_t(j)*D,host+size_t(src_ids[j])*(D+I),D*2);
      }else for(int j=0;j<K;++j)std::memcpy(dst+size_t(j)*D,host+size_t(src_ids[j])*(D+I),D*2);
     }
     c=now();ggml_backend_tensor_set(tiered.selected,gather.out->data,0,K*D*2);d=now();
    }
    auto &g=method?tiered:resident;
    need(ggml_backend_graph_compute(gpu,g.graph)==GGML_STATUS_SUCCESS,"GPU attention");ggml_backend_synchronize(gpu);double end=now();
    if(!check){total[method].push_back(end-a);idtime[method].push_back(b-a);gathertime[method].push_back(c-b);copytime[method].push_back(d-c);attentiontime[method].push_back(end-d);}
    else{
     ggml_backend_tensor_get(g.out,method?candidate.data():reference.data(),0,reference.size()*4);
     for(float value:(method?candidate:reference))need(std::isfinite(value),"Nonfinite Attention output");
     if(method){
      need(!std::memcmp(reference.data(),candidate.data(),reference.size()*4),"Attention bit mismatch");
      for(int j=0;j<K;++j)std::memcpy(expected.data()+size_t(j)*D,host+size_t(ids[j])*(D+I),D*2);
      ggml_backend_tensor_get(tiered.selected,actual.data(),0,actual.size()*2);need(expected==actual,"selected bytes mismatch");
     }
    }
   };
   auto set_ids=[&](int step,int pair,bool clustered){
    if(clustered){int start=(step*547+pair*701)%(rows-K);std::iota(ids.begin(),ids.end(),start);}
    else{
     // Odd affine stride is a permutation for these power-of-two row counts.
     const uint32_t base=rng()%rows;const uint32_t stride=(rng()|1u);
     for(int j=0;j<K;++j)ids[j]=(base+uint32_t(j)*stride)&(rows-1);
     std::sort(ids.begin(),ids.end());
    }
    ggml_backend_tensor_set(resident.ids,ids.data(),0,ids.size()*4);
   };
   for(bool clustered:{false,true}){
    for(int m=0;m<4;++m){total[m].clear();idtime[m].clear();gathertime[m].clear();copytime[m].clear();attentiontime[m].clear();}
    for(int warm=0;warm<3;++warm){set_ids(warm,0,clustered);if(warm==0){ids.front()=0;ids.back()=rows-1;ggml_backend_tensor_set(resident.ids,ids.data(),0,ids.size()*4);}for(int m=0;m<4;++m)run(m,true);}
    for(int repeat=0;repeat<8;++repeat)for(int pair=0;pair<8;++pair){
     set_ids(repeat,pair,clustered);std::vector<int>order{0,1,2,3};std::rotate(order.begin(),order.begin()+repeat%4,order.end());
     for(int method:order)run(method,false);std::reverse(order.begin(),order.end());for(int method:order)run(method,false);
     for(int method=0;method<4;++method)run(method,true);
    }
    for(int method=0;method<4;++method)std::printf("{\"rows\":%d,\"pattern\":\"%s\",\"method\":%d,\"median_pair_ms\":%.6f,\"eight_pair_serial_extrapolation_ms\":%.6f,\"ids_D2H_ms\":%.6f,\"CPU_gather_ms\":%.6f,\"KV_H2D_ms\":%.6f,\"GPU_graph_ms\":%.6f,\"selected_payload_bytes_per_pair\":%d,\"H2D_bytes_per_pair\":%d,\"D2H_bytes_per_pair\":%d,\"verified_ID_sets\":67,\"bit_mismatches\":0}\n",rows,clustered?"clustered":"spread",method,median(total[method]),8*median(total[method]),median(idtime[method]),median(gathertime[method]),median(copytime[method]),median(attentiontime[method]),K*D*2,method?K*D*2:0,method?K*4:0);
    std::fflush(stdout);
   }
  }
  ggml_backend_buffer_free(host_buf);ggml_free(ctx);
 }
 ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(cpu);ggml_backend_free(gpu);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
