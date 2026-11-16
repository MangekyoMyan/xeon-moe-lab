#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <set>
#include <unistd.h>

// Existing exported selection predicates: verify that the intended path executes.
bool ggml_cuda_should_use_mmq(ggml_type,int,int64_t,int64_t);
bool ggml_cuda_should_use_mmf(ggml_type,int,int,const int64_t*,const size_t*,int,bool);
int get_mmvq_mmid_max_batch(ggml_type,int);
static void need(bool v,const char*m){if(!v)throw std::runtime_error(m);}
static void ck(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
static void hash(const void *data,size_t bytes,char *hex){unsigned char digest[SHA256_DIGEST_LENGTH];need(SHA256(static_cast<const unsigned char*>(data),bytes,digest)!=nullptr,"SHA256 failed");for(size_t i=0;i<sizeof(digest);++i)std::sprintf(hex+2*i,"%02x",digest[i]);hex[64]=0;}
struct scenario {bool small;int tokens,columns,layout;bool compute;};
int main(int argc,char**argv)try{
 need(argc==5,"usage: test-gpu SHARD TENSOR OUTPUT_JSONL EXPECTED_CUDA_LIBRARY");
 Dl_info provider{};need(dladdr(reinterpret_cast<void*>(ggml_backend_cuda_init),&provider),"provider");need(std::filesystem::canonical(provider.dli_fname)==std::filesystem::canonical(argv[4]),"wrong CUDA library");
 need(!dlsym(RTLD_DEFAULT,"xeon_grouped_mode"),"Unexpected grouped-MMQ shim");
 auto meta=gguf_init_from_file(argv[1],{true,nullptr});need(meta,"GGUF metadata");int64_t index=gguf_find_tensor(meta,argv[2]);need(index>=0,"tensor absent");
 const auto type=gguf_get_tensor_type(meta,index);const auto*ne=gguf_get_tensor_ne(meta,index);const int64_t original_k=ne[0],original_m=ne[1];need(ne[2]==384&&ne[3]==1&&((original_k==5120&&original_m==2304)||(original_k==2304&&original_m==5120)),"expected real model geometry");
 need(type==GGML_TYPE_IQ2_S||type==GGML_TYPE_IQ3_XXS||type==GGML_TYPE_IQ3_S||type==GGML_TYPE_MXFP4,"type");
 const size_t original_row=ggml_row_size(type,original_k),original_expert=original_row*original_m;
 const size_t offset=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,index),page=sysconf(_SC_PAGESIZE),aligned=offset/page*page,skip=offset-aligned;
 const size_t mapping_bytes=skip+original_expert*384;int fd=open(argv[1],O_RDONLY);need(fd>=0,"open model");struct stat st{};need(fstat(fd,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>0&&aligned<=size_t(st.st_size)&&mapping_bytes<=size_t(st.st_size)-aligned,"tensor range exceeds regular shard");auto mapping=mmap(nullptr,mapping_bytes,PROT_READ,MAP_SHARED,fd,aligned);close(fd);need(mapping!=MAP_FAILED,"map model");const char*raw=static_cast<const char*>(mapping)+skip;gguf_free(meta);
 ck(cudaSetDevice(0));cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));const int cc=100*prop.major+10*prop.minor;need(cc==700&&prop.warpSize==32,"audited Volta target required");
 size_t free,total;ck(cudaMemGetInfo(&free,&total));need(free>size_t(4)*1024*1024*1024,"Stop the API before bounded GPU tests");
 auto cpu=ggml_backend_cpu_init();auto gpu=ggml_backend_cuda_init(0);need(cpu&&gpu,"backends");FILE*file=std::fopen(argv[3],"wx");need(file,"new output manifest");
 const int normal=std::strstr(argv[2],"down_exps")?6:1;
 const scenario cases[]={{false,63,normal,0,false},{false,64,normal,1,false},{false,65,normal,1,true},{false,128,2,2,false},{false,128,3,1,true},{false,1024,normal,1,false},{false,1536,normal,1,true},{true,63,normal,1,false},{true,64,1,2,true},{true,65,6,1,false},{true,2048,normal,2,true}};
 size_t outputs=0;int count=0;
 for(const auto &test:cases){
  const int E=test.small?384:24,K=test.small?256:original_k,M=test.small?17:original_m,T=test.tokens,C=test.columns;const size_t row=ggml_row_size(type,K),expert=row*M;
  std::vector<uint8_t>quant(expert*E);for(int e=0;e<E;++e)for(int m=0;m<M;++m)std::memcpy(quant.data()+e*expert+m*row,raw+e*original_expert+m*original_row,row);
  auto wctx=ggml_init({1024*1024,nullptr,true});auto ctx=ggml_init({4*1024*1024,nullptr,true});need(wctx&&ctx,"contexts");
  auto weights=ggml_new_tensor_3d(wctx,type,K,M,E);auto input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,K,C,T);
  const size_t id_nb0=test.layout==2?12:4,id_nb1=E*id_nb0+16,id_offset=test.layout?8:0;
  auto backing=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,(id_offset+T*id_nb1+3)/4);
  auto ids=ggml_view_2d(ctx,backing,6,T,id_nb1,id_offset);ids->nb[0]=id_nb0;
  auto output=ggml_mul_mat_id(ctx,weights,input,ids);auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,output);
  auto wbuf=ggml_backend_alloc_ctx_tensors(wctx,gpu);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);need(wbuf&&buffer,"GPU buffers");ggml_backend_buffer_set_usage(wbuf,test.compute?GGML_BACKEND_BUFFER_USAGE_COMPUTE:GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);ggml_backend_buffer_clear(wbuf,0);ggml_backend_buffer_clear(buffer,0);
  need(ggml_backend_supports_op(gpu,output),"unsupported graph");
  need(get_mmvq_mmid_max_batch(type,cc)==8,"MMVQ predicate changed");need(ggml_cuda_should_use_mmq(type,cc,T,E)==(T<64),"MMQ predicate changed");need(!ggml_cuda_should_use_mmf(type,cc,32,weights->ne,weights->nb,T,true),"MMF predicate changed");
  std::mt19937 rng(5986+K+M+T+E+C+test.layout);std::uniform_real_distribution<float> dist(-.5f,.5f);std::vector<float>x(ggml_nelements(input));for(auto&v:x)v=dist(rng);
  std::vector<uint8_t>routing(ggml_nbytes(backing),0xff);std::set<int> seen;
  for(int t=0;t<T;++t)for(int slot=0;slot<6;++slot){int32_t e=(t%4==0)?((t/4)*6+slot)%E:slot;seen.insert(e);std::memcpy(routing.data()+id_offset+t*id_nb1+slot*id_nb0,&e,4);}
  if(E==24||T==2048)need(seen.size()==size_t(E),"routing failed to cover full expert bank");
  ggml_backend_tensor_set(weights,quant.data(),0,quant.size());ggml_backend_tensor_set(input,x.data(),0,x.size()*4);ggml_backend_tensor_set(backing,routing.data(),0,routing.size());
  auto compute=[&]{need(ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS,"GPU compute");ggml_backend_synchronize(gpu);};
  std::vector<float>a(ggml_nelements(output)),b(a.size());compute();compute();ggml_backend_tensor_get(output,a.data(),0,a.size()*4);compute();ggml_backend_tensor_get(output,b.data(),0,b.size()*4);
  need(std::memcmp(a.data(),b.data(),a.size()*4)==0,"non-repeatable GPU output");for(float v:a)need(std::isfinite(v),"nonfinite GPU output");char sha[65];hash(a.data(),a.size()*4,sha);
  std::fprintf(file,"{\"type\":\"%s\",\"k\":%d,\"m\":%d,\"experts\":%d,\"tokens\":%d,\"input_columns\":%d,\"id_column_stride\":%zu,\"id_row_stride\":%zu,\"id_offset\":%zu,\"distinct_experts\":%zu,\"compute_buffer\":%s,\"fallback_verified\":%s,\"outputs\":%zu,\"repeat_byte_identical\":true,\"sha256\":\"%s\"}\n",ggml_type_name(type),K,M,E,T,C,id_nb0,id_nb1,id_offset,seen.size(),test.compute?"true":"false",T>=64?"true":"false",a.size(),sha);std::fflush(file);outputs+=a.size();++count;
  ggml_backend_buffer_free(buffer);ggml_backend_buffer_free(wbuf);ggml_free(ctx);ggml_free(wctx);
 }
 need(std::fclose(file)==0,"close");ggml_backend_free(gpu);ggml_backend_free(cpu);munmap(mapping,mapping_bytes);std::printf("{\"pass\":true,\"cases\":%d,\"outputs\":%zu,\"tensor\":\"%s\"}\n",count,outputs,argv[2]);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
