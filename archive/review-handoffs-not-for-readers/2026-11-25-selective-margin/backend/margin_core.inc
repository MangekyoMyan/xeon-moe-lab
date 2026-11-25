// An isolated, single-model study. Scratch has distinct ownership for gate and up.
// The ordinary DSO is not linked to this code.
#include "margin_shared.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
struct margin_scratch {
    int * host_indices=nullptr, * device_indices=nullptr;
    float * x=nullptr, * y=nullptr;
};
struct margin_session {
    margin_row * device=nullptr, * host=nullptr;
    margin_scratch projection[2];
    float cut=0;
    int n=0, policy=0, planned[2]{}, executed[2]{}, calls[2]{};
    bool host_ready=false;
};
static margin_session marginal;
extern "C" GGML_BACKEND_API void xeon_margin_open(float threshold) {
    GGML_ASSERT(marginal.device==nullptr && std::isfinite(threshold) && threshold>=0);
    marginal.cut=threshold;
    CUDA_CHECK(cudaMalloc(&marginal.device,size_t(40)*128*sizeof(margin_row)));
    CUDA_CHECK(cudaMallocHost(&marginal.host,128*sizeof(margin_row)));
    for(auto &p:marginal.projection){
        CUDA_CHECK(cudaMallocHost(&p.host_indices,128*6*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&p.device_indices,128*6*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&p.x,128*5120*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p.y,128*2304*sizeof(float)));
    }
}
// The caller synchronizes and resets the sequence before arming another prefix.
// Poison the observation buffer outside the timing clock to detect stale receipts.
extern "C" GGML_BACKEND_API void xeon_margin_begin(int policy,int n) {
    GGML_ASSERT(marginal.device && policy>=1 && policy<=4 && n>0);
    marginal.policy=policy;marginal.n=n;marginal.host_ready=false;
    for(int i=0;i<2;++i){marginal.planned[i]=marginal.executed[i]=marginal.calls[i]=0;}
    CUDA_CHECK(cudaMemset(marginal.device,0xff,40*128*sizeof(margin_row)));
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" GGML_BACKEND_API void xeon_margin_receipt(margin_row *out,int *counts) {
    CUDA_CHECK(cudaMemcpy(out,marginal.device,40*128*sizeof(margin_row),cudaMemcpyDeviceToHost));
    for(int i=0;i<2;++i){counts[i]=marginal.calls[i];counts[2+i]=marginal.planned[i];counts[4+i]=marginal.executed[i];}
    marginal.n=0;
}
extern "C" GGML_BACKEND_API void xeon_margin_close() {
    CUDA_CHECK(cudaDeviceSynchronize());
    for(auto &p:marginal.projection){CUDA_CHECK(cudaFreeHost(p.host_indices));CUDA_CHECK(cudaFree(p.device_indices));CUDA_CHECK(cudaFree(p.x));CUDA_CHECK(cudaFree(p.y));}
    CUDA_CHECK(cudaFreeHost(marginal.host));CUDA_CHECK(cudaFree(marginal.device));marginal={};
}

margin_row * xeon_margin_rows_for(const char *name,const char *prefix,int n,float *cut) {
    if(!marginal.device || n<64 || n>=128)return nullptr;
    const size_t len=std::strlen(prefix);
    if(std::strncmp(name,prefix,len)!=0)return nullptr;
    char *end=nullptr;const long l=std::strtol(name+len,&end,10);
    if(end==name+len||*end||l<0||l>=40)return nullptr;
    *cut=marginal.cut;
    return marginal.device+l*128;
}
static __global__ void margin_sorted(const float *scores,const int *ids,margin_row *out,int n,float cut) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n)return;
    scores+=t*384;ids+=t*384;
    const int sixth=ids[5],seventh=ids[6];
    const float d=scores[sixth]-scores[seventh];
    out[t]={sixth,seventh,d,d<=cut};
}
static void margin_after_argsort(ggml_backend_cuda_context &ctx,ggml_tensor *dst){
    if(dst->ne[0]!=384||dst->ne[2]!=1||dst->ne[3]!=1||dst->type!=GGML_TYPE_I32)return;
    float cut=0;auto target=xeon_margin_rows_for(dst->name,"ffn_moe_argsort-",dst->ne[1],&cut);
    if(!target)return;
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(dst->src[0]) && dst->src[0]->type==GGML_TYPE_F32 && dst->src[0]->data!=dst->data);
    margin_sorted<<<(dst->ne[1]+63)/64,64,0,ctx.stream()>>>((const float *)dst->src[0]->data,(const int *)dst->data,target,dst->ne[1],cut);
    CUDA_CHECK(cudaGetLastError());
}
static int margin_projection(const ggml_tensor *dst){
    if(!marginal.device || dst->ne[2]<64 || dst->ne[2]>=128 || dst->ne[2]!=marginal.n)return -1;
    int p=-1;
    if(std::strcmp(dst->name,"ffn_moe_gate-0")==0)p=0;
    if(std::strcmp(dst->name,"ffn_moe_up-0")==0)p=1;
    if(p>=0)GGML_ASSERT(dst->src[0]->type==GGML_TYPE_IQ2_S && dst->src[0]->ne[0]==5120 && dst->src[0]->ne[1]==2304 && dst->src[0]->ne[2]==384 && dst->src[1]->ne[1]==1 && dst->ne[1]==6);
    return p;
}
template<int Mode> static __global__ void margin_residual_input(const float*x,const int*sel,float*out,int elements){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=elements)return;
    float v=x[sel[i/5120]*5120+i%5120],approx=0;
    if constexpr(Mode==2)approx=__half2float(__float2half_rn(v));
    else {
        float peak=fabsf(v);
        for(int shift=16;shift;shift/=2)peak=fmaxf(peak,__shfl_xor_sync(0xffffffff,peak,shift));
        if(peak>0){
            if constexpr(Mode==0){float d=peak/127.f;approx=__fmul_rn(roundf(v/d),__half2float(__float2half_rn(d)));}
            else {float inv=127.f/peak;approx=__fmul_rn(roundf(v*inv),1.f/inv);}
        }
    }
    out[i]=__fsub_rn(v,approx)*128.f;
}
static __global__ void margin_add_back(float*y,const float*c,const int*sel,int elements){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=elements)return;
    const int at=sel[i/2304]*2304+i%2304;
    y[at]=__fadd_rn(y[at],__fmul_rn(c[i],1.f/128));
}

struct margin_plan { int starts[385]; };
static void margin_build_plan(margin_plan &plan,int p,const ggml_tensor *ids,const char *actual,
                    const std::vector<int32_t>&sorted,const std::vector<int32_t>&counts,cudaStream_t stream) {
    auto &s=marginal;auto &buf=s.projection[p];
    const int n=s.n;GGML_ASSERT(ids->ne[1]==n && counts.size()==384 && sorted.size()>=size_t(n)*6 && s.calls[p]++==0);
    int token_hits=0;
    for(int t=0;t<n;t++){
        const auto &r=s.host[t];int id;memcpy(&id,actual+t*ids->nb[1]+5*ids->nb[0],sizeof(id));
        GGML_ASSERT(r.sixth==id && r.seventh>=0 && r.seventh<384 && r.sixth!=r.seventh && std::isfinite(r.gap) && r.gap>=0 && r.near_boundary==(r.gap<=s.cut));
        token_hits+=r.near_boundary;
    }
    int cursor=0,written=0;
    for(int e=0;e<384;e++){
        plan.starts[e]=written;
        for(int j=0;j<counts[e];j++){
            int t=sorted[cursor+j];GGML_ASSERT(t>=0 && t<n);
            if(s.policy==3)t=(t+37)%n;
            if(s.policy==4 || s.host[t].near_boundary)buf.host_indices[written++]=j;
        }
        cursor+=counts[e];
    }
    plan.starts[384]=written;
    GGML_ASSERT(cursor==6*n && written==6*(s.policy==4?n:token_hits));
    s.planned[p]=written;
    // Decision-only includes building and transferring the same correction list.
    if(written)CUDA_CHECK(cudaMemcpyAsync(buf.device_indices,buf.host_indices,sizeof(int)*written,cudaMemcpyHostToDevice,stream));
}
static void margin_correct(ggml_backend_cuda_context &ctx,int p,const ggml_tensor &w,const ggml_tensor &x,
                         ggml_tensor &y,int at,int selected,int cc){
    if(!selected || marginal.policy==1)return;
    auto &buf=marginal.projection[p];
    const int *indices=buf.device_indices+at;const int len=selected*5120;
    cudaStream_t stream=ctx.stream();
    if(ggml_cuda_should_use_mmvq(w.type,cc,x.ne[1]))
        margin_residual_input<0><<<(len+255)/256,256,0,stream>>>((const float*)x.data,indices,buf.x,len);
    else if(ggml_cuda_should_use_mmq(w.type,cc,x.ne[1],0))
        margin_residual_input<1><<<(len+255)/256,256,0,stream>>>((const float*)x.data,indices,buf.x,len);
    else margin_residual_input<2><<<(len+255)/256,256,0,stream>>>((const float*)x.data,indices,buf.x,len);
    CUDA_CHECK(cudaGetLastError());
    ggml_tensor xx=x,yy=y;
    xx.ne[1]=yy.ne[1]=selected;
    xx.nb[2]=xx.nb[3]=size_t(selected)*xx.nb[1]; yy.nb[2]=yy.nb[3]=size_t(selected)*yy.nb[1];
    xx.data=buf.x;yy.data=buf.y;
    ggml_cuda_mul_mat(ctx,&w,&xx,&yy);
    int count=selected*2304;
    margin_add_back<<<(count+255)/256,256,0,stream>>>((float*)y.data,buf.y,indices,count);
    CUDA_CHECK(cudaGetLastError());marginal.executed[p]+=selected;
}
