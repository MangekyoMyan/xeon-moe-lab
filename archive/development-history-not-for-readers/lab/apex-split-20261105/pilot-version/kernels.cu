#include <cuda_runtime.h>
#include <stdint.h>
// Slots identify token-expert pairs in original token-major order. Each slot is written once.
__global__ void gather(const float *x,const int32_t *slots,float *out,int D,int N,int topk){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(D)*N;i+=int64_t(gridDim.x)*blockDim.x)
        out[i]=x[int64_t(slots[i/D]/topk)*D+i%D];
}
__global__ void scatter(const float *x,const int32_t *slots,float *out,int D,int N){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(D)*N;i+=int64_t(gridDim.x)*blockDim.x)
        out[int64_t(slots[i/D])*D+i%D]=x[i];
}
__global__ void merge(const float *pairs,const float *weights,float *out,int D,int T,int topk){
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(D)*T;i+=int64_t(gridDim.x)*blockDim.x){
        int t=i/D,d=i%D;float value=0;
        for(int k=0;k<topk;++k)value+=pairs[(int64_t(t)*topk+k)*D+d]*weights[t*topk+k];out[i]=value;
    }
}
extern "C" void split_gather(const float*x,const int32_t*s,float*y,int D,int N,int topk,cudaStream_t stream){gather<<<256,256,0,stream>>>(x,s,y,D,N,topk);}
extern "C" void split_scatter(const float*x,const int32_t*s,float*y,int D,int N,cudaStream_t stream){scatter<<<256,256,0,stream>>>(x,s,y,D,N);}
extern "C" void split_merge(const float*x,const float*w,float*y,int D,int T,int topk,cudaStream_t stream){merge<<<256,256,0,stream>>>(x,w,y,D,T,topk);}
