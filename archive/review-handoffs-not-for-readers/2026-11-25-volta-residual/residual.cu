#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Keep the second input in the normal dynamic range of the half-scale paths.
// The companion graph multiplies the correction result by the exact reciprocal.
template<int Mode, bool Paired>
__global__ void prepare_residual(const float *x, float *out, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float v = x[i];
    float approx;
    if constexpr (Mode == 2) {
        approx = __half2float(__float2half_rn(v));
    } else {
        float peak = fabsf(v);
        for (int offset = 16; offset; offset /= 2) {
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, offset));
        }
        approx = 0.f;
        if (peak > 0) {
            if constexpr (Mode == 0) {
                const float d = peak / 127.f;
                approx = __fmul_rn(roundf(v / d), __half2float(__float2half_rn(d)));
            } else {
                const float inv = 127.f / peak;
                approx = __fmul_rn(roundf(v * inv), 1.f / inv);
            }
        }
    }
    if constexpr (Paired) out[i] = v;
    out[(Paired ? elements : 0) + i] = __fsub_rn(v, approx) * 128.f;
}

template<bool Paired>
static cudaError_t dispatch(int mode, const float *x, float *out, int n, cudaStream_t stream) {
    const int blocks = (n + 255) / 256;
    switch (mode) {
        case 0: prepare_residual<0,Paired><<<blocks,256,0,stream>>>(x,out,n); break;
        case 1: prepare_residual<1,Paired><<<blocks,256,0,stream>>>(x,out,n); break;
        case 2: prepare_residual<2,Paired><<<blocks,256,0,stream>>>(x,out,n); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

extern "C" cudaError_t residual_input(int mode, bool paired, const float *input,
                                       float *out, int elements, cudaStream_t stream) {
    if (elements <= 0 || elements % 32) return cudaErrorInvalidValue;
    return paired ? dispatch<true>(mode,input,out,elements,stream) : dispatch<false>(mode,input,out,elements,stream);
}
