#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

#define CU(x) do { auto e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(2); } } while (0)
#define BL(x) do { auto e = (x); if (e != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: %d\n", #x, int(e)); std::exit(3); } } while (0)

template<class T> __global__ void fill(T * p, size_t count, float value) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) p[i] = T(value);
}
__global__ void copy_vector(const float4 * a, float4 * b, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) b[i] = a[i];
}

static void measure(const char * name, const std::function<void()> & launch, double amount, const char * unit, double duration) {
    cudaEvent_t a, b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    for (int i = 0; i < 4; ++i) launch();
    CU(cudaDeviceSynchronize());
    CU(cudaEventRecord(a)); for (int i = 0; i < 10; ++i) launch(); CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
    float ms = 0; CU(cudaEventElapsedTime(&ms, a, b));
    const int iterations = std::clamp(int(duration * 10000 / ms), 1, 100000);
    CU(cudaEventRecord(a)); for (int i = 0; i < iterations; ++i) launch(); CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
    CU(cudaEventElapsedTime(&ms, a, b));
    std::printf("{\"test\":\"%s\",\"iterations\":%d,\"seconds\":%.9g,\"rate\":%.9g,\"unit\":\"%s\"}\n", name, iterations, ms / 1000.0, amount * iterations / (ms / 1000.0), unit);
    std::fflush(stdout); CU(cudaEventDestroy(a)); CU(cudaEventDestroy(b));
}

static void verify(float * p, size_t count, float expected) {
    float values[16];
    for (size_t offset : {size_t(0), count / 2, count - 16}) {
        CU(cudaMemcpy(values, p + offset, sizeof(values), cudaMemcpyDeviceToHost));
        for (float value : values) if (!std::isfinite(value) || std::abs(value - expected) > 1e-4f) {
            std::fprintf(stderr, "Validation failed: %g != %g\n", value, expected); std::exit(4);
        }
    }
}

int main(int argc, char ** argv) {
    const double duration = argc == 2 ? std::atof(argv[1]) : 2;
    if (duration <= 0 || duration > 20) return 1;
    const size_t bytes = 256ull << 20;
    float * src, * dst; CU(cudaMalloc(&src, bytes)); CU(cudaMalloc(&dst, bytes));
    fill<<<512, 256>>>(src, bytes / sizeof(float), 0.5f); CU(cudaGetLastError());
    measure("device_copy_256MiB", [&] { copy_vector<<<512, 256>>>((float4 *)src, (float4 *)dst, bytes / sizeof(float4)); CU(cudaGetLastError()); }, 2.0 * bytes / 1e9, "GB/s_read_plus_write", duration);
    verify(dst, bytes / sizeof(float), 0.5f); CU(cudaFree(src)); CU(cudaFree(dst));
    const int n = 4096; const size_t count = size_t(n) * n;
    float * a, * b, * c; CU(cudaMalloc(&a, count * 4)); CU(cudaMalloc(&b, count * 4)); CU(cudaMalloc(&c, count * 4));
    fill<<<512,256>>>(a,count,0.5f); fill<<<512,256>>>(b,count,0.25f); CU(cudaGetLastError());
    cublasHandle_t handle; BL(cublasCreate(&handle));
    float alpha = 1, beta = 0;
    measure("fp32_sgemm_4096", [&] { BL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,a,n,b,n,&beta,c,n)); }, 2.0 * n * n * n / 1e12, "TFLOPS", duration);
    verify(c,count,float(n)*0.125f); CU(cudaFree(a)); CU(cudaFree(b));
    __half * ah, * bh; CU(cudaMalloc(&ah,count*2)); CU(cudaMalloc(&bh,count*2));
    fill<<<512,256>>>(ah,count,0.5f); fill<<<512,256>>>(bh,count,0.25f); CU(cudaGetLastError());
    BL(cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH));
    measure("fp16_tensor_gemm_fp32_accumulate_4096", [&] { BL(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,ah,CUDA_R_16F,n,bh,CUDA_R_16F,n,&beta,c,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }, 2.0 * n * n * n / 1e12, "TFLOPS", duration);
    verify(c,count,float(n)*0.125f); CU(cudaFree(ah)); CU(cudaFree(bh)); CU(cudaFree(c)); BL(cublasDestroy(handle));
    std::puts("Validation passed (sampled output positions)");
}
