#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
int main(int argc,char ** argv){
    int count=0;auto status=cudaGetDeviceCount(&count);
    std::printf("{\"device_count\":%d,\"cuda_status\":%d,\"message\":\"%s\"}\n",count,int(status),cudaGetErrorString(status));std::fflush(stdout);
    if(status!=cudaSuccess||count!=1)return 2;
    void * memory=nullptr;if(cudaMalloc(&memory,4096)!=cudaSuccess)return 3;
    if(cudaMemset(memory,0,4096)!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess||cudaFree(memory)!=cudaSuccess)return 4;
    if(argc==2&&!std::strcmp(argv[1],"host-oob")){
        volatile char * probe=new char[16];probe[16]=1;delete[] probe;
    }
    return 0;
}
