#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <cstdlib>
static std::atomic<uint64_t> prepare_calls{0},dot_calls{0},unsafe_calls{0};
extern "C" int xeon_iq3_rhs_prepare_v1(int n,const void * x,int begin,int end,void * sidecar){
 using fn=int(*)(int,const void*,int,int,void*);static fn real=reinterpret_cast<fn>(dlsym(RTLD_NEXT,"xeon_iq3_rhs_prepare_v1"));if(!real)std::abort();
 int safe=real(n,x,begin,end,sidecar);prepare_calls.fetch_add(1,std::memory_order_relaxed);if(!safe)unsafe_calls.fetch_add(1,std::memory_order_relaxed);return safe;
}
extern "C" void xeon_iq3_rhs_dot_v1(int n,float * out,const void * w,const void * x,const void * sidecar){
 using fn=void(*)(int,float*,const void*,const void*,const void*);static fn real=reinterpret_cast<fn>(dlsym(RTLD_NEXT,"xeon_iq3_rhs_dot_v1"));if(!real)std::abort();
 dot_calls.fetch_add(1,std::memory_order_relaxed);real(n,out,w,x,sidecar);
}
extern "C" void xeon_iq_test_counters(uint64_t * out){out[0]=prepare_calls.load();out[1]=dot_calls.load();out[2]=unsafe_calls.load();}
