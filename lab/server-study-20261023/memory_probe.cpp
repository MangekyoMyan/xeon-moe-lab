#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#include <unistd.h>
#include <time.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

static double now(){timespec t{};assert(!clock_gettime(CLOCK_MONOTONIC_RAW,&t));return t.tv_sec+t.tv_nsec*1e-9;}
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);assert(!sched_setaffinity(0,sizeof(s),&s));}
static std::vector<int> choose(int threads,const std::string &scope){
    std::vector<int> v;
    for(int c=0;c<24;++c){
        if(scope=="node0" && c%12>=6)continue;
        if(scope=="node1" && c%12<6)continue;
        v.push_back(c);
    }
    assert(threads>0&&int(v.size())>=threads);v.resize(threads);return v;
}
static void *allocate(size_t bytes,int node){
    void *p=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);assert(p!=MAP_FAILED);
    assert(!madvise(p,bytes,MADV_NOHUGEPAGE));
    unsigned long mask=node<0?3UL:(1UL<<node);
    assert(!syscall(SYS_mbind,p,bytes,node<0?MPOL_INTERLEAVE:MPOL_BIND,&mask,64UL,0UL));
    return p;
}
static void placements(void *ptr,size_t bytes,const char *label,int expected){
    const int n=1024;std::vector<void*> pages(n);std::vector<int> status(n,-1000);size_t total=bytes/4096;
    for(int i=0;i<n;++i)pages[i]=(char*)ptr+((size_t(i)*7919)%total)*4096;
    assert(!syscall(SYS_move_pages,0,n,pages.data(),nullptr,status.data(),0));
    int counts[2]={0,0};
    for(int x:status){assert(x==0||x==1);++counts[x];if(expected>=0)assert(x==expected);}
    if(expected<0)assert(counts[0]>400&&counts[1]>400);
    printf("{\"placement\":\"%s\",\"node0_pages\":%d,\"node1_pages\":%d,\"validated\":true}\n",label,counts[0],counts[1]);fflush(stdout);
}
static double read_array(const float *data,size_t count,int threads){
    double total=0;
    #pragma omp parallel num_threads(threads) reduction(+:total)
    {
        const size_t blocks=count/32;
        size_t first=blocks*omp_get_thread_num()/threads*32,last=blocks*(omp_get_thread_num()+1)/threads*32;
        __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
        for(size_t i=first;i<last;i+=32){
            a=_mm256_add_ps(a,_mm256_loadu_ps(data+i));
            b=_mm256_add_ps(b,_mm256_loadu_ps(data+i+8));
            c=_mm256_add_ps(c,_mm256_loadu_ps(data+i+16));
            d=_mm256_add_ps(d,_mm256_loadu_ps(data+i+24));
        }
        float lanes[32];_mm256_storeu_ps(lanes,a);_mm256_storeu_ps(lanes+8,b);_mm256_storeu_ps(lanes+16,c);_mm256_storeu_ps(lanes+24,d);
        for(float x:lanes)total+=x;
    }
    return total;
}
struct alignas(64) Node {Node *next;char pad[56];};
static uint64_t random64(uint64_t &state){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
int main(int argc,char **argv){
    if(argc!=8)return 2;
    const std::string mode=argv[1],scope=argv[3];int threads=std::atoi(argv[2]),source_node=std::atoi(argv[4]),dest_node=std::atoi(argv[5]);
    const size_t bytes=size_t(std::strtoull(argv[6],nullptr,10))*1024*1024;const double seconds=std::atof(argv[7]);
    assert(mode=="read"||mode=="copy"||mode=="latency");
    assert(scope=="both"||scope=="node0"||scope=="node1");assert(source_node>=-1&&source_node<=1&&dest_node>=-1&&dest_node<=1);
    assert(bytes>=16*1024*1024&&bytes<=size_t(2)*1024*1024*1024&&seconds>=.1&&seconds<=120);
    assert(mode!="latency"||threads==1);const auto cpus=choose(threads,scope);
    omp_set_dynamic(0);auto *src=(float*)allocate(bytes,source_node);float *dst=mode=="copy"?(float*)allocate(bytes,dest_node):nullptr;
    #pragma omp parallel num_threads(threads)
    {
        pin(cpus[omp_get_thread_num()]);
        #pragma omp for schedule(static)
        for(size_t i=0;i<bytes/sizeof(float);++i){src[i]=1.0f;if(dst)dst[i]=0;}
    }
    Node *cursor=nullptr;
    if(mode=="latency"){
        const size_t n=bytes/sizeof(Node);std::vector<uint32_t> order(n);std::iota(order.begin(),order.end(),0);
        uint64_t state=0x1ff2d91a3ULL;
        for(size_t i=n-1;i>0;--i)std::swap(order[i],order[random64(state)%(i+1)]);
        auto *nodes=(Node*)src;for(size_t i=0;i<n;++i)nodes[order[i]].next=&nodes[order[(i+1)%n]];
        cursor=&nodes[order[0]];
        for(size_t i=0;i<n;++i)cursor=cursor->next;
        assert(cursor==&nodes[order[0]]);
    }
    placements(src,bytes,"source",source_node);if(dst)placements(dst,bytes,"destination",dest_node);
    auto work=[&](){
        if(mode=="read")assert(read_array(src,bytes/sizeof(float),threads)==double(bytes/sizeof(float)));
        else if(mode=="copy"){
            #pragma omp parallel num_threads(threads)
            {
                size_t first=bytes/64*omp_get_thread_num()/threads*64,last=bytes/64*(omp_get_thread_num()+1)/threads*64;
                std::memcpy((char*)dst+first,(char*)src+first,last-first);
            }
        }else{
            for(int i=0;i<65536;++i){cursor=cursor->next;asm volatile("" : "+r"(cursor) :: "memory");}
        }
    };
    auto affinity_check=[&](){
        #pragma omp parallel num_threads(threads)
        {
            cpu_set_t set;CPU_ZERO(&set);assert(!sched_getaffinity(0,sizeof(set),&set));
            assert(CPU_COUNT(&set)==1&&CPU_ISSET(cpus[omp_get_thread_num()],&set));
        }
    };
    double warm=now();do{work();}while(now()-warm<.5);affinity_check();
    printf("{\"ready\":true,\"mode\":\"%s\",\"threads\":%d,\"scope\":\"%s\",\"source_node\":%d,\"dest_node\":%d,\"bytes\":%zu,\"page_size\":4096,\"seconds\":%.3f,\"cpus\":[",mode.c_str(),threads,scope.c_str(),source_node,dest_node,bytes,seconds);
    for(size_t i=0;i<cpus.size();++i)printf("%s%d",i?",":"",cpus[i]);
    printf("]}\n");fflush(stdout);
    assert(std::getchar()=='G');uint64_t passes=0;const double start=now();
    do{work();++passes;}while(now()-start<seconds);
    const double elapsed=now()-start;
    printf("{\"done\":true,\"seconds\":%.9f,\"passes\":%llu,\"logical_GBs\":%.9f,\"dependent_load_ns\":%.9f}\n",elapsed,(unsigned long long)passes,mode=="latency"?0:double(bytes)*passes/elapsed/1e9,mode=="latency"?elapsed*1e9/(passes*65536.0):0);fflush(stdout);
    assert(std::getchar()=='Q');affinity_check();
    if(dst)assert(read_array(dst,bytes/sizeof(float),threads)==double(bytes/sizeof(float)));
    printf("{\"post_validation\":true}\n");fflush(stdout);
    if(dst)munmap(dst,bytes);
    munmap(src,bytes);
}
