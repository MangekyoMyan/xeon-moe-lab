#define GGML_COMMON_IMPL_CPP
#include "ggml-cpu/quants.h"
#include <immintrin.h>
#include <array>
#include <type_traits>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <omp.h>
#include "compact-kernel.inc"
#include "iq-small-kernels.inc"
#include "paired-rows.inc"

static void need(bool c,const char *s){if(!c)throw std::runtime_error(s);}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t checked=0;
template<int Kind> static float single(int n,const pair_block<Kind>*w,const block_q8_K*x){
    if constexpr(Kind==0)return compact_dot(n,w,x);else return iq_compact_dot<Kind==1>(n,w,x);
}
template<int Kind> static void run(int k,int m,int threads,bool bench){
    const size_t blocks=k/256;std::mt19937 rng(1931+Kind+k+m);
    std::vector<pair_block<Kind>> weights(blocks*m);
    std::vector<block_q8_K>x(blocks);std::vector<float>a(m),b(m);
    for(auto &w:weights){
        for(auto &q:w.qs)q=rng();w.d=float(int(rng()%1024)-512)/4096;
        if constexpr(Kind==0)for(auto &s:w.scale)s=2*(rng()%16)+1;
        else for(auto &s:w.scales)s=rng();
        if constexpr(Kind==1)for(auto &q:w.qs)q=(rng()%3)|((rng()%2)<<3)|((rng()%3)<<4)|((rng()%2)<<7);
    }
    for(auto &v:x){v.d=float(int(rng()%1024)-512)/4096;for(auto &q:v.qs)q=int(rng()%256)-128;}
    auto compute=[&](bool paired,std::vector<float>&out){
        // Blocks of 17 exercise odd starts and per-worker tails.
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int begin=0;begin<m;begin+=17){
            const int end=std::min(begin+17,m);int row=begin;
            if(paired)for(;row+1<end;row+=2)paired_dot<Kind>(k,out.data()+row,weights.data()+row*blocks,weights.data()+(row+1)*blocks,x.data());
            for(;row<end;++row)out[row]=single<Kind>(k,weights.data()+row*blocks,x.data());
        }
    };
    compute(false,a);compute(true,b);need(std::memcmp(a.data(),b.data(),m*4)==0,"paired bit mismatch");checked+=m;
    if(!bench)return;
    std::vector<double>s,p;
    for(int repeat=0;repeat<8;++repeat)for(bool pair:{false,true,true,false}){
        double t=now();compute(pair,b);double dt=now()-t;
        need(std::memcmp(a.data(),b.data(),m*4)==0,"timed bit mismatch");
        (pair?p:s).push_back(dt);
    }
    std::printf("{\"kind\":%d,\"k\":%d,\"m\":%d,\"threads\":%d,\"weight_MiB\":%.2f,\"single_ms\":%.6f,\"pair_ms\":%.6f,\"speedup\":%.5f}\n",Kind,k,m,threads,weights.size()*sizeof(weights[0])/1048576.,median(s),median(p),median(s)/median(p));
    std::fflush(stdout);
}
int main(int argc,char**)try{
    for(int k:{256,512,2304,5120})for(int m:{1,2,3,17,129}){
        run<0>(k,m,1,false);run<1>(k,m,1,false);run<2>(k,m,1,false);
    }
    if(argc>1)for(int threads:{1,12})for(int k:{2304,5120}){
        // >=100MiB exceeds both sockets' aggregate LLC; this is synthetic recoded data, not real weights.
        int rows=k==2304?98304:49152;
        run<0>(k,rows,threads,true);run<1>(k,rows,threads,true);run<2>(k,rows,threads,true);
    }
    std::printf("{\"pass\":true,\"outputs_checked\":%zu}\n",checked);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
