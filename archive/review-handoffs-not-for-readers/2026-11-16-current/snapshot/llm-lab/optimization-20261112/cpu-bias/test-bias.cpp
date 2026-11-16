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
#include "biased-iq3.inc"
static void need(bool c,const char*s){if(!c)throw std::runtime_error(s);}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static double med(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static size_t checked=0;
static void run(int k,int m,int threads,bool edge,bool bench){
    size_t blocks=k/256;std::mt19937 rng(1931+k+m);
    std::vector<compact_block>w(blocks*m);std::vector<biased_iq3_block>b(w.size());
    std::vector<block_q8_K>x(blocks),u(blocks);std::vector<float>base(m),got(m);
    for(size_t i=0;i<w.size();++i){
        auto&t=w[i];for(auto&q:t.qs)q=rng();t.d=float(int(rng()%1024)-512)/4096;
        for(auto&s:t.scale)s=2*(rng()%16)+1;make_biased(t,b[i]);
    }
    for(auto&t:x){t.d=float(int(rng()%1024)-512)/4096;for(auto&q:t.qs)q=int(rng()%(edge?256:255))-(edge?128:127);}
    if(edge)x[0].qs[0]=-128;
    bool fallback=contains_minus128(k,x.data());need(fallback==edge,"minus128 detection");bias_activation(k,x.data(),u.data());
    auto compute=[&](int mode,std::vector<float>&out){
        #pragma omp parallel for schedule(static) num_threads(threads)
        for(int row=0;row<m;++row){
            if(mode==0)out[row]=compact_dot(k,w.data()+row*blocks,x.data());
            else if(fallback)out[row]=biased_dot<2>(k,b.data()+row*blocks,x.data());
            else if(mode==1)out[row]=biased_dot<0>(k,b.data()+row*blocks,x.data());
            else out[row]=biased_dot<1>(k,b.data()+row*blocks,u.data());
        }
    };
    compute(0,base);for(int mode:{1,2}){compute(mode,got);need(std::memcmp(base.data(),got.data(),m*4)==0,"bias bit mismatch");checked+=m;}
    if(!bench)return;
    for(int mode:{1,2}){
        std::vector<double>a,c;
        for(int rep=0;rep<8;++rep)for(int arm:{0,mode,mode,0}){
            double t=now();compute(arm,got);double dt=now()-t;
            need(std::memcmp(base.data(),got.data(),m*4)==0,"timed bias mismatch");(arm?c:a).push_back(dt);
        }
        std::printf("{\"k\":%d,\"m\":%d,\"threads\":%d,\"mode\":%d,\"old_MiB\":%.2f,\"new_MiB\":%.2f,\"original_ms\":%.6f,\"bias_ms\":%.6f,\"speedup\":%.5f}\n",k,m,threads,mode,w.size()*sizeof(w[0])/1048576.,b.size()*sizeof(b[0])/1048576.,med(a),med(c),med(a)/med(c));std::fflush(stdout);
    }
}
int main(int argc,char**)try{
    for(int k:{256,512,2304,5120})for(int m:{1,2,17,129})for(bool edge:{false,true})run(k,m,1,edge,false);
    if(argc>1)for(int nth:{1,12})for(int k:{2304,5120})run(k,k==2304?98304:49152,nth,false,true);
    std::printf("{\"pass\":true,\"outputs_checked\":%zu}\n",checked);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
