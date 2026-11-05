#include "route_features.h"
#include <cassert>
#include <random>
#include <cstdio>
int main(){
    std::mt19937 rng(491);size_t cases=0;
    for(int E:{6,12,24,96,384})for(int T:{1,3,16,64})for(size_t gap:{0,4,20}){
        const int top=6;size_t nb0=4,nb1=top*4+gap,bytes=(T-1)*nb1+top*4;
        std::vector<uint8_t> raw(bytes,0xa5);std::vector<uint32_t> expected(E);
        for(int t=0;t<T;++t){std::vector<int> pool(E);for(int e=0;e<E;++e)pool[e]=e;std::shuffle(pool.begin(),pool.end(),rng);
            for(int j=0;j<top;++j){int32_t e=pool[j];std::memcpy(raw.data()+t*nb1+j*nb0,&e,4);++expected[e];}}
        auto f=expert_shadow::describe(raw.data(),bytes,E,T,top,nb0,nb1,1000);assert(f.counts==expected);
        uint64_t active=0,runs=0,padding=0;
        for(int e=0;e<E;++e)if(expected[e]){++active;if(e==0||!expected[e-1])++runs;if(e<E-1&&!expected[e+1])padding+=512;}
        assert(f.active==active&&f.runs==runs&&f.transfer_bytes==active*1000+padding);++cases;
    }
    for(const auto &bad:std::vector<std::vector<int32_t>>{{0,0},{-1,1},{0,6}}){bool threw=false;try{expert_shadow::describe(bad.data(),8,6,1,2,4,8,1000);}catch(const std::exception&){threw=true;}assert(threw);++cases;}
    int32_t ids[]={0,1};bool threw=false;try{expert_shadow::describe(ids,7,6,1,2,4,8,1000);}catch(const std::exception&){threw=true;}assert(threw);
    std::printf("route feature checks passed: %zu\n",cases+1);
}
