// Paired latency of a greedy CPU scan and a combined top-two scan on the same real vocabulary arrays.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
static constexpr int V=129280;
using clock_type=std::chrono::steady_clock;
static uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch()).count();}
struct best2{int best,second;float gap;};
__attribute__((noinline)) static int cached_top1(const llama_token_data*a,int n){
    int top=0;float best=a[0].logit;
    for(int i=1;i<n;i++)if(a[i].logit>best){best=a[i].logit;top=i;}
    return top;
}
__attribute__((noinline)) static best2 combined(const llama_token_data*a,int n){
    int top=0,second=1;
    if(a[1].logit>a[0].logit){top=1;second=0;}
    float first_value=a[top].logit, second_value=a[second].logit;
    for(int i=2;i<n;i++){
        const float v=a[i].logit;
        if(v>second_value){
            if(v>first_value){second=top;second_value=first_value;top=i;first_value=v;}
            else{second=i;second_value=v;}
        }
    }
    return {top,second,first_value-second_value};
}
static volatile uint64_t sink=0;
int main(int argc,char **argv)try{
    if(argc<4)throw std::runtime_error("output TSV and F32 files required");
    std::vector<std::vector<llama_token_data>> rows;
    for(int i=2;i<argc;i++){
        std::ifstream f(argv[i],std::ios::binary);if(!f)throw std::runtime_error("input");
        std::vector<float> x(V);
        while(f.read((char*)x.data(),V*4)){
            auto &a=rows.emplace_back(V);
            for(int j=0;j<V;j++){if(!std::isfinite(x[j]))throw std::runtime_error("finite");a[j]={j,x[j],0};}
        }
        if(!f.eof()||f.gcount()!=0)throw std::runtime_error("file size");
    }
    auto sampler=llama_sampler_init_greedy();if(!sampler||rows.empty())throw std::runtime_error("sampler");
    std::ofstream out(argv[1]);out<<"rep\trow\torder\tmethod\tns\tid\tsecond\tgap\tflag\n"<<std::setprecision(12);
    for(int rep=-1;rep<6;rep++)for(size_t z=0;z<rows.size();z++){
        size_t i=rep%2==0?z:rows.size()-1-z;auto &a=rows[i];
        best2 v{-1,-1,0};int selected=-1,fast=-1;
        const int orders[6][3]={{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
        for(int ord=0;ord<3;ord++){
            const int method=orders[(int(z)+rep+6)%6][ord];uint64_t dt;
            if(method==2){
                auto start=ns();v=combined(a.data(),V);bool flag=v.gap<=.25f;sink+=uint64_t(v.best+1)+flag;dt=ns()-start;
                if(rep>=0)out<<rep<<'\t'<<i<<'\t'<<ord<<"\ttop2\t"<<dt<<'\t'<<v.best<<'\t'<<v.second<<'\t'<<v.gap<<'\t'<<flag<<'\n';
            }else if(method==1){
                auto start=ns();fast=cached_top1(a.data(),V);sink+=fast+1;dt=ns()-start;
                if(rep>=0)out<<rep<<'\t'<<i<<'\t'<<ord<<"\tcached1\t"<<dt<<'\t'<<fast<<"\t-1\t0\t0\n";
            }else{
                llama_token_data_array arr{a.data(),a.size(),-1,false};
                auto start=ns();llama_sampler_apply(sampler,&arr);selected=int(arr.selected);sink+=selected+1;dt=ns()-start;
                if(rep>=0)out<<rep<<'\t'<<i<<'\t'<<ord<<"\tnative\t"<<dt<<'\t'<<selected<<"\t-1\t0\t0\n";
            }
        }
        if(v.best!=selected||fast!=selected||v.best==v.second)throw std::runtime_error("greedy mismatch");
    }
    llama_sampler_free(sampler);std::cout<<rows.size()<<" fixed rows, checksum "<<sink<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
