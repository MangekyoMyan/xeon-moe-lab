// Check the same 32 physical weight rows for every position, not just the original six sample positions.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace fs=std::filesystem;
static void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
template<class T>static std::vector<T> readbin(const fs::path&p,size_t n){
    need(fs::file_size(p)==n*sizeof(T),"capture length");std::vector<T>v(n);
    std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));need(bool(f),"capture read");return v;
}
int main(int argc,char**argv)try{
    need(argc==4,"expected shard, native endpoint capture, new TSV");
    const fs::path captured=argv[2];const int K=5120,M=2304,E=384,T=96,slots=6;
    ggml_cpu_init();auto meta=gguf_init_from_file(argv[1],{true,nullptr});need(meta,"GGUF header");
    std::ifstream f(argv[1],std::ios::binary);std::ofstream out(argv[3]);need(bool(f)&&bool(out),"paths");
    const size_t rowbytes=ggml_row_size(GGML_TYPE_IQ2_S,K);auto deq=ggml_get_type_traits(GGML_TYPE_IQ2_S)->to_float;
    std::vector<uint8_t> bytes(rowbytes);std::vector<float>w(K);
    out<<"case\tprojection\ttoken\tslot\texpert\trow\tF64\tGPU\n"<<std::setprecision(17);
    int total=0;
    for(const std::string tag:{"JA96","PY96"})for(const std::string role:{"gate","up"}){
        auto path=captured/(tag+"-early");const auto prefix="ffn_moe_"+role+"-0";
        auto ids=readbin<int32_t>(path/(prefix+"-ids.bin"),T*slots);
        auto x=readbin<float>(path/(prefix+"-input.bin"),T*K);
        auto y=readbin<float>(path/(prefix+".bin"),T*slots*M);
        auto key=std::string("blk.0.ffn_")+role+"_exps.weight";auto idx=gguf_find_tensor(meta,key.c_str());
        need(idx>=0 && gguf_get_tensor_type(meta,idx)==GGML_TYPE_IQ2_S,"IQ2_S tensor");
        auto ne=gguf_get_tensor_ne(meta,idx);need(ne[0]==K&&ne[1]==M&&ne[2]==E,"dimensions");
        size_t base=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,idx);
        std::vector<std::vector<int>> pos(E);
        for(int i=0;i<T*slots;i++){need(ids[i]>=0&&ids[i]<E,"ID");pos[ids[i]].push_back(i);}
        for(int e=0;e<E;e++)if(!pos[e].empty())for(int j=0;j<32;j++){
            int row=j*(M-1)/31;f.seekg(base+(size_t(e)*M+row)*rowbytes);
            f.read(reinterpret_cast<char*>(bytes.data()),rowbytes);need(bool(f),"weight read");deq(bytes.data(),w.data(),K);
            for(int p:pos[e]){
                int t=p/slots,slot=p%slots;const float*a=&x[size_t(t)*K];
                double ref=0;for(int k=0;k<K;k++)ref+=double(w[k])*a[k];
                float actual=y[size_t(p)*M+row];need(std::isfinite(ref)&&std::isfinite(actual),"numerics");
                out<<tag<<'\t'<<role<<'\t'<<t<<'\t'<<slot<<'\t'<<e<<'\t'<<row<<'\t'<<ref<<'\t'<<actual<<'\n';total++;
            }
        }
    }
    gguf_free(meta);need(total==73728&&out.good(),"complete sample");std::cout<<total<<" real dots\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
