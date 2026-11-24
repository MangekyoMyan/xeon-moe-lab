// Replay native IQ dot products on exactly the measured rows, including a second residual dot.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "gguf.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace fs=std::filesystem;
static void need(bool good,const char*s){if(!good)throw std::runtime_error(s);}
template<class T> static std::vector<T> load(const fs::path &p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"file open");
    auto bytes=(size_t)f.tellg();need(bytes%sizeof(T)==0,"length");std::vector<T> v(bytes/sizeof(T));
    f.seekg(0);f.read((char*)v.data(),bytes);need(bool(f),"file read");return v;
}
static float half(float x){return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));}
int main(int argc,char**argv){
    try{
        need(argc==4,"expected shard, early directory, output TSV");
        ggml_cpu_init();
        ggml_context *meta=nullptr;auto file=gguf_init_from_file(argv[1],{true,&meta});need(file&&meta,"GGUF metadata");
        std::ifstream shard(argv[1],std::ios::binary);fs::path leaf=argv[2];std::ifstream jf(leaf/"CAPTURE.json");json rec;jf>>rec;
        std::ofstream o(argv[3]);need(bool(o),"TSV");
        o<<"projection\ttoken\tslot\texpert\trow\tobserved\tfull\tq8K_double\tcpu_dot\tcpu_corrected\tq32_fp16_scale\tq32_f32_scale\tfp16_both\n"<<std::setprecision(13);
        for(std::string role:{"gate","up"}){
            const std::string name="blk.0.ffn_"+role+"_exps.weight";const std::string label="ffn_moe_"+role+"-0";
            int64_t tensor_id=gguf_find_tensor(file,name.c_str());need(tensor_id>=0,"tensor in shard");
            auto w=ggml_get_tensor(meta,name.c_str());need(w && w->type==GGML_TYPE_IQ2_S && w->ne[0]==5120 && w->ne[1]==2304,"first layer weight shape/type");
            const auto traits=ggml_get_type_traits_cpu(w->type);
            need(traits->vec_dot && traits->vec_dot_type==GGML_TYPE_Q8_K,"native CPU dot pair");
            int dim=w->ne[0],rows=w->ne[1];auto x=load<float>(leaf/(label+"-input.bin"));auto ids=load<int32_t>(leaf/(label+"-ids.bin"));
            need(x.size()==size_t(dim)*96 && ids.size()==6*96,"measured input geometry");
            const auto off=gguf_get_data_offset(file)+gguf_get_tensor_offset(file,tensor_id);
            std::vector<float> weights(dim),xq(dim),residual(dim),h32(dim),f32(dim),xhalf(dim);
            std::vector<block_q8_K> q1(dim/256),q2(dim/256);
            std::vector<uint8_t> packed(w->nb[1]);
            auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
            auto deq=ggml_get_type_traits(w->type)->to_float;
            std::map<std::tuple<int,int,int>,std::array<double,3>> expected;
            for(auto &r:rec.at("F64_samples"))if(r[0]==label)expected[{r[1],r[2],r[4]}]={r[5],r[6],r[7]};
            for(int t:{0,1,23,47,71,95}){
                const float *v=x.data()+dim*t;
                quant(v,q1.data(),dim);dequantize_row_q8_K(q1.data(),xq.data(),dim);
                for(int k=0;k<dim;++k){residual[k]=v[k]-xq[k];xhalf[k]=half(v[k]);}
                quant(residual.data(),q2.data(),dim);
                for(int a=0;a<dim;a+=32){
                    float peak=0;for(int k=a;k<a+32;++k)peak=std::max(peak,std::abs(v[k]));
                    const float scale=peak/127.0f,inv=peak?127.f/peak:0.f,flat=inv?1.f/inv:0.f;
                    for(int k=a;k<a+32;++k){
                        h32[k]=peak?std::round(v[k]/scale)*half(scale):0.f;
                        f32[k]=peak?std::round(v[k]*inv)*flat:0.f;
                    }
                }
                for(int slot=0;slot<6;++slot){
                    const int e=ids[6*t+slot];need(e>=0&&e<384,"expert");
                    for(int j=0;j<32;++j){
                        const int m=(j*(rows-1))/31;
                        shard.seekg(off+size_t(e)*w->nb[2]+size_t(m)*w->nb[1]);
                        shard.read((char*)packed.data(),packed.size());need(bool(shard),"physical row");
                        deq(packed.data(),weights.data(),dim);
                        float dot1=0,dot2=0;traits->vec_dot(dim,&dot1,0,packed.data(),0,q1.data(),0,1);
                        traits->vec_dot(dim,&dot2,0,packed.data(),0,q2.data(),0,1);
                        double full=0,q8=0,subh=0,subf=0,hh=0;
                        for(int k=0;k<dim;++k){const double ww=weights[k];full+=ww*v[k];q8+=ww*xq[k];subh+=ww*h32[k];subf+=ww*f32[k];hh+=double(half(weights[k]))*xhalf[k];}
                        auto ref=expected.at({t,slot,m});need(std::abs(full-ref[0])<1.e-8 && std::abs(q8-ref[1])<1.e-8,"host reconstruction differs");
                        o<<role<<'\t'<<t<<'\t'<<slot<<'\t'<<e<<'\t'<<m<<'\t'<<ref[2]<<'\t'<<full<<'\t'<<q8<<'\t'<<dot1<<'\t'<<dot1+dot2<<'\t'<<subh<<'\t'<<subf<<'\t'<<hh<<'\n';
                    }
                }
            }
        }
        ggml_free(meta);gguf_free(file);need(o.good(),"write");return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
