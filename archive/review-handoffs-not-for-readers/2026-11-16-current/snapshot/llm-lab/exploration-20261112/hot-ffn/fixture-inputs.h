#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
namespace fixture_input {
constexpr int D=5120,F=2304,E=384,TOP=6;
static void need(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
struct Info{std::string path,name;ggml_type type;int64_t k,m,e;size_t offset,expert_bytes;};
struct Meta{std::array<Info,3>w;float clamp=0;};
static Meta metadata(const std::string & directory,int layer){
    Meta out;std::array<bool,3>found{};bool clamp_found=false;
    std::vector<std::filesystem::path>paths;
    for(auto & p:std::filesystem::directory_iterator(directory))if(p.path().extension()==".gguf")paths.push_back(p.path());
    std::sort(paths.begin(),paths.end());const char * parts[]={"gate","up","down"};
    for(auto & path:paths){
        auto m=gguf_init_from_file(path.c_str(),{true,nullptr});need(m,"metadata");
        for(int64_t i=0;i<gguf_get_n_kv(m);++i)if(std::string(gguf_get_key(m,i)).ends_with(".swiglu_clamp_exp")){
            if(gguf_get_kv_type(m,i)==GGUF_TYPE_ARRAY){
                need(gguf_get_arr_type(m,i)==GGUF_TYPE_FLOAT32&&size_t(layer)<gguf_get_arr_n(m,i),"clamp array");
                out.clamp=static_cast<const float*>(gguf_get_arr_data(m,i))[layer];
            }else{need(gguf_get_kv_type(m,i)==GGUF_TYPE_FLOAT32,"clamp type");out.clamp=gguf_get_val_f32(m,i);}
            clamp_found=true;
        }
        for(int j=0;j<3;++j){
            auto name="blk."+std::to_string(layer)+".ffn_"+parts[j]+"_exps.weight";
            auto id=gguf_find_tensor(m,name.c_str());if(id<0)continue;need(!found[j],"duplicate tensor");
            const auto * n=gguf_get_tensor_ne(m,id);auto q=gguf_get_tensor_type(m,id);
            need(n[3]==1&&n[2]==E,"expert shape");
            out.w[j]={path.string(),name,q,n[0],n[1],n[2],gguf_get_data_offset(m)+gguf_get_tensor_offset(m,id),ggml_row_size(q,n[0])*size_t(n[1])};found[j]=true;
        }
        gguf_free(m);
    }
    need(std::all_of(found.begin(),found.end(),[](bool v){return v;})&&clamp_found,"missing tensor/clamp");
    need(std::isfinite(out.clamp)&&out.clamp>0,"finite positive clamp required");
    for(int j=0;j<3;++j)need(out.w[j].k==(j==2?F:D)&&out.w[j].m==(j==2?D:F),"model dimensions");
    return out;
}
struct Mapping{
    Info info;void * base=nullptr;size_t length=0;uint8_t * raw=nullptr;ggml_backend_buffer_t buffer=nullptr;
    explicit Mapping(const Info & i):info(i){
        size_t page=sysconf(_SC_PAGESIZE),start=i.offset/page*page,skip=i.offset-start;length=skip+i.expert_bytes*i.e;
        int fd=open(i.path.c_str(),O_RDONLY);need(fd>=0,"open model");struct stat st{};
        bool valid=fstat(fd,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>0&&start<=size_t(st.st_size)&&length<=size_t(st.st_size)-start;
        if(!valid){close(fd);throw std::runtime_error("model tensor range");}
        base=mmap(nullptr,length,PROT_READ,MAP_SHARED,fd,start);close(fd);need(base!=MAP_FAILED,"mmap");
        raw=static_cast<uint8_t*>(base)+skip;
        buffer=ggml_backend_cpu_buffer_from_ptr(raw,i.expert_bytes*i.e);need(buffer,"CPU weight wrapper");
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ~Mapping(){if(buffer)ggml_backend_buffer_free(buffer);if(base&&base!=MAP_FAILED)munmap(base,length);}
};

}
