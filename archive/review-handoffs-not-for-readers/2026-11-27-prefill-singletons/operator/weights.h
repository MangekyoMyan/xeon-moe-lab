struct TensorInfo {std::string path,name;ggml_type q;int64_t K=0,M=0,E=0;size_t offset=0,expertbytes=0;};
struct Metadata {std::array<TensorInfo,3>w;float clamp=0;};
static Metadata metadata(const std::string &directory,int layer){
    Metadata result;std::array<bool,3> found{};bool clamp_found=false;
    std::vector<std::filesystem::path>files;for(auto &p:std::filesystem::directory_iterator(directory))if(p.path().extension()==".gguf")files.push_back(p.path());std::sort(files.begin(),files.end());
    const char*parts[]={"gate","up","down"};
    for(auto &path:files){auto m=gguf_init_from_file(path.c_str(),{true,nullptr});need(m,"GGUF metadata");
        for(int64_t j=0;j<gguf_get_n_kv(m);++j){std::string key=gguf_get_key(m,j);if(key.find(".swiglu_clamp_exp")!=std::string::npos){
            if(gguf_get_kv_type(m,j)==GGUF_TYPE_ARRAY){need(gguf_get_arr_type(m,j)==GGUF_TYPE_FLOAT32&&size_t(layer)<gguf_get_arr_n(m,j),"clamp array");result.clamp=static_cast<const float*>(gguf_get_arr_data(m,j))[layer];}
            else {need(gguf_get_kv_type(m,j)==GGUF_TYPE_FLOAT32,"clamp scalar");result.clamp=gguf_get_val_f32(m,j);}clamp_found=true;
        }}
        for(int j=0;j<3;++j){std::string name="blk."+std::to_string(layer)+".ffn_"+parts[j]+"_exps.weight";auto id=gguf_find_tensor(m,name.c_str());if(id<0)continue;
            need(!found[j],"duplicate tensor");const auto*n=gguf_get_tensor_ne(m,id);need(n[3]==1,"weight fourth dimension");auto q=gguf_get_tensor_type(m,id);
            result.w[j]={path.string(),name,q,n[0],n[1],n[2],gguf_get_data_offset(m)+gguf_get_tensor_offset(m,id),ggml_row_size(q,n[0])*size_t(n[1])};found[j]=true;
        }gguf_free(m);
    }
    need(found[0]&&found[1]&&found[2]&&clamp_found,"missing weights or actual activation clamp");
    auto &g=result.w[0],&u=result.w[1],&d=result.w[2];need(g.K==u.K&&g.M==u.M&&g.E==u.E&&d.K==g.M&&d.M==g.K&&d.E==g.E,"FFN shapes");
    need(g.K==5120&&g.M==2304&&g.E==384&&std::isfinite(result.clamp),"unexpected model shape");return result;
}
struct Mapping {
    TensorInfo info;void*base=nullptr;size_t length=0;uint8_t*raw=nullptr;bool registered=false;ggml_backend_buffer_t buffer=nullptr;
    Mapping(const TensorInfo&i):info(i){size_t page=sysconf(_SC_PAGESIZE),start=i.offset/page*page,skip=i.offset-start;length=skip+i.expertbytes*i.E;
        int fd=open(i.path.c_str(),O_RDONLY);need(fd>=0,"open weights");struct stat st;need(!fstat(fd,&st)&&i.offset<=size_t(st.st_size)&&i.expertbytes*i.E<=size_t(st.st_size)-i.offset,"file range");
        base=mmap(nullptr,length,PROT_READ,MAP_SHARED,fd,start);close(fd);need(base!=MAP_FAILED,"mmap");raw=static_cast<uint8_t*>(base)+skip;
        registered=ggml_backend_cuda_register_host_buffer(base,length);need(registered,"host weight registration");buffer=ggml_backend_cpu_buffer_from_ptr(raw,i.expertbytes*i.E);need(buffer,"host weight wrapper");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ~Mapping(){if(buffer)ggml_backend_buffer_free(buffer);if(registered)ggml_backend_cuda_unregister_host_buffer(base);if(base&&base!=MAP_FAILED)munmap(base,length);}
};
