#define main supplied_probe_main
#include "q8_probe.cpp"
#undef main
#include "json.hpp"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>

static std::string provider(void * f) {
    Dl_info d{};
    if(!f || !dladdr(f,&d)) throw std::runtime_error("provider lookup");
    return std::filesystem::canonical(d.dli_fname).string();
}
int main(int argc,char ** argv) try {
    if(argc<5) throw std::runtime_error("CPU BASE RECEIPT check-or-bench...");
    const std::string cpu=std::filesystem::canonical(argv[1]),base=std::filesystem::canonical(argv[2]);
    const auto from=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    nlohmann::ordered_json p;
    p["CPU_init"]=provider(reinterpret_cast<void*>(ggml_cpu_init));
    p["from_float"]=provider(reinterpret_cast<void*>(from));
    p["reference"]=provider(dlsym(RTLD_DEFAULT,"quantize_row_q8_K_ref"));
    p["base_init"]=provider(reinterpret_cast<void*>(ggml_init));
    if(p["CPU_init"]!=cpu || p["from_float"]!=cpu || p["reference"]!=base || p["base_init"]!=base) throw std::runtime_error("wrong live provider");
    std::ifstream maps("/proc/self/maps");std::string line;p["engine_maps"]=nlohmann::ordered_json::array();
    while(std::getline(maps,line)) if(line.find("/libggml")!=std::string::npos) p["engine_maps"].push_back(line);
    std::ofstream out(argv[3]);out<<p.dump(2)<<'\n';out.close();if(!out)throw std::runtime_error("receipt write");
    std::vector<char*> args{argv[0]};for(int i=4;i<argc;++i)args.push_back(argv[i]);
    return supplied_probe_main(int(args.size()),args.data());
} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 2;}
