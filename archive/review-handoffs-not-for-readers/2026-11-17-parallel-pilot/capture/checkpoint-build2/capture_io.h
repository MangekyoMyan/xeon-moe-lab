#pragma once
#include "bind_parent_cells_proposal.h"

static json compact_metadata(const llama_model & model, int layer,
                             const std::array<int32_t,6> & ids) {
    using Lookup = const void * const * (*)(const ggml_tensor *);
    auto fn = reinterpret_cast<Lookup>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_lookup_experts_v2"));
    need(fn != nullptr,"compact lookup ABI");
    Dl_info info{}; need(dladdr(reinterpret_cast<void*>(fn),&info) && info.dli_fname,"compact provider");
    Dl_info cpu{};need(dladdr(reinterpret_cast<void*>(ggml_backend_cpu_init),&cpu)&&cpu.dli_fname,"CPU provider");
    need(std::filesystem::canonical(cpu.dli_fname)==std::filesystem::canonical(info.dli_fname),"compact provider mismatch");
    auto & l=model.layers.at(layer); json result=json::array();
    for(auto * w : {l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps}) {
        auto table=fn(w); std::array<bool,6> mask{};
        for(int j=0;j<6;++j) { need(ids[j]>=0&&ids[j]<384,"compact ID"); mask[j]=table&&table[ids[j]]; }
        result.push_back({{"name",w->name},{"type",ggml_type_name(w->type)},
            {"ne",std::vector<int64_t>(w->ne,w->ne+4)},
            {"nb",std::vector<size_t>(w->nb,w->nb+4)},
            {"compact_mask",mask},{"lookup_provider",std::filesystem::canonical(info.dli_fname).string()}});
    }
    return result;
}
static void export_capture(bounded_moe_capture::Capture & cap,const llama_model &model,
                           const std::filesystem::path &out,const json &trace) {
    json meta={{"cells",json::array()},{"callback_timing_valid",false},
               {"call5_8_intermediate_callback_free_bit_identity_proven",false}};
    auto write=[&](const std::string &name,const auto &data,bool floating) {
        if(floating)for(auto x:data)need(std::isfinite(double(x)),"capture finite");
        std::ofstream f(out/name,std::ios::binary);need(bool(f),"capture open");
        f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(data[0]));
        f.close();need(!f.fail(),"capture write");return name;
    };
    for(size_t i=0;i<cap.count;++i){const auto &c=cap.cells[i];need(c.flags==0&&c.seen==7,"complete capture");
        json record;int matches=0;for(const auto &r:trace["records"])if(r["layer"]==c.layer&&r["decode_call"]==c.key){record=r;++matches;}
        need(matches==1,"unique capture route join");
        need(record["layer"]==c.layer&&record["decode_call"]==c.key&&record["ids"]==json(c.ids),"capture route join");
        std::string base="cell"+std::to_string(i);
        meta["cells"].push_back({{"cell",i},{"layer",c.layer},{"call",c.key},{"flags",c.flags},{"seen",c.seen},
            {"x_file",write(base+"-x.f32",c.x,true)},{"ids_file",write(base+"-ids.i32",c.ids,false)},
            {"down_file",write(base+"-down.f32",c.down,true)},{"routing_weights_file",write(base+"-weights.f32",c.routing_weight,true)},
            {"ids",c.ids},{"x_ne",c.x_ne},{"x_nb",c.x_nb},{"ids_ne",c.ids_ne},{"ids_nb",c.ids_nb},
            {"clamp",model.hparams.swiglu_clamp_exp[c.layer]},{"projections",compact_metadata(model,c.layer,c.ids)},
            {"route_record",record}});
    }
    save_json(out/"CAPTURE.json",meta);
}
