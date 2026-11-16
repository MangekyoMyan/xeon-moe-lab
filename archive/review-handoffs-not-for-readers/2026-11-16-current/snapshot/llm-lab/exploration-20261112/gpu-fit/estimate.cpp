#include "arg.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <stdexcept>

int main(int argc,char ** argv) try {
    common_params params;common_init();
    if(!common_params_parse(argc,argv,params,LLAMA_EXAMPLE_FIT_PARAMS))return 1;
    if(params.model.path.empty()||params.n_ctx!=131072||params.n_parallel!=1||params.n_sequences!=1||params.lazy_mode!=LLAMA_LAZY_MODE_OFF)
        throw std::runtime_error("This estimator requires an explicit local model,128k,single sequence,and lazy-mode off");
    if(!params.mmproj.path.empty()||params.speculative.has_dft()||params.speculative.has_synth())throw std::runtime_error("No projector/draft allocation is allowed in metadata-only estimator");
    for(auto type:params.speculative.types)if(type!=COMMON_SPECULATIVE_TYPE_NONE)throw std::runtime_error("Speculation is outside the measured server profile");
    params.n_outputs_max=1;params.n_outputs_max_per_seq=1;
    llama_backend_init();llama_numa_init(params.numa);
    auto model=common_model_params_to_llama(params);auto context=common_context_params_to_llama(params);
    model.no_alloc=true;model.load_mode=LLAMA_LOAD_MODE_NONE;model.check_tensors=false;
    if(context.n_outputs_max!=1||context.n_outputs_max_per_seq!=1)throw std::runtime_error("Output reservation mismatch");
    uint32_t train=0,actual=0;
    auto rows=common_fit_measure(params.model.path.c_str(),&model,&context,&train,&actual,GGML_LOG_LEVEL_ERROR);
    if(actual!=131072)throw std::runtime_error("Context was changed by measurement");
    nlohmann::json result={{"metadata_only",true},{"context",actual},{"training_context",train},{"batch",context.n_batch},{"ubatch",context.n_ubatch},{"outputs_max",1},{"outputs_per_sequence",1},{"rows",nlohmann::json::array()}};
    for(const auto &row:rows)result["rows"].push_back({{"kind",int(row.kind)},{"name",row.name},{"description",row.description},{"device_total_bytes",row.total},{"device_free_bytes",row.free},{"model_bytes",row.model},{"context_bytes",row.context},{"compute_bytes",row.compute},{"estimated_bytes",row.model+row.context+row.compute}});
    std::puts(result.dump().c_str());llama_backend_free();return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
