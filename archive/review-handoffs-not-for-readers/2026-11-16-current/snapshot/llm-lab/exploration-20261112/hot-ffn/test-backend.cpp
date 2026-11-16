// Include the private implementation to test the exact dispatcher through ggml's
// scheduler without loading a full llama model or adding a test-only public ABI.
#include "hot-backend.cpp"
#include "fixture-inputs.h"
#include <dlfcn.h>
#include <random>
#include <sys/prctl.h>

struct CacheOwner{
    void * owner=nullptr;void (*release)(void*)=nullptr;
    CacheOwner(const std::array<ggml_tensor*,3>&weights,const std::vector<uint32_t>&ids){
        auto create=reinterpret_cast<void*(*)(size_t,size_t,int)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_create_v1"));
        auto add=reinterpret_cast<int(*)(void*,const ggml_tensor*,const uint32_t*,size_t)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_add_selected_v2"));
        release=reinterpret_cast<void(*)(void*)>(dlsym(RTLD_DEFAULT,"xeon_iq3_cache_free_v1"));require(create&&add&&release,"cache test ABI");
        owner=create(size_t(256)*1024*1024,0,12);require(owner,"test compact owner");
        for(auto w:weights)if(w->type!=GGML_TYPE_MXFP4)require(add(owner,w,ids.data(),ids.size())==1,"test compact add");
    }
    ~CacheOwner(){if(owner)release(owner);}
};
static void check_provider(void * fn,const char * path){Dl_info p{};require(dladdr(fn,&p),"provider lookup");require(std::filesystem::canonical(p.dli_fname)==std::filesystem::canonical(path),"wrong test provider");}
int main(int argc,char ** argv)try{
    require(argc==5,"MODEL_DIRECTORY LAYER EXPECTED_CPU_LIBRARY EXPECTED_CUDA_LIBRARY");
    check_provider(reinterpret_cast<void*>(ggml_backend_cpu_init),argv[3]);check_provider(reinterpret_cast<void*>(ggml_backend_cuda_init),argv[4]);
    require(!prctl(PR_SET_THP_DISABLE,1,0,0,0),"THP disable");
    int layer_index=std::stoi(argv[2]);require(layer_index==4||layer_index==24,"test layer");
    auto meta=fixture_input::metadata(argv[1],layer_index);
    auto outer_cpu=ggml_backend_cpu_init();require(outer_cpu,"outer CPU backend");
    auto outer_gpu=ggml_backend_cuda_init(0);
    if(!outer_gpu){ggml_backend_free(outer_cpu);throw std::runtime_error("xeon-hot: outer CUDA backend initialization failed");}
    auto pool_params=ggml_threadpool_params_default(12);pool_params.poll=0;pool_params.strict_cpu=true;for(int i=0;i<12;++i)pool_params.cpumask[i]=true;
    auto outer_pool=ggml_threadpool_new(&pool_params);require(outer_pool,"outer pool");ggml_backend_cpu_set_n_threads(outer_cpu,12);ggml_backend_cpu_set_threadpool(outer_cpu,outer_pool);
    size_t cases=0,outputs=0;double worst_nmse=0;
    {
        auto weights_ctx=ggml_init({1024*1024,nullptr,true});require(weights_ctx,"weight metadata");
        std::array<std::unique_ptr<fixture_input::Mapping>,3>mappings;std::array<ggml_tensor*,3>weights{};
        for(int j=0;j<3;++j){auto & info=meta.w[j];mappings[j]=std::make_unique<fixture_input::Mapping>(info);weights[j]=ggml_new_tensor_3d(weights_ctx,info.type,info.k,info.m,info.e);
            require(ggml_backend_tensor_alloc(mappings[j]->buffer,weights[j],mappings[j]->raw)==GGML_STATUS_SUCCESS,"original weight bind");}
        {
            std::vector<uint32_t>selected={3,41,92,151,244,370},cold={8,79,107,203,277,371},all=selected;all.insert(all.end(),cold.begin(),cold.end());
            CacheOwner cache(weights,all);Graph reference(outer_cpu,weights,TOP,meta.clamp);
            std::mt19937 rng(197+layer_index);std::uniform_real_distribution<float>dist(-.1f,.1f);
            for(int resident:{0,1,3,6}){
                auto state=std::make_unique<State>(nullptr);state->outer_cpu=outer_cpu;state->outer_gpu=outer_gpu;
                auto l=std::make_unique<Layer>();l->owner=state.get();l->index=layer_index;l->clamp=meta.clamp;l->eligible=true;l->original=weights;
                l->selected.assign(selected.begin(),selected.begin()+resident);for(int i=0;i<resident;++i)l->local[l->selected[i]]=i;
                size_t payload=0;for(auto & info:meta.w)payload+=info.expert_bytes*resident;state->payload=payload;state->planned_count=resident;
                state->layers[layer_index]=std::move(l);auto hot=attach(std::move(state));auto s=lookup(outer_cpu);require(s&&s->self==hot,"registry attach");
                xeon_hot_backend_finish(outer_cpu);
                auto & initialized=*s->layers[layer_index];
                // Verify native bank bytes independently of the compute/scatter paths.
                if(resident)for(int j=0;j<3;++j){std::vector<uint8_t>got(ggml_nbytes(initialized.gpu_weights[j]));
                    ggml_backend_tensor_get(initialized.gpu_weights[j],got.data(),0,got.size());auto bytes=meta.w[j].expert_bytes;
                    for(int local=0;local<resident;++local)require(!std::memcmp(got.data()+local*bytes,mappings[j]->raw+size_t(selected[local])*bytes,bytes),"resident native weight bytes");}
                for(int input_mode:{0,1,2}){
                    const bool inputs_on_gpu=input_mode!=0;
                    auto input_ctx=ggml_init({1024*1024,nullptr,true});auto ctx=ggml_init({2*1024*1024,nullptr,true});require(input_ctx&&ctx,"test graph metadata");
                    auto x=ggml_new_tensor_3d(input_ctx,GGML_TYPE_F32,D,1,1);auto ids=ggml_new_tensor_2d(input_ctx,GGML_TYPE_I32,TOP,1);
                    ggml_set_input(x);ggml_set_input(ids);auto ibuf=ggml_backend_alloc_ctx_tensors(input_ctx,inputs_on_gpu?outer_gpu:outer_cpu);require(ibuf,"input buffer");
                    ggml_backend_t backends[]={outer_gpu,hot,outer_cpu};
                    ggml_backend_buffer_type_t bufts[]={ggml_backend_get_default_buffer_type(outer_gpu),ggml_backend_cuda_host_buffer_type(),ggml_backend_cuda_host_buffer_type()};
                    auto sched=ggml_backend_sched_new(backends,bufts,3,256,false,true);require(sched,"test scheduler");
                    ggml_tensor * actual_x=x,*actual_ids=ids;
                    if(input_mode==2){
                        // Real decoder inputs are GPU intermediate tensors, not
                        // user INPUT leaves. Exercise that scheduler copy path too.
                        actual_x=ggml_scale(ctx,x,1.0f);actual_ids=ggml_cont(ctx,ids);
                        ggml_backend_sched_set_tensor_backend(sched,actual_x,outer_gpu);
                        ggml_backend_sched_set_tensor_backend(sched,actual_ids,outer_gpu);
                        require(!(actual_x->flags&GGML_TENSOR_FLAG_INPUT)&&!(actual_ids->flags&GGML_TENSOR_FLAG_INPUT),"intermediate flags");
                    }
                    auto y=xeon_hot_build(outer_cpu,sched,ctx,actual_x,actual_ids,weights[0],weights[1],weights[2],meta.clamp,layer_index);require(y,"custom build");
                    auto tail=ggml_scale(ctx,y,0.5f);xeon_hot_place_tail(outer_cpu,sched,tail);
                    auto graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,tail);
                    require(ggml_backend_sched_alloc_graph(sched,graph),"scheduler allocation");
                    require(ggml_backend_sched_get_tensor_backend(sched,y)==hot&&ggml_backend_sched_get_tensor_backend(sched,tail)==outer_gpu,"operator placement");
                    const std::array<int,TOP>order={4,1,5,0,3,2};
                    for(int h:{0,1,3,6})if(h<=resident)for(int repeat=0;repeat<3;++repeat){
                        std::vector<float>input(D),base(D*TOP),actual(D*TOP),scaled(D*TOP);for(auto & v:input)v=dist(rng);
                        std::array<int32_t,TOP>routes{};std::copy(cold.begin(),cold.end(),routes.begin());
                        for(int j=0;j<h;++j)routes[order[j]]=selected[(j+repeat)%resident];
                        ggml_backend_tensor_set(reference.x,input.data(),0,input.size()*4);ggml_backend_tensor_set(reference.ids,routes.data(),0,TOP*4);
                        require(ggml_backend_graph_compute(outer_cpu,reference.graph)==GGML_STATUS_SUCCESS,"CPU reference");ggml_backend_tensor_get(reference.y,base.data(),0,base.size()*4);
                        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(ids,routes.data(),0,TOP*4);
                        const auto before_hist=initialized.hist;const auto before_calls=initialized.calls;
                        require(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,"scheduled custom compute");ggml_backend_sched_synchronize(sched);
                        ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);ggml_backend_tensor_get(tail,scaled.data(),0,scaled.size()*4);
                        double err=0,norm=0;
                        for(int slot=0;slot<TOP;++slot){bool hit=initialized.local[routes[slot]]>=0;
                            if(!hit)require(!std::memcmp(base.data()+slot*D,actual.data()+slot*D,D*4),"cold scheduler slot bit mismatch");
                            for(int k=0;k<D;++k){size_t at=slot*D+k;require(std::isfinite(actual[at])&&scaled[at]==actual[at]*.5f,"finite output and GPU tail");double delta=double(actual[at])-base[at];err+=delta*delta;norm+=double(base[at])*base[at];}
                        }
                        require(norm>0&&err/norm<1e-2,"scheduled output numeric diagnostic");worst_nmse=std::max(worst_nmse,err/norm);++cases;outputs+=actual.size();
                        require(initialized.calls==before_calls+1,"exact custom call count");
                        for(int bucket=0;bucket<=TOP;++bucket)require(initialized.hist[bucket]==before_hist[bucket]+(bucket==h?1:0),"exact hot-count dispatch");
                    }
                    ggml_backend_sched_free(sched);ggml_free(ctx);ggml_backend_buffer_free(ibuf);ggml_free(input_ctx);
                }
                ggml_backend_free(hot);require(lookup(outer_cpu)==nullptr,"registry removed after free");
            }
        }
        ggml_free(weights_ctx);
    }
    ggml_backend_cpu_set_threadpool(outer_cpu,nullptr);ggml_threadpool_free(outer_pool);ggml_backend_free(outer_gpu);ggml_backend_free(outer_cpu);
    std::printf("{\"passed\":true,\"layer\":%d,\"cases\":%zu,\"outputs\":%zu,\"worst_nmse_vs_CPU\":%.12g,\"scope\":\"real weights; exact residentbytes,coldslots,GPUtail;scheduler CPU/GPU INPUT leaves andGPU intermediates;H0/1/3/6;registrylifecycle;not wholemodel quality\"}\n",layer_index,cases,outputs,worst_nmse);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
