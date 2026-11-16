#include "fixture.h"
#include "call_control.h"
#include <iostream>
static xeon_route37_label tag(int call,int phase=XR_DECODE){xeon_route37_label t{};t.request=1;t.sequence=5;t.decode_call=call;t.phase=phase;t.logical_token=call;return t;}
static void emit(API &a,Fixture &f,int call,int status=GGML_STATUS_SUCCESS,int layers=37,bool limit=false){
    need(a.begin(&f,f.gates.data(),37,call==0,1,1,1,5,call,0),"begin");
    for(int i=0;i<layers;++i)a.record(f.gates[i],f.ids[i]);
    int valid=a.end(&f,status);need(bool(valid)==(status==GGML_STATUS_SUCCESS&&layers==37&&!limit),"end result");
}
int main(int argc,char **argv)try{
    need(argc==2,"output directory");std::filesystem::path dir=argv[1];need(std::filesystem::create_directory(dir),"fresh directory required");
    save_json(dir/"providers.json",{{"backend_autoload_called",false},{"providers",providers()}});
    API a;Fixture f;ggml_backend_t backends[]={f.cpu};auto sched=ggml_backend_sched_new(backends,nullptr,1,256,false,true);need(sched,"scheduler");
    need(a.open(&f,1,1,4096,512,128),"open scheduler session");
    for(int cycle=0;cycle<3;++cycle){
        f.inputs(cycle);auto reference=f.run();
        if(cycle==2){ggml_backend_sched_reset(sched);auto *g=ggml_new_graph_custom(f.ctx,256,false);for(auto *y:f.out)ggml_build_forward_expand(g,y);f.graph=g;}
        if(cycle!=1)need(ggml_backend_sched_alloc_graph(sched,f.graph),"scheduler allocation");
        for(int i=0;i<37;++i)need(f.out[i]->src[0]==f.gates[i]&&f.out[i]->src[0]->data==f.gates[i]->data,"scheduler gate identity");
        auto t=tag(cycle);need(a.label(&f,&t),"scheduler label");need(a.begin(&f,f.gates.data(),37,cycle!=1,1,1,1,5,cycle,0),"scheduler begin");
        auto status=ggml_backend_sched_graph_compute_async(sched,f.graph);need(a.end(&f,status),"scheduler producer completion");
        ggml_backend_sched_synchronize(sched);need(status==GGML_STATUS_SUCCESS&&a.finish(&f,0,1),"scheduler caller completion");
        std::vector<float>out(reference.size());for(int i=0;i<37;++i)ggml_backend_tensor_get(f.out[i],out.data()+i*24,0,24*sizeof(float));need(!std::memcmp(reference.data(),out.data(),out.size()*4),"scheduler numerical mismatch");
    }
    auto trace=snapshot(a);need(trace["stats"]["records"]==111&&trace["stats"]["flags"]==0,"scheduler coverage");
    for(int i=0;i<3;++i){need(trace["attempts"][i]["id"]==i+1,"attempt");need(trace["attempts"][i]["generation"]==(i==2?2:1),"generation");}
    need(a.close(&f),"scheduler close");save_json(dir/"scheduler.json",trace);ggml_backend_sched_free(sched);
    json cases=json::array();f.inputs(0);
    for(int kind=0;kind<8;++kind){
        need(a.open(&f,1,1,kind==6?36:4096,512,kind==7?1:128),"caller session");std::vector<std::string>order;
        struct Logged { API &a;std::vector<std::string>&o;int label(const void*p,const xeon_route37_label*t){o.push_back("label");return a.label(p,t);}int finish(const void*p,int s,int sync){o.push_back("complete");return a.finish(p,s,sync);} } logged{a,order};
        auto t=tag(0,kind==3?XR_PREFILL:kind==4?XR_UNKNOWN:XR_DECODE);
        auto result=drive_call(logged,&f,t,[&](){order.push_back("submit");if(kind==1)return -2;if(kind==2){emit(a,f,0,GGML_STATUS_ABORTED,7);return 2;}emit(a,f,0,GGML_STATUS_SUCCESS,37,kind==6);return 0;},[&](){order.push_back("output_sync");return kind!=5;},[&](){order.push_back("drain");});
        const bool expected=kind==0||kind==3||kind==4||kind==7;need(result.ok()==expected,"caller result");
        need(order==std::vector<std::string>({"label","submit",kind==1||kind==2?"drain":"output_sync","complete"}),"cleanup order");
        if(kind==1||kind==2){auto retry=drive_call(logged,&f,tag(1),[&](){emit(a,f,1);return 0;},[](){return true;},[](){});need(retry.ok(),"retry");}
        if(kind==7){bool called=false;auto rejected=drive_call(logged,&f,tag(1),[&](){called=true;return 0;},[](){return true;},[](){});need(!rejected.label_accepted&&!called&&!rejected.finalize_called,"limit stop");}
        auto state=snapshot(a);if(kind==1)need(state["calls"][0]["attempts"]==0,"no-eval failure");
        if(kind==2)need(state["attempts"][0]["graph_status"]==GGML_STATUS_ABORTED&&state["calls"][0]["public_or_driver_status"]==2,"status namespaces");
        if(kind==3||kind==4)need(state["stats"]["complete_decode_attempts"]==0,"phase count");
        need(a.close(&f),"caller close");cases.push_back({{"case",kind},{"passed",true},{"callback_order",order},{"output_valid",result.output_valid},{"trace",state}});
    }
    save_json(dir/"caller_cases.json",cases);
    std::cout<<json({{"passed",true},{"scheduler_graphs",3},{"scheduler_records",111},{"scalar_comparisons_to_direct_CPU",2664},{"caller_cases",8},{"scope","Real CPU-only scheduler fixture; caller cleanup uses fake submit/output/drain and direct hooks, no model/GPU"}}).dump()<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
