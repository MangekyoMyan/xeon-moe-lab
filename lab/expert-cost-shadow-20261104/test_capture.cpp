#include "shadow_capture.h"
#include <cstdio>
#include <vector>
#include <string>
int main(int argc,char **argv){
    xeon_shadow::initialize();ggml_tensor ids{},weights{};ids.type=GGML_TYPE_I32;ids.ne[0]=2;ids.ne[1]=3;ids.ne[2]=ids.ne[3]=1;ids.nb[0]=4;ids.nb[1]=16;
    weights.type=GGML_TYPE_MXFP4;weights.ne[0]=32;weights.ne[1]=16;weights.ne[2]=8;weights.ne[3]=1;weights.nb[2]=272;
    std::snprintf(weights.name,sizeof(weights.name),"blk.0.ffn_up_exps.weight");
    std::vector<uint8_t> raw(40);int values[6]={0,1,2,3,4,5};int owner=0;
    auto fill=[&]{for(int t=0;t<3;++t)for(int j=0;j<2;++j){int32_t e=values[t*2+j];std::memcpy(raw.data()+16*t+4*j,&e,4);}};fill();
    auto emit=[&](bool refresh){xeon_shadow::capture c;c.begin(&owner,&ids,raw.data(),raw.size(),&weights,refresh,refresh?10000:0,xeon_shadow::ns());c.submitted(123);c.finish();};
    if(argc>1&&std::string(argv[1])=="cap"){
        for(int i=0;i<2300;++i){emit(true);std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    }else{
        emit(true);emit(false);values[0]=7;fill();emit(true);
        values[0]=-1;fill();emit(true);values[0]=6;fill();emit(false);
        // An excluded callback refreshes shared IDs; next eligible callback reuses their pointer.
        values[0]=0;fill();xeon_shadow::invalidate_route();emit(false);
        values[0]=7;fill();std::thread([]{xeon_shadow::invalidate_route();}).join();emit(false);
        xeon_shadow::skipped(xeon_shadow::empty_graph);
    }
    std::puts("capture fixture completed");
}
