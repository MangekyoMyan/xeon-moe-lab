#include PLUGIN_SOURCE

int main() {
    setenv("GGML_XV_TEST_MUTABLE","1",1);
    setenv("GGML_XV_TEST_SLOTS","3",1);
    setenv("GGML_XV_ADAPTIVE","1",1);
    ggml_tensor weights={};weights.type=GGML_TYPE_Q8_0;weights.ne[2]=16;
    std::strcpy(weights.name,"blk.1.ffn_gate_exps.weight");
    constexpr int tokens=513,stride=11;
    std::vector<int32_t> ids(tokens*stride,9999);
    std::vector<double> reference(16,0.0);
    for(int t=0;t<tokens;++t) {
        for(double & x:reference)x*=.99;
        for(int j=0;j<8;++j) {
            const int e=t%7==0?9:(17*t+3*j+(t%5==0?7:0))%16;
            ids[t*stride+j]=e;++reference[e];
        }
    }
    bool all=true;
    for(int chunk:{1,7,128,257})for(bool one_record:{false,true}) {
        config().histories.clear();
        int record=0;
        for(int first=0;first<tokens;first+=chunk) {
            ggml_tensor tensor={};tensor.type=GGML_TYPE_I32;tensor.ne[0]=8;
            tensor.ne[1]=std::min(chunk,tokens-first);tensor.nb[0]=sizeof(int32_t);
            tensor.nb[1]=stride*sizeof(int32_t);tensor.data=ids.data()+first*stride;
            adaptive_observe(&weights,&tensor,one_record?0:record++);
        }
        const auto & actual=config().histories.at(weights.name).counts;
        double error=0;for(int e=0;e<16;++e)error=std::max(error,std::abs(actual[e]-reference[e]));
        const bool pass=error<1e-10;all=all&&pass;
        std::printf("{\"chunk\":%d,\"one_record\":%s,\"max_error\":%.15g,\"passed\":%s}\n",chunk,one_record?"true":"false",error,pass?"true":"false");
    }
    return all?0:1;
}
