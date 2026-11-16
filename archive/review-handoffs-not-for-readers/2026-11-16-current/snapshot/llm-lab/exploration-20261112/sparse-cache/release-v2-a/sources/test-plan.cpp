#include "selection-plan.h"
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>

static void need(bool ok,const char * message) { if (!ok) throw std::runtime_error(message); }
int main() try {
    auto ctx = ggml_init({1024*1024,nullptr,true}); need(ctx,"context");
    auto a = ggml_new_tensor_3d(ctx,GGML_TYPE_IQ3_XXS,256,17,384);
    auto b = ggml_new_tensor_3d(ctx,GGML_TYPE_IQ2_S,256,17,384);
    ggml_set_name(a,"blk.0.ffn_gate_exps.weight"); ggml_set_name(b,"blk.0.ffn_up_exps.weight");
    char aa,bb; a->data=&aa; b->data=&bb;
    std::vector<const ggml_tensor *> tensors={a,b};
    const std::string good="XEON_IQ_CACHE_PLAN_V2 2 3 8192\nblk.0.ffn_gate_exps.weight iq3_xxs 256 17 384 3 0 17 383\nblk.0.ffn_up_exps.weight iq2_s 256 17 384 0\n";
    auto plan=xeon_iq3::parse_plan(good,tensors,8192,4096);
    need(plan.selected==3&&plan.bytes==8192&&plan.experts[0]==std::vector<uint32_t>({0,17,383})&&plan.experts[1].empty(),"valid sparse plan");
    size_t rejected=0;
    auto reject=[&](const std::string & text,size_t budget=8192) {
        bool failed=false; try { (void)xeon_iq3::parse_plan(text,tensors,budget,4096); } catch(const std::exception &) { failed=true; }
        need(failed,"invalid plan accepted");++rejected;
    };
    auto change=[&](std::string from,std::string to) {
        auto text=good;const auto pos=text.find(from);need(pos!=std::string::npos,"test replacement");text.replace(pos,from.size(),to);reject(text);
    };
    reject(good,8191);
    change("PLAN_V2","PLAN_V1");change("2 3 8192","3 3 8192");change("2 3 8192","2 4 8192");change("2 3 8192","2 3 4096");
    change(" 256 17 384 3"," 512 17 384 3");change(" 256 17 384 3"," 256 18 384 3");change(" 256 17 384 3"," 256 17 383 3");
    change("iq3_xxs","iq3_s");change("3 0 17 383","3 0 0 383");change("3 0 17 383","3 17 0 383");change("3 0 17 383","3 0 17 384");
    change("3 0 17 383","3 -1 17 383");change("3 0 17 383","3 0x0 17 383");change("3 0 17 383","3 0 17 18446744073709551616");
    change("blk.0.ffn_up_exps.weight iq2_s","blk.0.ffn_gate_exps.weight iq3_xxs");change("blk.0.ffn_up_exps.weight","blk.9.ffn_up_exps.weight");
    reject(good+"extra");reject(good.substr(0,good.size()-4));reject(std::string(1024*1024+1,' '));
    b->data=a->data;reject(good);b->data=&bb;
    const std::string zero="XEON_IQ_CACHE_PLAN_V2 2 0 0\nblk.0.ffn_up_exps.weight iq2_s 256 17 384 0\nblk.0.ffn_gate_exps.weight iq3_xxs 256 17 384 0\n";
    plan=xeon_iq3::parse_plan(zero,tensors,8192,4096);need(plan.bytes==0&&plan.selected==0,"empty plan");
    ggml_free(ctx);
    std::printf("{\"valid_plans\":2,\"invalid_plans_rejected\":%zu,\"pass\":true}\n",rejected);
} catch(const std::exception & error) { std::fprintf(stderr,"FAIL: %s\n",error.what());return 1; }
