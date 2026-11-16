#include "route37.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstring>
#include <type_traits>

namespace {
constexpr size_t L=37, R=4096, A=512, C=128;
constexpr uint64_t ALL=(UINT64_C(1)<<L)-1;
struct Binding { const ggml_tensor * tensor; const void * data; ggml_type type; int64_t ne[4]; size_t nb[4]; };
struct Data {
    const void * owner; uint32_t context,rlimit,alimit,climit,flags;
    bool observing,call_open,eval_open;
    uint64_t nr,na,nc,dropped,ndecode,generation;
    Binding binding[L];
    xeon_route37_record records[R]; xeon_route37_attempt attempts[A]; xeon_route37_call calls[C];
};
static_assert(std::is_trivially_copyable<Data>::value,"prefault/reset POD only");
Data d{};
std::atomic<bool> enabled{false}, active{false};
static_assert(std::atomic<bool>::is_always_lock_free,"hook flag must not use a lock");
bool owns(const void *p){return enabled.load(std::memory_order_acquire)&&p&&p==d.owner;}
bool host(const ggml_tensor *t){auto b=t->buffer?t->buffer:(t->view_src?t->view_src->buffer:nullptr);return b&&ggml_backend_buffer_is_host(b);}
bool supported(ggml_type t){return t==GGML_TYPE_IQ2_S||t==GGML_TYPE_IQ3_S||t==GGML_TYPE_IQ3_XXS||t==GGML_TYPE_MXFP4;}
void mark(uint32_t bits){d.flags|=bits;}
bool readable(){return !active.load(std::memory_order_acquire)&&!d.eval_open&&!d.call_open;}
}
int xeon_route37_open_v1(const void *owner,uint32_t context,int observing,uint32_t nr,uint32_t na,uint32_t nc){
    if(!owner||!context||enabled.load()||active.load()||!nr||nr>R||!na||na>A||!nc||nc>C)return 0;
    // Fixed buffers are fully touched before any measured/observed graph.
    std::memset(&d,0,sizeof(d));d.owner=owner;d.context=context;d.observing=observing!=0;
    d.rlimit=nr;d.alimit=na;d.climit=nc;enabled.store(true,std::memory_order_release);return 1;
}
int xeon_route37_matches_v1(const void *owner){return owns(owner);}
int xeon_route37_label_v1(const void *owner,const xeon_route37_label *label){
    if(!owns(owner)||!label||d.call_open||d.eval_open||label->phase>XR_WARMUP)return 0;
    if(d.nc==d.climit){mark(XR_OVERFLOW);++d.dropped;return 0;}
    auto &c=d.calls[d.nc++];c={};c.label=*label;c.first_attempt=d.na;d.call_open=true;return 1;
}
int xeon_route37_begin_eval_v1(const void *owner,const ggml_tensor *const *gates,size_t count,
    int rebuilt,uint32_t nt,uint32_t npos,uint32_t nseq,int32_t seq,int64_t pos,int pipeline){
    if(!owns(owner))return 0;
    if(!d.call_open||d.eval_open){mark(XR_UNTAGGED);return 0;}
    if(d.na==d.alimit){mark(XR_OVERFLOW);++d.dropped;d.calls[d.nc-1].flags|=XR_OVERFLOW;return 0;}
    auto &c=d.calls[d.nc-1];auto &a=d.attempts[d.na++];a={};a.label=c.label;
    a.id=d.na;a.first_record=d.nr;a.context=d.context;a.ubatch=uint32_t(c.attempts++);
    if(rebuilt||!d.generation)++d.generation;
    a.generation=d.generation;a.position=pos;a.n_tokens=nt;a.observing=d.observing;
    a.scope_t1=nt==1;d.eval_open=true;
    if(pipeline||npos!=1||nseq!=1||seq!=a.label.sequence||pos<0||!nt)a.flags|=XR_GEOMETRY;
    if(a.label.phase==XR_DECODE&&nt!=1)a.flags|=XR_GEOMETRY;
    // Multi-token prefill is explicitly out of this collector's scope.
    if(count!=L||!gates)a.flags|=XR_MAPPING;
    if(!(a.flags&XR_MAPPING))for(size_t i=0;i<L;++i){
        const auto *w=gates[i];
        if(!w||!w->data||!host(w)||!supported(w->type)||w->ne[0]<=0||w->ne[0]%256||w->ne[1]<=0||w->ne[2]!=384||w->ne[3]!=1){a.flags|=XR_MAPPING;break;}
        for(size_t j=0;j<i;++j)if(gates[j]==w||gates[j]->data==w->data)a.flags|=XR_MAPPING;
        auto &b=d.binding[i];b.tensor=w;b.data=w->data;b.type=w->type;
        std::memcpy(b.ne,w->ne,sizeof(b.ne));std::memcpy(b.nb,w->nb,sizeof(b.nb));
    }
    mark(a.flags);
    // Release publishes the immutable label and bindings to CPU worker0.
    // The caller does not modify them until CPU work has returned.
    active.store(d.observing&&a.scope_t1&&!a.flags,std::memory_order_release);return 1;
}
void xeon_route37_record_v1(const ggml_tensor *w,const ggml_tensor *ids){
    if(!active.load(std::memory_order_acquire))return;
    auto &a=d.attempts[d.na-1];size_t layer=L;
    for(size_t i=0;i<L;++i)if(d.binding[i].tensor==w){layer=i;break;}
    if(layer==L){
        // Non-gate up/down weights are deliberately ignored. A copied gate
        // tensor sharing its backing data is explicitly invalid, not guessed.
        if(w)for(size_t i=0;i<L;++i)if(d.binding[i].data==w->data){a.flags|=XR_MAPPING;mark(XR_MAPPING);break;}
        return;
    }
    const auto &b=d.binding[layer];uint32_t flags=0;
    if(w->data!=b.data||w->type!=b.type||std::memcmp(w->ne,b.ne,sizeof(b.ne))||std::memcmp(w->nb,b.nb,sizeof(b.nb)))flags|=XR_MAPPING;
    const bool good=ids&&ids->data&&ids->type==GGML_TYPE_I32&&ids->ne[0]==6&&ids->ne[1]==1&&ids->ne[2]==1&&ids->ne[3]==1&&ids->nb[0]>=4;
    if(!good)flags|=XR_GEOMETRY;
    const uint64_t bit=UINT64_C(1)<<layer;if(a.layers&bit)flags|=XR_DUPLICATE;a.layers|=bit;
    if(d.nr==d.rlimit){a.flags|=flags|XR_OVERFLOW;mark(flags|XR_OVERFLOW);++d.dropped;return;}
    auto &r=d.records[d.nr];r={};r.ordinal=d.nr++;r.attempt=a.id;r.request=a.label.request;r.sequence=a.label.sequence;
    r.decode_call=a.label.decode_call;r.ubatch=a.ubatch;r.logical_token=a.label.logical_token;r.position=a.position;
    r.graph_generation=a.generation;r.context=a.context;r.layer=uint16_t(layer);r.phase=a.label.phase;r.projection=1;r.n_tokens=a.n_tokens;
    for(int j=0;j<6;++j){r.ids[j]=-1;if(good)std::memcpy(&r.ids[j],static_cast<const char*>(ids->data)+size_t(j)*ids->nb[0],4);
        if(r.ids[j]<0||r.ids[j]>=384)flags|=XR_IDS;
        for(int k=0;k<j;++k)if(r.ids[k]==r.ids[j])flags|=XR_IDS;
    }
    r.flags=flags;a.flags|=flags;mark(flags);
}
int xeon_route37_end_eval_v1(const void *owner,int32_t status){
    if(!owns(owner)||!d.eval_open)return 0;
    // Preconditions: CPU graph calls and all worker hooks have returned.
    active.store(false,std::memory_order_release);
    auto &a=d.attempts[d.na-1];a.records=d.nr-a.first_record;a.graph_status=status;a.graph_returned=1;
    if(status!=GGML_STATUS_SUCCESS)a.flags|=XR_GRAPH_FAILED;
    if(d.observing&&a.scope_t1&&(a.records!=L||a.layers!=ALL))a.flags|=XR_COVERAGE;
    mark(a.flags);d.eval_open=false;return !a.flags;
}
int xeon_route37_complete_call_v1(const void *owner,int32_t status,int synced){
    if(!owns(owner)||!d.call_open||d.eval_open)return 0;
    auto &c=d.calls[d.nc-1];c.status=status;c.synchronized=synced!=0;c.completed=1;
    if(status)c.flags|=XR_CALL_FAILED;
    if(!synced)c.flags|=XR_UNSYNCHRONIZED;
    if(!c.attempts)c.flags|=XR_NO_EVAL;
    for(size_t i=c.first_attempt;i<d.na;++i){auto &a=d.attempts[i];c.flags|=a.flags;if(!a.graph_returned)c.flags|=XR_GRAPH_FAILED;}
    if(!c.flags)for(size_t i=c.first_attempt;i<d.na;++i){auto &a=d.attempts[i];a.caller_completed=1;if(d.observing&&a.scope_t1&&a.label.phase==XR_DECODE)++d.ndecode;}
    mark(c.flags);d.call_open=false;return !c.flags;
}
int xeon_route37_close_v1(const void *owner){if(!owns(owner)||!readable())return 0;enabled.store(false,std::memory_order_release);return 1;}
int xeon_route37_stats_v1(xeon_route37_stats *out){
    if(!out||!readable())return 0;
    *out={d.nr,d.na,d.nc,d.dropped,d.ndecode,d.flags,uint8_t(enabled.load()),uint8_t(d.call_open),uint8_t(d.eval_open),uint8_t(d.observing)};return 1;
}
int xeon_route37_read_record_v1(size_t i,xeon_route37_record *out){if(!out||!readable()||i>=d.nr)return 0;*out=d.records[i];return 1;}
int xeon_route37_read_attempt_v1(size_t i,xeon_route37_attempt *out){if(!out||!readable()||i>=d.na)return 0;*out=d.attempts[i];return 1;}
int xeon_route37_read_call_v1(size_t i,xeon_route37_call *out){if(!out||!readable()||i>=d.nc)return 0;*out=d.calls[i];return 1;}
