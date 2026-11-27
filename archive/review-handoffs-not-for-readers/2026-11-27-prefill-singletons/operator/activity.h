// CUPTI is initialized before touching CUDA. The uninstrumented timing process
// does not enable CUPTI activity or consult its clock.
#include <cupti.h>
#include <cstdlib>
#include <map>
#include <set>
namespace activity {
static bool enabled=false;
static std::atomic<bool> keep{false};
static std::mutex mutex;
struct Event { uint64_t start,end,bytes; uint32_t kind,detail,corr,stream; std::string name; };
static std::vector<Event> events;
static size_t dropped=0, errors=0;
static uint64_t clock_ns(){
    if(!enabled)return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    uint64_t t=0;need(cuptiGetTimestamp(&t)==CUPTI_SUCCESS,"CUPTI clock");return t;
}
static void CUPTIAPI request(uint8_t **buf,size_t *size,size_t *max){
    *size=2*1024*1024;*max=0;*buf=nullptr;
    if(posix_memalign(reinterpret_cast<void**>(buf),8,*size)!=0)std::abort();
}
static void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t *buf,size_t,size_t valid){
    std::lock_guard<std::mutex> lock(mutex);
    if(keep.load()){
        CUpti_Activity *r=nullptr;CUptiResult rc;
        while((rc=cuptiActivityGetNextRecord(buf,valid,&r))==CUPTI_SUCCESS){
            if(r->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
                const auto *e=(CUpti_ActivityMemcpy6*)r;events.push_back({e->start,e->end,e->bytes,1,e->copyKind,e->correlationId,e->streamId,""});
            }else if(r->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
                const auto *e=(CUpti_ActivityKernel9*)r;events.push_back({e->start,e->end,0,2,0,e->correlationId,e->streamId,e->name?e->name:""});
            }else if(r->kind==CUPTI_ACTIVITY_KIND_RUNTIME || r->kind==CUPTI_ACTIVITY_KIND_DRIVER){
                const auto *e=(CUpti_ActivityAPI*)r;events.push_back({e->start,e->end,0,(uint32_t)r->kind,e->cbid,e->correlationId,e->threadId,""});
            }
        }
        if(valid && rc!=CUPTI_ERROR_MAX_LIMIT_REACHED)++errors;
        size_t count=0;if(cuptiActivityGetNumDroppedRecords(context,stream,&count)!=CUPTI_SUCCESS)++errors;
        dropped+=count;
    }
    std::free(buf);
}
static void init(bool active){
    enabled=active;if(!enabled)return;
    need(cuptiActivityRegisterCallbacks(request,complete)==CUPTI_SUCCESS,"activity callbacks");
    need(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)==CUPTI_SUCCESS,"kernel activity");
}
static void begin(){
    need(enabled && !keep,"trace window");
    need(cuptiActivityFlushAll(0)==CUPTI_SUCCESS,"flush warmup");
    for(auto k:{CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_RUNTIME,CUPTI_ACTIVITY_KIND_DRIVER})
        need(cuptiActivityEnable(k)==CUPTI_SUCCESS,"activity enable");
    keep=true;
}
static void end(const fs::path &file){
    need(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)==CUPTI_SUCCESS,"activity flush");
    keep=false;
    for(auto k:{CUPTI_ACTIVITY_KIND_MEMCPY,CUPTI_ACTIVITY_KIND_RUNTIME,CUPTI_ACTIVITY_KIND_DRIVER})
        need(cuptiActivityDisable(k)==CUPTI_SUCCESS,"activity disable");
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream f(file);f<<"start_ns\tend_ns\tkind\tdetail\tbytes\tcorrelation\tstream_or_tid\tname\n";
    for(const auto &e:events)f<<e.start<<'\t'<<e.end<<'\t'<<e.kind<<'\t'<<e.detail<<'\t'<<e.bytes<<'\t'<<e.corr<<'\t'<<e.stream<<'\t'<<e.name<<'\n';
    f.close();need(!f.fail()&&!events.empty()&&!dropped&&!errors,"complete CUPTI output");
    events.clear();
}
}
