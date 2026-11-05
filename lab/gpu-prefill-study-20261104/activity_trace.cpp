#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
static FILE * log_file=nullptr;
static void check(CUptiResult r){if(r!=CUPTI_SUCCESS){const char *s=nullptr;cuptiGetResultString(r,&s);std::fprintf(stderr,"CUPTI %s\n",s?s:"error");std::abort();}}
static void CUPTIAPI requested(uint8_t **p,size_t *size,size_t *records){*size=4*1024*1024;*records=0;void *buf=nullptr;if(posix_memalign(&buf,8,*size))std::abort();*p=static_cast<uint8_t *>(buf);}
static void CUPTIAPI completed(CUcontext ctx,uint32_t stream,uint8_t *buf,size_t,size_t valid){
    CUpti_Activity *rec=nullptr;
    while(true){auto r=cuptiActivityGetNextRecord(buf,valid,&rec);if(r==CUPTI_ERROR_MAX_LIMIT_REACHED)break;check(r);
        if(rec->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){auto k=reinterpret_cast<CUpti_ActivityKernel9 *>(rec);std::fprintf(log_file,"K\t%llu\t%llu\t%u\t%s\n",(unsigned long long)k->start,(unsigned long long)k->end,k->correlationId,k->name);}
        else if(rec->kind==CUPTI_ACTIVITY_KIND_MEMCPY){auto m=reinterpret_cast<CUpti_ActivityMemcpy6 *>(rec);std::fprintf(log_file,"M\t%llu\t%llu\t%u\t%u\t%llu\n",(unsigned long long)m->start,(unsigned long long)m->end,m->correlationId,(unsigned)m->copyKind,(unsigned long long)m->bytes);}
    }
    size_t dropped=0;check(cuptiActivityGetNumDroppedRecords(ctx,stream,&dropped));if(dropped)std::fprintf(log_file,"DROPPED\t%zu\n",dropped);std::free(buf);
}
extern "C" void xeon_trace_begin(){
    const char *path=std::getenv("XEON_ACTIVITY_FILE");if(!path)std::abort();log_file=std::fopen(path,"w");if(!log_file)std::abort();
    check(cuptiActivityRegisterCallbacks(requested,completed));check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    uint64_t ts;check(cuptiGetTimestamp(&ts));std::fprintf(log_file,"BEGIN\t%llu\n",(unsigned long long)ts);
}
extern "C" void xeon_trace_end(){
    uint64_t ts;check(cuptiGetTimestamp(&ts));std::fprintf(log_file,"END\t%llu\n",(unsigned long long)ts);
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));std::fflush(log_file);
}
