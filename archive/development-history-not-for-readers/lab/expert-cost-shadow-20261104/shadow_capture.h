#pragma once
// Optional metadata-only observer. No CUDA calls and no changes to graph assignment.
#include "ggml.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace xeon_shadow {
constexpr size_t max_experts=512, capacity=128, max_records=2048, max_bytes=8*1024*1024;
using clock=std::chrono::steady_clock;
inline uint64_t ns(){return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();}
enum skip_reason {empty_graph,other_node,empty_ids,native_cache,unsupported_route,skip_count};
struct record {
    uint64_t operation=0,route=0,session=0,submitted_bytes=0,copy_calls=0,fetch_wait_ns=0,callback_ns=0;
    int64_t experts=0,tokens=0,topk=0,K=0,M=0;size_t expert_bytes=0;
    bool refreshed=false;std::array<uint32_t,max_experts> counts{};
    char tensor[GGML_MAX_NAME]{},quant[24]{};
};
struct recorder {
    std::atomic<bool> enabled{false};std::atomic<uint64_t> operations{0},routes{0},dropped{0},accepted{0},attempted{0};
    std::array<std::atomic<uint64_t>,skip_count> skipped{};
    std::array<record,capacity> queue{};size_t head=0,tail=0,used=0,written=0,bytes=0;
    std::mutex mutex;std::condition_variable wake;int fd=-1;uint64_t session=0;std::thread worker;
    bool stop=false,window_reported=false,io_failed=false;uint64_t write_dropped=0;
    recorder() noexcept {
        const char *path=std::getenv("XEON_SHADOW_LOG");if(!path||!*path)return;
        // A fresh regular file only: never append to a key, FIFO, prior capture, or symlink target.
        fd=::open(path,O_WRONLY|O_CREAT|O_EXCL|O_NONBLOCK|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)return;
        struct stat st;if(::fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_uid!=::geteuid()||(st.st_mode&077)||st.st_nlink!=1){::close(fd);fd=-1;return;}
        if(::flock(fd,LOCK_EX|LOCK_NB)){::close(fd);fd=-1;return;}
        const auto size=::lseek(fd,0,SEEK_END);if(size<0 || size_t(size)>=max_bytes){::close(fd);fd=-1;return;}bytes=size_t(size);
        session=uint64_t(std::chrono::system_clock::now().time_since_epoch().count());
        enabled.store(true);
        try{worker=std::thread([this]{consume();});}catch(...){enabled.store(false);::close(fd);fd=-1;}
    }
    bool write_all(const char *data,size_t length) noexcept {
        size_t done=0;
        while(done<length){auto n=::write(fd,data+done,length-done);if(n<0&&errno==EINTR)continue;if(n<=0){io_failed=true;return false;}done+=size_t(n);bytes+=size_t(n);}
        return true;
    }
    ~recorder(){
        {std::lock_guard<std::mutex> lock(mutex);stop=true;}wake.notify_one();
        if(worker.joinable()){worker.join();}
        if(fd>=0){::close(fd);}
    }
    static void escaped(char *out,size_t &n,size_t cap,const char *text){
        for(size_t i=0;text[i] && i<GGML_MAX_NAME && n+7<cap;++i){unsigned char c=text[i];
            if(c=='"'||c=='\\'){out[n++]='\\';out[n++]=c;}else if(c>=32&&c<127)out[n++]=c;
            else n+=std::snprintf(out+n,cap-n,"\\u%04x",unsigned(c));
        }
    }
    bool write_record(const record &r) noexcept {
        char out[8192];size_t n=std::snprintf(out,sizeof(out),
            "{\"schema\":1,\"kind\":\"route_observation\",\"mode\":\"shadow_only\",\"pid\":%ld,\"session\":%llu,\"operation\":%llu,\"route_epoch\":%llu,\"current_assignment\":\"gpu\",\"assignment_changed\":false,\"feature_stage\":\"host_routes_ready\",\"decision_stage\":\"post_assignment_shadow\",\"unique_routes_validated\":true,\"native_moe_cache\":false,\"host_weights_ready\":true,\"host_registration_verified\":null,\"gpu_weight_resident\":false,\"cpu_queue_ms\":null,\"gpu_queue_ms\":null,\"tensor\":\"",
            long(getpid()),(unsigned long long)r.session,(unsigned long long)r.operation,(unsigned long long)r.route);
        escaped(out,n,sizeof(out),r.tensor);
        n+=std::snprintf(out+n,sizeof(out)-n,"\",\"quant\":\"%s\",\"K\":%lld,\"M\":%lld,\"experts\":%lld,\"tokens\":%lld,\"topk\":%lld,\"expert_bytes\":%zu,\"route_refreshed\":%s,\"shared_fetch_wait_ms\":%.6f,\"copy_callback_host_ms\":%.6f,\"submitted_H2D_bytes\":%llu,\"submitted_copy_calls\":%llu,\"dropped_before\":%llu,\"histogram\":[",
            r.quant,(long long)r.K,(long long)r.M,(long long)r.experts,(long long)r.tokens,(long long)r.topk,r.expert_bytes,r.refreshed?"true":"false",r.fetch_wait_ns/1e6,r.callback_ns/1e6,
            (unsigned long long)r.submitted_bytes,(unsigned long long)r.copy_calls,(unsigned long long)dropped.load());
        for(int64_t e=0;e<r.experts;++e)n+=std::snprintf(out+n,sizeof(out)-n,"%s%u",e?",":"",r.counts[e]);
        n+=std::snprintf(out+n,sizeof(out)-n,"]}\n");
        if(n>=sizeof(out)||bytes+n>max_bytes-2048)return false;
        return write_all(out,n);
    }
    void consume() noexcept {
        while(true){record item;
            {std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[this]{return used>0||stop||(!enabled.load()&&!window_reported);});
                if(used==0&&stop)break;
                if(used==0&&!enabled.load()){lock.unlock();status("capture_window_closed",false);window_reported=true;continue;}
                item=queue[head];head=(head+1)%capacity;--used;
            }
            if(!enabled.load()) {++write_dropped;continue;}
            if(written>=max_records || !write_record(item)){enabled.store(false);++write_dropped;continue;}
            ++written;
        }
        enabled.store(false);status("shutdown",true);
        if(io_failed)std::fprintf(stderr,"xeon_shadow: incomplete capture after file I/O failure\n");
    }
    void status(const char *reason,bool final) noexcept {
        if(fd<0||io_failed){return;}
        char out[1024];
        int n=std::snprintf(out,sizeof(out),"{\"schema\":1,\"kind\":\"capture_status\",\"reason\":\"%s\",\"final\":%s,\"pid\":%ld,\"session\":%llu,\"written\":%zu,\"attempted\":%llu,\"accepted\":%llu,\"enqueue_dropped\":%llu,\"write_dropped\":%llu,\"skipped\":[%llu,%llu,%llu,%llu,%llu],\"coverage\":\"eligible callbacks until capture window closes; CPU and GPU-resident operations unobserved; missing final footer means incomplete capture\"}\n",reason,final?"true":"false",long(getpid()),(unsigned long long)session,written,(unsigned long long)attempted.load(),(unsigned long long)accepted.load(),(unsigned long long)dropped.load(),(unsigned long long)write_dropped,(unsigned long long)skipped[0].load(),(unsigned long long)skipped[1].load(),(unsigned long long)skipped[2].load(),(unsigned long long)skipped[3].load(),(unsigned long long)skipped[4].load());
        if(n>0&&size_t(n)<sizeof(out)&&bytes+size_t(n)<=max_bytes)write_all(out,n);
    }
    void submit(record &item) noexcept {
        ++attempted;
        if(!enabled.load()){++dropped;return;}
        if(!mutex.try_lock()){++dropped;return;}
        if(!enabled.load()){mutex.unlock();++dropped;return;}
        if(used==capacity){mutex.unlock();++dropped;return;}
        queue[tail]=item;tail=(tail+1)%capacity;++used;++accepted;mutex.unlock();wake.notify_one();
    }
};
inline recorder & instance(){static recorder value;return value;}
inline bool requested() noexcept {static const bool value=[](){const char*p=std::getenv("XEON_SHADOW_LOG");return p&&*p;}();return value;}
inline void initialize() noexcept {if(requested())try{(void)instance();}catch(...){}}
inline bool enabled() noexcept {if(!requested())return false;try{return instance().enabled.load();}catch(...){return false;}}
inline void skipped(skip_reason reason) noexcept {if(enabled())++instance().skipped[reason];}

struct cached_route {
    const void *owner=nullptr,*ids=nullptr;int64_t experts=0,tokens=0,topk=0;uint64_t epoch=0;
    size_t stride_slot=0,stride_token=0;uint64_t generation=0;
    std::array<uint32_t,max_experts> counts{};
};
inline cached_route & route_cache() noexcept {static thread_local cached_route value;return value;}
inline std::atomic<uint64_t> & route_generation() noexcept {static std::atomic<uint64_t> value{0};return value;}
// Called on every actual shared-ID refresh, including callbacks excluded from logging.
// A process-wide generation also invalidates caches if a context migrates between host threads.
inline void invalidate_route() noexcept {route_generation().fetch_add(1,std::memory_order_acq_rel);auto &c=route_cache();c.owner=nullptr;c.ids=nullptr;}
struct capture {
    bool active=false;record row;uint64_t start=0;
    void begin(const void *owner,const ggml_tensor *ids,const void *data,size_t bytes,const ggml_tensor *weights,
               bool refreshed,uint64_t fetch_ns,uint64_t callback_start) noexcept {
        if(!enabled())return;
        auto &cached=route_cache();
        if(refreshed){invalidate_route();}
        const uint64_t generation=route_generation().load(std::memory_order_acquire);
        const int64_t E=weights->ne[2],T=ids->ne[1],top=ids->ne[0];
        if(!data||E<1||E>int64_t(max_experts)||T<1||T>65536||top<1||top>64||top>E||ids->ne[2]!=1||ids->ne[3]!=1||ids->type!=GGML_TYPE_I32||weights->ne[3]!=1||weights->ne[0]<1||weights->ne[1]<1){skipped(unsupported_route);return;}
        if(ids->nb[0]<4||ids->nb[1]<4||(T>1&&ids->nb[1]>bytes/size_t(T-1))||(top>1&&ids->nb[0]>bytes/size_t(top-1))){skipped(unsupported_route);return;}
        const size_t endrow=size_t(T-1)*ids->nb[1],endcol=size_t(top-1)*ids->nb[0];
        if(endrow>bytes||endcol>bytes-endrow||bytes-endrow-endcol<4){skipped(unsupported_route);return;}
        // The actual refresh flag is authoritative; pointer reuse after graph reset never reuses old counts.
        if(refreshed||cached.generation!=generation||cached.owner!=owner||cached.ids!=ids||cached.experts!=E||cached.tokens!=T||cached.topk!=top||cached.stride_slot!=ids->nb[0]||cached.stride_token!=ids->nb[1]){
            cached.owner=nullptr;cached.ids=nullptr;
            std::array<uint32_t,max_experts> counts{};std::array<int64_t,max_experts> seen;seen.fill(-1);
            for(int64_t t=0;t<T;++t)for(int64_t j=0;j<top;++j){
                const size_t offset=size_t(t)*ids->nb[1]+size_t(j)*ids->nb[0];
                if(offset>bytes||bytes-offset<4){skipped(unsupported_route);return;}
                int32_t e;std::memcpy(&e,static_cast<const uint8_t*>(data)+offset,4);
                if(e<0||e>=E||seen[e]==t){skipped(unsupported_route);return;}seen[e]=t;++counts[e];
            }
            cached.owner=owner;cached.ids=ids;cached.experts=E;cached.tokens=T;cached.topk=top;
            cached.stride_slot=ids->nb[0];cached.stride_token=ids->nb[1];
            cached.generation=generation;
            cached.counts=counts;cached.epoch=++instance().routes;
        }
        row=record{};row.operation=++instance().operations;row.route=cached.epoch;row.session=instance().session;
        row.experts=E;row.tokens=T;row.topk=top;row.K=weights->ne[0];row.M=weights->ne[1];row.expert_bytes=weights->nb[2];
        row.refreshed=refreshed;row.fetch_wait_ns=refreshed?fetch_ns:0;row.counts=cached.counts;
        std::snprintf(row.tensor,sizeof(row.tensor),"%s",weights->name);std::snprintf(row.quant,sizeof(row.quant),"%s",ggml_type_name(weights->type));
        start=callback_start;active=true;
    }
    void submitted(size_t bytes) noexcept {if(active){row.submitted_bytes+=bytes;++row.copy_calls;}}
    void finish() noexcept {if(active){row.callback_ns=ns()-start;instance().submit(row);active=false;}}
};
}
