#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <unistd.h>
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static void need(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
extern "C" void split_gather(const float*,const int32_t*,float*,int,int,int,cudaStream_t);
extern "C" void split_scatter(const float*,const int32_t*,float*,int,int,cudaStream_t);
extern "C" void split_merge(const float*,const float*,float*,int,int,int,cudaStream_t);
constexpr int TOP=6;
struct TensorInfo {std::string path,name;ggml_type q;int64_t K=0,M=0,E=0;size_t offset=0,expertbytes=0;};
struct Metadata {std::array<TensorInfo,3>w;float clamp=0;};
static Metadata metadata(const std::string &directory,int layer){
    Metadata result;std::array<bool,3> found{};bool clamp_found=false;
    std::vector<std::filesystem::path>files;for(auto &p:std::filesystem::directory_iterator(directory))if(p.path().extension()==".gguf")files.push_back(p.path());std::sort(files.begin(),files.end());
    const char*parts[]={"gate","up","down"};
    for(auto &path:files){auto m=gguf_init_from_file(path.c_str(),{true,nullptr});need(m,"GGUF metadata");
        for(int64_t j=0;j<gguf_get_n_kv(m);++j){std::string key=gguf_get_key(m,j);if(key.find(".swiglu_clamp_exp")!=std::string::npos){
            if(gguf_get_kv_type(m,j)==GGUF_TYPE_ARRAY){need(gguf_get_arr_type(m,j)==GGUF_TYPE_FLOAT32&&size_t(layer)<gguf_get_arr_n(m,j),"clamp array");result.clamp=static_cast<const float*>(gguf_get_arr_data(m,j))[layer];}
            else {need(gguf_get_kv_type(m,j)==GGUF_TYPE_FLOAT32,"clamp scalar");result.clamp=gguf_get_val_f32(m,j);}clamp_found=true;
        }}
        for(int j=0;j<3;++j){std::string name="blk."+std::to_string(layer)+".ffn_"+parts[j]+"_exps.weight";auto id=gguf_find_tensor(m,name.c_str());if(id<0)continue;
            need(!found[j],"duplicate tensor");const auto*n=gguf_get_tensor_ne(m,id);need(n[3]==1,"weight fourth dimension");auto q=gguf_get_tensor_type(m,id);
            result.w[j]={path.string(),name,q,n[0],n[1],n[2],gguf_get_data_offset(m)+gguf_get_tensor_offset(m,id),ggml_row_size(q,n[0])*size_t(n[1])};found[j]=true;
        }gguf_free(m);
    }
    need(found[0]&&found[1]&&found[2]&&clamp_found,"missing weights or actual activation clamp");
    auto &g=result.w[0],&u=result.w[1],&d=result.w[2];need(g.K==u.K&&g.M==u.M&&g.E==u.E&&d.K==g.M&&d.M==g.K&&d.E==g.E,"FFN shapes");
    need(g.K==5120&&g.M==2304&&g.E==384&&std::isfinite(result.clamp),"unexpected model shape");return result;
}
struct Mapping {
    TensorInfo info;void*base=nullptr;size_t length=0;uint8_t*raw=nullptr;bool registered=false;ggml_backend_buffer_t buffer=nullptr;
    Mapping(const TensorInfo&i):info(i){size_t page=sysconf(_SC_PAGESIZE),start=i.offset/page*page,skip=i.offset-start;length=skip+i.expertbytes*i.E;
        int fd=open(i.path.c_str(),O_RDONLY);need(fd>=0,"open weights");struct stat st;need(!fstat(fd,&st)&&i.offset<=size_t(st.st_size)&&i.expertbytes*i.E<=size_t(st.st_size)-i.offset,"file range");
        base=mmap(nullptr,length,PROT_READ,MAP_SHARED,fd,start);close(fd);need(base!=MAP_FAILED,"mmap");raw=static_cast<uint8_t*>(base)+skip;
        registered=ggml_backend_cuda_register_host_buffer(base,length);need(registered,"host weight registration");buffer=ggml_backend_cpu_buffer_from_ptr(raw,i.expertbytes*i.E);need(buffer,"host weight wrapper");ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ~Mapping(){if(buffer)ggml_backend_buffer_free(buffer);if(registered)ggml_backend_cuda_unregister_host_buffer(base);if(base&&base!=MAP_FAILED)munmap(base,length);}
};
struct Banks {
    ggml_context*ctx=nullptr;ggml_backend_buffer_t gpu_buffer=nullptr;std::array<ggml_tensor*,3>cw{},gw{};std::array<std::unique_ptr<Mapping>,3>maps;
    Banks(const Metadata&m,ggml_backend_t gpu,int capacity){ctx=ggml_init({1024*1024,nullptr,true});need(ctx,"bank metadata");
        for(int j=0;j<3;++j){maps[j]=std::make_unique<Mapping>(m.w[j]);auto&i=m.w[j];cw[j]=ggml_new_tensor_3d(ctx,i.q,i.K,i.M,i.E);need(ggml_backend_tensor_alloc(maps[j]->buffer,cw[j],maps[j]->raw)==GGML_STATUS_SUCCESS,"host tensor bind");
            gw[j]=ggml_new_tensor_3d(ctx,i.q,i.K,i.M,capacity);}
        gpu_buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);need(gpu_buffer,"GPU weight bank allocation");ggml_backend_buffer_set_usage(gpu_buffer,GGML_BACKEND_BUFFER_USAGE_COMPUTE);ggml_backend_buffer_clear(gpu_buffer,0);
    }
    ~Banks(){if(gpu_buffer)ggml_backend_buffer_free(gpu_buffer);if(ctx)ggml_free(ctx);}
};
struct Graph {
    ggml_context*ctx=nullptr;ggml_backend_t backend;ggml_backend_buffer_t buf=nullptr,pinbuf=nullptr;ggml_tensor*input,*ids,*output;ggml_cgraph*gf;
    float*pinned=nullptr;int N,top;bool cpu;
    Graph(ggml_backend_t b,const std::array<ggml_tensor*,3>&w,int n,int k,float clamp,bool on_cpu):backend(b),N(n),top(k),cpu(on_cpu){
        ctx=ggml_init({2*1024*1024,nullptr,true});need(ctx,"graph metadata");input=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,w[0]->ne[0],1,N);ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,top,N);
        if(cpu){ck(cudaHostAlloc(&pinned,ggml_nbytes(input),cudaHostAllocPortable));pinbuf=ggml_backend_cpu_buffer_from_ptr(pinned,ggml_nbytes(input));need(ggml_backend_tensor_alloc(pinbuf,input,pinned)==GGML_STATUS_SUCCESS,"pinned input bind");}
        auto*gate=ggml_mul_mat_id(ctx,w[0],input,ids);auto*up=ggml_mul_mat_id(ctx,w[1],input,ids);
        auto*act=clamp>1e-6f?ggml_swiglu_clamp(ctx,gate,up,clamp):ggml_swiglu_split(ctx,gate,up);output=ggml_mul_mat_id(ctx,w[2],act,ids);
        gf=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(gf,output);
        for(int i=0;i<ggml_graph_n_nodes(gf);++i)need(ggml_backend_supports_op(backend,ggml_graph_node(gf,i)),"unsupported graph node");
        buf=ggml_backend_alloc_ctx_tensors(ctx,backend);need(buf,"graph activation allocation");
    }
    void compute(){need(ggml_backend_graph_compute(backend,gf)==GGML_STATUS_SUCCESS,"graph compute");ggml_backend_synchronize(backend);}
    ~Graph(){if(buf)ggml_backend_buffer_free(buf);if(pinbuf)ggml_backend_buffer_free(pinbuf);if(pinned)cudaFreeHost(pinned);if(ctx)ggml_free(ctx);}
};
struct Worker {
    std::mutex mutex;std::condition_variable cv;bool stop=false,ready=false,done=true;std::function<void()>job;std::exception_ptr error;std::thread thread;
    Worker():thread([this]{ck(cudaSetDevice(0));for(;;){std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return ready||stop;});if(stop)return;auto fn=job;ready=false;lock.unlock();
        try{fn();}catch(...){error=std::current_exception();}lock.lock();done=true;cv.notify_all();}}){}
    void start(std::function<void()>fn){std::lock_guard<std::mutex>lock(mutex);need(done,"worker busy");error=nullptr;done=false;job=std::move(fn);ready=true;cv.notify_all();}
    void wait(){std::unique_lock<std::mutex>lock(mutex);cv.wait(lock,[&]{return done;});if(error)std::rethrow_exception(error);}
    ~Worker(){try{wait();}catch(...){} {std::lock_guard<std::mutex>lock(mutex);stop=true;}cv.notify_all();thread.join();}
};
struct Chunk {std::vector<int>experts;std::vector<int32_t>slots,ids;std::unique_ptr<Graph>graph;int32_t*dslots=nullptr;~Chunk(){if(dslots)cudaFree(dslots);}};
struct Timing {double wall=0,cpu_done=0,gpu_done=0,cpu_compute=0,cpu_input=0,cpu_return=0;};
struct Plan {
    Banks&banks;ggml_backend_t cpu,gpu;int D,T,capacity,pgpu,active;size_t weight_bytes=0,copy_calls=0;float clamp;
    std::vector<std::unique_ptr<Chunk>>chunks;std::unique_ptr<Chunk>c;
    float *dx=nullptr,*pairs=nullptr,*out=nullptr,*mix=nullptr,*ctmp=nullptr;int32_t*unused=nullptr;
    cudaStream_t gs=nullptr,cs=nullptr;float*return_host=nullptr;
    Plan(Banks&bank,ggml_backend_t cb,ggml_backend_t gb,const std::vector<int32_t>&routes,const std::vector<float>&x,const std::vector<float>&weights,int t,int cap,int p,float limit):banks(bank),cpu(cb),gpu(gb),D(5120),T(t),capacity(cap),pgpu(p),clamp(limit){
        std::vector<int>count(384),rank;for(auto e:routes)++count.at(e);for(int e=0;e<384;++e)if(count[e])rank.push_back(e);active=rank.size();need(p>=0&&p<=active,"split count");
        std::stable_sort(rank.begin(),rank.end(),[&](int a,int b){return count[a]>count[b];});std::vector<int>which(384,-1),local(384,-1);
        std::vector<int>gexp(rank.begin(),rank.begin()+p);std::sort(gexp.begin(),gexp.end());
        for(size_t i=0;i<gexp.size();i+=cap){auto q=std::make_unique<Chunk>();q->experts.assign(gexp.begin()+i,gexp.begin()+std::min(i+cap,gexp.size()));
            for(size_t j=0;j<q->experts.size();++j){which[q->experts[j]]=chunks.size();local[q->experts[j]]=j;}chunks.push_back(std::move(q));}
        c=std::make_unique<Chunk>();for(size_t i=p;i<rank.size();++i)c->experts.push_back(rank[i]);
        for(size_t slot=0;slot<routes.size();++slot){int e=routes[slot];auto&q=which[e]<0?c:chunks[which[e]];q->slots.push_back(slot);q->ids.push_back(which[e]<0?e:local[e]);}
        // Every original token-expert pair belongs to exactly one branch and one chunk.
        std::vector<int>seen(routes.size());for(auto&chunk:chunks)for(auto s:chunk->slots)++seen[s];for(auto s:c->slots)++seen[s];need(std::all_of(seen.begin(),seen.end(),[](int n){return n==1;}),"partition coverage");
        ck(cudaStreamCreateWithFlags(&gs,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&cs,cudaStreamNonBlocking));
        ck(cudaMalloc(&dx,x.size()*4));ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));ck(cudaMalloc(&pairs,size_t(D)*TOP*T*4));ck(cudaMalloc(&out,size_t(D)*T*4));ck(cudaMalloc(&mix,weights.size()*4));ck(cudaMemcpy(mix,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));
        for(auto&q:chunks){q->graph=std::make_unique<Graph>(gpu,banks.gw,q->ids.size(),1,clamp,false);ck(cudaMalloc(&q->dslots,q->slots.size()*4));}
        if(!c->slots.empty()){c->graph=std::make_unique<Graph>(cpu,banks.cw,c->ids.size(),1,clamp,true);ck(cudaMalloc(&c->dslots,c->slots.size()*4));ck(cudaMalloc(&ctmp,size_t(D)*c->slots.size()*4));ck(cudaHostAlloc(&return_host,size_t(D)*c->slots.size()*4,cudaHostAllocPortable));}
        for(auto&q:chunks)for(int j=0;j<3;++j){weight_bytes+=q->experts.size()*banks.maps[j]->info.expertbytes;for(size_t k=0;k<q->experts.size();){++copy_calls;size_t last=k+1;while(last<q->experts.size()&&q->experts[last]==q->experts[last-1]+1)++last;k=last;}}
    }
    void upload(Chunk&q){for(int j=0;j<3;++j){auto&m=*banks.maps[j];for(size_t k=0;k<q.experts.size();){size_t last=k+1;while(last<q.experts.size()&&q.experts[last]==q.experts[last-1]+1)++last;
                ggml_backend_tensor_set_async(gpu,banks.gw[j],m.raw+size_t(q.experts[k])*m.info.expertbytes,k*m.info.expertbytes,(last-k)*m.info.expertbytes);k=last;}}}
    Timing run(Worker&worker,bool concurrent){
        Timing result;auto start=Clock::now();
        auto cpath=[&]{auto tick=Clock::now();auto&q=*c;auto&g=*q.graph;
            std::memcpy(g.ids->data,q.ids.data(),q.ids.size()*4);ck(cudaMemcpyAsync(q.dslots,q.slots.data(),q.slots.size()*4,cudaMemcpyHostToDevice,cs));
            split_gather(dx,q.dslots,ctmp,D,q.slots.size(),TOP,cs);ck(cudaGetLastError());ck(cudaMemcpyAsync(g.input->data,ctmp,ggml_nbytes(g.input),cudaMemcpyDeviceToHost,cs));ck(cudaStreamSynchronize(cs));result.cpu_input=elapsed(tick);
            tick=Clock::now();g.compute();result.cpu_compute=elapsed(tick);tick=Clock::now();
            // A pinned bounce buffer avoids hidden pageable-transfer serialization; this memcpy is timed.
            std::memcpy(return_host,g.output->data,ggml_nbytes(g.output));ck(cudaMemcpyAsync(ctmp,return_host,ggml_nbytes(g.output),cudaMemcpyHostToDevice,cs));
            split_scatter(ctmp,q.dslots,pairs,D,q.slots.size(),cs);ck(cudaGetLastError());ck(cudaStreamSynchronize(cs));result.cpu_return=elapsed(tick);result.cpu_done=elapsed(start);
        };
        struct Join {Worker&w;bool active=false;~Join(){if(active)try{w.wait();}catch(...){}}} join{worker};
        if(c->graph&&concurrent){worker.start(cpath);join.active=true;}else if(c->graph)cpath();
        for(auto&q:chunks){auto&g=*q->graph;upload(*q);ggml_backend_tensor_set_async(gpu,g.ids,q->ids.data(),0,q->ids.size()*4);
            ck(cudaMemcpyAsync(q->dslots,q->slots.data(),q->slots.size()*4,cudaMemcpyHostToDevice,gs));split_gather(dx,q->dslots,static_cast<float*>(g.input->data),D,q->slots.size(),TOP,gs);ck(cudaGetLastError());ck(cudaStreamSynchronize(gs));
            g.compute();split_scatter(static_cast<float*>(g.output->data),q->dslots,pairs,D,q->slots.size(),gs);ck(cudaGetLastError());ck(cudaStreamSynchronize(gs));
        }if(!chunks.empty())result.gpu_done=elapsed(start);
        if(c->graph&&concurrent)worker.wait();
        split_merge(pairs,mix,out,D,T,TOP,gs);ck(cudaGetLastError());ck(cudaStreamSynchronize(gs));result.wall=elapsed(start);return result;
    }
    ~Plan(){c.reset();chunks.clear();if(return_host)cudaFreeHost(return_host);for(auto p:{dx,pairs,out,mix,ctmp})if(p)cudaFree(p);if(gs)cudaStreamDestroy(gs);if(cs)cudaStreamDestroy(cs);}
};
static void print_times(const char*name,const std::vector<Timing>&times,double Timing::*field){std::printf(",\"%s\":[",name);for(size_t i=0;i<times.size();++i)std::printf("%s%.9g",i?",":"",times[i].*field);std::printf("]");}
int main(int argc,char**argv)try{
    need(argc>=4,"directory layer describe | directory layer routes.bin T seed chunk repeats threads");int layer=std::stoi(argv[2]);auto meta=metadata(argv[1],layer);
    if(std::string(argv[3])=="describe"){std::printf("{\"layer\":%d,\"clamp\":%.9g,\"weights\":[",layer,meta.clamp);for(int j=0;j<3;++j){auto&w=meta.w[j];std::printf("%s{\"name\":\"%s\",\"type\":\"%s\",\"K\":%lld,\"M\":%lld,\"experts\":%lld,\"expert_bytes\":%zu}",j?",":"",w.name.c_str(),ggml_type_name(w.q),(long long)w.K,(long long)w.M,(long long)w.E,w.expertbytes);}std::puts("]}");return 0;}
    need(argc==9||argc==10,"directory layer routes.bin T seed chunk repeats threads [fixed GPU expert count]");int T=std::stoi(argv[4]),seed=std::stoi(argv[5]),capacity=std::stoi(argv[6]),repeats=std::stoi(argv[7]),threads=std::stoi(argv[8]);
    need(T>=16&&T<=1024&&capacity>=4&&capacity<=64&&repeats>=3&&repeats<=10&&(threads==12||threads==24),"probe bounds");need(!prctl(PR_SET_THP_DISABLE,1,0,0,0),"THP disable");
    {std::ifstream f("/proc/self/status");std::string s;bool yes=false;while(std::getline(f,s))if(s=="THP_enabled:\t0")yes=true;need(yes,"THP verification");}
    std::vector<int32_t>routes(T*TOP);std::ifstream file(argv[3],std::ios::binary);need(bool(file.read(reinterpret_cast<char*>(routes.data()),routes.size()*4))&&file.peek()==EOF,"route length");
    std::vector<int>hist(384);for(int t=0;t<T;++t){std::array<bool,384>seen{};for(int k=0;k<TOP;++k){int e=routes[t*TOP+k];need(e>=0&&e<384&&!seen[e],"invalid or duplicate route");seen[e]=true;++hist[e];}}
    int active=std::count_if(hist.begin(),hist.end(),[](int n){return n>0;});std::vector<int>ps={0,active/4,active/2,active*3/4,active*9/10,active*19/20,active};std::sort(ps.begin(),ps.end());ps.erase(std::unique(ps.begin(),ps.end()),ps.end());
    ck(cudaSetDevice(0));size_t free,total;ck(cudaMemGetInfo(&free,&total));size_t bankbytes=0;for(auto&w:meta.w)bankbytes+=w.expertbytes*capacity;
    size_t estimated=bankbytes+size_t(T)*TOP*(5120*4+2304*3)*4+size_t(T)*5120*8+350ull*1024*1024;
    need(free>estimated+300ull*1024*1024,"conservative VRAM headroom check");
    auto cpu=ggml_backend_cpu_init(),gpu=ggml_backend_cuda_init(0);need(cpu&&gpu,"backends");auto pp=ggml_threadpool_params_default(threads);pp.poll=0;pp.strict_cpu=true;for(int i=0;i<threads;++i)pp.cpumask[i]=true;auto pool=ggml_threadpool_new(&pp);ggml_backend_cpu_set_n_threads(cpu,threads);ggml_backend_cpu_set_threadpool(cpu,pool);
    std::mt19937 random(seed);std::uniform_real_distribution<float>dist(-.1f,.1f),wd(.1f,1.f);std::vector<float>x(size_t(T)*5120),weights(T*TOP);for(auto&v:x)v=dist(random);for(int t=0;t<T;++t){float sum=0;for(int k=0;k<TOP;++k)sum+=(weights[t*TOP+k]=wd(random));for(int k=0;k<TOP;++k)weights[t*TOP+k]/=sum;}
    {
        Banks banks(meta,gpu,capacity);Worker worker;std::vector<float>reference(size_t(T)*5120);double refnorm=0;
        {Graph original(cpu,banks.cw,T,TOP,meta.clamp,true);ggml_backend_tensor_set(original.input,x.data(),0,x.size()*4);ggml_backend_tensor_set(original.ids,routes.data(),0,routes.size()*4);original.compute();auto*y=static_cast<const float*>(original.output->data);
            for(int t=0;t<T;++t)for(int d=0;d<5120;++d){double sum=0;for(int k=0;k<TOP;++k)sum+=double(y[(size_t(t)*TOP+k)*5120+d])*weights[t*TOP+k];reference[size_t(t)*5120+d]=sum;refnorm+=sum*sum;}}
        need(refnorm>0&&std::isfinite(refnorm),"reference finite/norm");
        std::printf("{\"schema\":1,\"layer\":%d,\"tokens\":%d,\"experts\":384,\"active_experts\":%d,\"topk\":6,\"chunk_capacity\":%d,\"cpu_threads\":%d,\"clamp\":%.9g,\"weights_types\":[\"%s\",\"%s\",\"%s\"],\"estimated_GPU_bytes\":%zu,\"initial_free_GPU_bytes\":%zu,\"histogram\":[",layer,T,active,capacity,threads,meta.clamp,ggml_type_name(meta.w[0].q),ggml_type_name(meta.w[1].q),ggml_type_name(meta.w[2].q),estimated,free);
        for(int e=0;e<384;++e)std::printf("%s%d",e?",":"",hist[e]);std::printf("],\"measurements\":[");bool first=true;
        // Each p is bracketed by fresh all-GPU measurements; alternate candidate order by seed parity.
        std::vector<int>schedule;
        if(argc==10){int p=std::stoi(argv[9]);need(p>=0&&p<=active,"fixed split count");schedule={active,0,active};if(p>0&&p<active){schedule.push_back(p);schedule.push_back(active);}}
        else for(auto p:ps){if(p==active)continue;schedule.push_back(active);schedule.push_back(p);schedule.push_back(active);}
        if(seed%2)std::reverse(schedule.begin(),schedule.end());
        for(int p:schedule){auto prep=Clock::now();auto plan=std::make_unique<Plan>(banks,cpu,gpu,routes,x,weights,T,capacity,p,meta.clamp);double prepare=elapsed(prep);
            plan->run(worker,true);std::vector<float>got(reference.size());ck(cudaMemcpy(got.data(),plan->out,got.size()*4,cudaMemcpyDeviceToHost));double error=0,maxabs=0;
            for(size_t i=0;i<got.size();++i){need(std::isfinite(got[i]),"nonfinite output");double delta=double(got[i])-reference[i];error+=delta*delta;maxabs=std::max(maxabs,std::abs(delta));}
            need(error/refnorm<2e-3,"FFN numeric validation failed");if(p==0)need(error/refnorm<1e-8,"CPU packing/reference mismatch");
            std::vector<Timing>times;for(int i=0;i<repeats;++i)times.push_back(plan->run(worker,true));
            std::vector<Timing>serial;double serial_nmse=0;
            if(p>0&&p<active){plan->run(worker,false);for(int i=0;i<3;++i)serial.push_back(plan->run(worker,false));
                std::vector<float>serial_out(got.size());ck(cudaMemcpy(serial_out.data(),plan->out,serial_out.size()*4,cudaMemcpyDeviceToHost));
                for(size_t i=0;i<got.size();++i){need(std::isfinite(serial_out[i]),"nonfinite serial output");serial_nmse+=std::pow(double(got[i])-serial_out[i],2);}serial_nmse/=refnorm;need(serial_nmse<1e-10,"parallel/serial consistency");}
            std::printf("%s{\"gpu_experts\":%d,\"cpu_experts\":%d,\"cpu_pairs\":%zu,\"gpu_chunks\":%zu,\"weight_H2D_bytes\":%zu,\"weight_H2D_calls\":%zu,\"prepare_ms\":%.9g,\"nmse_vs_native_CPU\":%.12g,\"max_abs\":%.9g",first?"":",",p,active-p,plan->c->slots.size(),plan->chunks.size(),plan->weight_bytes,plan->copy_calls,prepare,error/refnorm,maxabs);first=false;
            print_times("wall_ms",times,&Timing::wall);print_times("cpu_done_ms",times,&Timing::cpu_done);print_times("gpu_done_ms",times,&Timing::gpu_done);print_times("cpu_compute_ms",times,&Timing::cpu_compute);print_times("cpu_input_ms",times,&Timing::cpu_input);print_times("cpu_return_ms",times,&Timing::cpu_return);print_times("serial_wall_ms",serial,&Timing::wall);
            print_times("serial_cpu_compute_ms",serial,&Timing::cpu_compute);print_times("serial_cpu_done_ms",serial,&Timing::cpu_done);print_times("serial_gpu_done_ms",serial,&Timing::gpu_done);std::printf(",\"serial_parallel_nmse\":%.12g}",serial_nmse);std::fflush(stdout);
        }std::puts("],\"pass\":true}");
    }
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
