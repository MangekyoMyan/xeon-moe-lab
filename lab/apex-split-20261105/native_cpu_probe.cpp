// Reuse the same checked GGUF mapping and FFN construction; this endpoint retains top6,T input layout.
#define main unused_split_probe_main
#include "split_probe.cpp"
#undef main
int main(int argc,char**argv)try{
    need(argc==8,"directory layer routes T seed repeats threads");auto meta=metadata(argv[1],std::stoi(argv[2]));int T=std::stoi(argv[4]),seed=std::stoi(argv[5]),reps=std::stoi(argv[6]),threads=std::stoi(argv[7]);
    need(T>=16&&T<=1024&&reps>=3&&reps<=10&&(threads==12||threads==24),"bounds");need(!prctl(PR_SET_THP_DISABLE,1,0,0,0),"THP disable");
    std::vector<int32_t>routes(T*TOP);std::ifstream file(argv[3],std::ios::binary);need(bool(file.read(reinterpret_cast<char*>(routes.data()),routes.size()*4))&&file.peek()==EOF,"routes");
    for(int t=0;t<T;++t){std::array<bool,384>seen{};for(int k=0;k<TOP;++k){int e=routes[t*TOP+k];need(e>=0&&e<384&&!seen[e],"route validity");seen[e]=true;}}
    ck(cudaSetDevice(0));auto cpu=ggml_backend_cpu_init(),gpu=ggml_backend_cuda_init(0);need(cpu&&gpu,"backends");auto pp=ggml_threadpool_params_default(threads);pp.poll=0;pp.strict_cpu=true;for(int i=0;i<threads;++i)pp.cpumask[i]=true;auto pool=ggml_threadpool_new(&pp);ggml_backend_cpu_set_n_threads(cpu,threads);ggml_backend_cpu_set_threadpool(cpu,pool);
    std::mt19937 random(seed);std::uniform_real_distribution<float>dist(-.1f,.1f),wd(.1f,1.f);std::vector<float>x(size_t(T)*5120),weights(T*TOP);for(auto&v:x)v=dist(random);for(int t=0;t<T;++t){float sum=0;for(int k=0;k<TOP;++k)sum+=(weights[t*TOP+k]=wd(random));for(int k=0;k<TOP;++k)weights[t*TOP+k]/=sum;}
    {
        Banks banks(meta,gpu,1);Graph graph(cpu,banks.cw,T,TOP,meta.clamp,true);ggml_backend_tensor_set(graph.ids,routes.data(),0,routes.size()*4);
        float *dx,*dpairs,*dweights,*dy,*bounce;cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaMalloc(&dx,x.size()*4));ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));ck(cudaMalloc(&dpairs,ggml_nbytes(graph.output)));ck(cudaMalloc(&dweights,weights.size()*4));ck(cudaMemcpy(dweights,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));ck(cudaMalloc(&dy,x.size()*4));ck(cudaHostAlloc(&bounce,ggml_nbytes(graph.output),cudaHostAllocPortable));
        auto run=[&]{Timing result;auto start=Clock::now(),tick=start;ck(cudaMemcpyAsync(graph.input->data,dx,x.size()*4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));result.cpu_input=elapsed(tick);
            tick=Clock::now();graph.compute();result.cpu_compute=elapsed(tick);tick=Clock::now();std::memcpy(bounce,graph.output->data,ggml_nbytes(graph.output));ck(cudaMemcpyAsync(dpairs,bounce,ggml_nbytes(graph.output),cudaMemcpyHostToDevice,stream));split_merge(dpairs,dweights,dy,5120,T,TOP,stream);ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));result.cpu_return=elapsed(tick);result.wall=elapsed(start);return result;};
        run();std::vector<Timing>times;for(int i=0;i<reps;++i)times.push_back(run());std::vector<float>got(x.size());ck(cudaMemcpy(got.data(),dy,got.size()*4,cudaMemcpyDeviceToHost));
        double norm=0,err=0;auto*pairs=static_cast<const float*>(graph.output->data);
        for(int t=0;t<T;++t)for(int d=0;d<5120;++d){double ref=0;for(int k=0;k<TOP;++k)ref+=double(pairs[(size_t(t)*TOP+k)*5120+d])*weights[t*TOP+k];double value=got[size_t(t)*5120+d];need(std::isfinite(value),"finite output");norm+=ref*ref;err+=std::pow(ref-value,2);}
        need(norm>0&&err/norm<1e-10,"merge correctness");
        std::printf("{\"schema\":1,\"pass\":true,\"layer\":%d,\"tokens\":%d,\"threads\":%d,\"native_input_shape\":[5120,1,%d],\"ids_shape\":[6,%d],\"merge_nmse\":%.12g",std::stoi(argv[2]),T,threads,T,T,err/norm);
        print_times("wall_ms",times,&Timing::wall);print_times("cpu_compute_ms",times,&Timing::cpu_compute);print_times("input_ms",times,&Timing::cpu_input);print_times("return_merge_ms",times,&Timing::cpu_return);std::puts("}");
        cudaFreeHost(bounce);for(auto p:{dx,dpairs,dweights,dy})cudaFree(p);cudaStreamDestroy(stream);
    }
    ggml_backend_cpu_set_threadpool(cpu,nullptr);ggml_threadpool_free(pool);ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
