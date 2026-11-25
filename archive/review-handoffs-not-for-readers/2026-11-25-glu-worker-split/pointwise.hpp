// Exercise the CPU DSO's actual GLU entry point. Every worker starts with
// pristine buffers so partial-alias fallback can be checked without a race.
struct ShapeCase {
    const char *name;
    int width, rows, threads, input_pad, up_pad, output_pad;
    float limit;
    bool joined;
    int alias; // 0 independent, 1 exactly gate, 2 exactly up, 3 one float after gate
};

static float formula(float raw, float up, float limit) {
    const float g = std::min(raw, limit);
    return g / (1.f + expf(-g)) * std::clamp(up, -limit, limit);
}

static json pointwise_cases(bool candidate) {
    const std::vector<ShapeCase> cases = {
        {"contiguous",2304,6,24,0,0,0,10,false,0},
        {"offsets-padded",2304,6,24,9,21,35,10,false,0},
        {"alternate-limit",2304,6,24,3,7,11,2.5f,false,0},
        {"inplace-gate",2304,6,24,9,15,9,10,false,1},
        {"inplace-up",2304,6,24,3,7,7,2.5f,false,2},
        {"partial-gate-single-worker-check",2304,6,24,11,13,11,10,false,3},
        {"other-width",2303,6,24,1,7,5,10,false,0},
        {"other-rows",2304,5,24,0,5,1,10,false,0},
        {"other-team",2304,6,12,7,7,5,10,false,0},
        {"concatenated-input",2304,6,24,3,0,5,10,true,0}
    };
    json out = json::array();
    for (const auto &c : cases) {
        auto ctx=ggml_init({128*1024,nullptr,true});
        need(ctx!=nullptr, "synthetic ggml context");
        const int w=c.width, n=c.rows, gate_pitch=(c.joined?2*w:w)+c.input_pad;
        const int up_pitch=w+c.up_pad, out_pitch=w+c.output_pad;
        auto gate=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,c.joined?2*w:w,n);
        auto up=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w,n);
        auto dst=ggml_swiglu_clamp(ctx,gate,c.joined?nullptr:up,c.limit);
        const float mark=-87654.25f;
        std::vector<float> a(n*gate_pitch+40,mark), b(n*up_pitch+40,mark), y(n*out_pitch+40,mark);
        constexpr int ag=3, bg=7, yg=11;
        const std::array<float,13> edge={-150.0f,-20.0f,-c.limit,std::nextafter(-c.limit,0.0f),
            std::nextafter(-c.limit,-INFINITY),-0.0f,0.0f,std::nextafter(c.limit,0.0f),c.limit,
            std::nextafter(c.limit,INFINITY),20.0f,100.0f,0.125f};
        for(int j=0;j<n;++j) for(int k=0;k<w;++k){
            const unsigned h=unsigned(j*w+k)*2654435761u;
            const float gv=k%3?edge[(k+2*j)%edge.size()]:(int(h%2041)-1000)*0.015625f;
            const float uv=k%5?edge[(5*k+3*j+1)%edge.size()]:(int((h>>11)%2011)-1000)*0.03125f;
            a[ag+j*gate_pitch+k]=gv;
            if(c.joined) a[ag+j*gate_pitch+w+k]=uv; else b[bg+j*up_pitch+k]=uv;
        }
        const auto orig_a=a,orig_b=b,orig_y=y;
        gate->nb[1]=gate_pitch*4;up->nb[1]=up_pitch*4;dst->nb[1]=out_pitch*4;
        const bool distribute=candidate && !c.joined && w==2304 && n==6 && c.threads==24 && c.alias!=3;
        std::vector<unsigned> coverage(w*n,0), assigned(c.threads,0);
        for(int ith=0;ith<c.threads;++ith){
            a=orig_a;b=orig_b;y=orig_y;
            auto ea=orig_a,eb=orig_b,ey=orig_y;
            auto select_out=[&](auto &ga,auto &ub,auto &outbuf){
                return c.alias==1?ga.data()+ag:c.alias==2?ub.data()+bg:
                       c.alias==3?ga.data()+ag+1:outbuf.data()+yg;
            };
            gate->data=a.data()+ag;up->data=b.data()+bg;dst->data=select_out(a,b,y);
            float *expected=select_out(ea,eb,ey);
            const int dr=(n+c.threads-1)/c.threads;
            for(int j=0;j<n;++j) for(int k=0;k<w;++k){
                const int owner=distribute?k/96:j/dr;
                if(owner!=ith)continue;
                float g=ea[ag+j*gate_pitch+k];
                float u=c.joined?ea[ag+j*gate_pitch+w+k]:eb[bg+j*up_pitch+k];
                expected[j*out_pitch+k]=formula(g,u,c.limit);
                ++coverage[j*w+k];++assigned[ith];
            }
            ggml_compute_params params{};params.ith=ith;params.nth=c.threads;
            ggml_compute_forward_glu(&params,dst);
            auto equal=[](const auto &l,const auto &r){return std::memcmp(l.data(),r.data(),l.size()*4)==0;};
            need(equal(ea,a)&&equal(eb,b)&&equal(ey,y), std::string("synthetic values/partition/guards: ")+c.name);
        }
        need(std::all_of(coverage.begin(),coverage.end(),[](unsigned x){return x==1;}),"single owner");
        out.push_back({{"name",c.name},{"ownership","full backing arrays compared after each real DSO worker call"},
                       {"partition",distribute?"channels":"rows"},{"elements",w*n},{"worker_elements",assigned},
                       {"individual_worker_checks",c.threads},{"arrays_and_guards_bit_exact",true}});
        ggml_free(ctx);
    }
    need(formula(20,1,10)<10.f,"clamp order");
    return out;
}
