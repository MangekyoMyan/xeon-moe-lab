"""Derive a fixed-mode cross-library harness; preserve earlier ABBA probe."""
from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'probe.cpp').read_text()
def replace(a,b):
    global s
    assert s.count(a)==1,a[:100]
    s=s.replace(a,b)
replace('const char*directory){\n    need(set_mode&&diag&&counts,"experimental API required");',
        'const char*directory,int fixed){\n    need(fixed>=0&&fixed<=2&&(fixed==0||(set_mode&&diag&&counts)),"fixed mode unavailable");')
replace('for(int a=0;a<3;++a){mode(a);diag(1);uint64_t before[3],after[3];counts(before);compute();counts(after);',
        'for(int a=0;a<(set_mode?3:1);++a){mode(a);if(diag)diag(1);uint64_t before[3]{},after[3]{};if(counts)counts(before);compute();if(counts)counts(after);')
replace('need(after[1]-before[1]==uint64_t(a&&hits?6*M:0),"warm dispatch");}diag(0);',
        'if(counts)need(after[1]-before[1]==uint64_t(a&&hits?6*M:0),"warm dispatch");}if(diag)diag(0);')
replace('mode(0);diag(0);compute();','mode(0);if(diag)diag(0);compute();')
start=s.index('        std::printf("{\\\"kind\\\":\\\"operator')
end=s.index('        xeon_iq3_cache_free_v1(owner);',start)
s=s[:start]+r'''
        const std::string output_file=std::string(directory)+"/outputs-hit"+std::to_string(hits)+".bin";
        FILE*outputs=std::fopen(output_file.c_str(),"wb");need(outputs,"output file");
        std::printf("{\"kind\":\"fixed\",\"tensor\":\"%s\",\"type\":\"%s\",\"mode\":%d,\"experimental\":%s,\"threads\":12,\"ggml_numa\":%s,\"cached_of_six\":%d,\"cache_bytes\":%zu,\"samples\":[",tensor,ggml_type_name(type),fixed,set_mode?"true":"false",ggml_is_numa()?"true":"false",hits,xeon_iq3_cache_used_v1(owner));
        for(int block=0;block<32;++block){
            int r=block%8;int32_t route[6];for(int j=0;j<6;++j)route[j]=permutation[r*6+j];
            ggml_backend_tensor_set(ids,route,0,sizeof(route));ggml_backend_tensor_set(x,input[r].data(),0,input[r].size()*4);
            mode(0);compute();ggml_backend_tensor_get(y,expected.data(),0,ggml_nbytes(y));
            need(std::all_of(expected.begin(),expected.end(),[](float v){return std::isfinite(v);}),"nonfinite real output");
            write(outputs,expected.data(),expected.size()*4);
            mode(fixed);compute();const double begin=now();for(int repeat=0;repeat<32;++repeat)compute();double elapsed=(now()-begin)/32;
            ggml_backend_tensor_get(y,out.data(),0,ggml_nbytes(y));need(!std::memcmp(out.data(),expected.data(),out.size()*4),"fixed output mismatch");need(std::isfinite(elapsed)&&elapsed>0,"time");
            std::printf("%s{\"block\":%d,\"route_group\":%d,\"ms\":%.9f}",block?",":"",block,r,elapsed);
        }
        std::fclose(outputs);
        std::printf("],\"routes\":[");for(int i=0;i<48;++i)std::printf("%s%u",i?",":"",permutation[i]);std::printf("],\"pins\":[");for(int i=0;i<12;++i)std::printf("%s%d",i?",":"",pins[i]);std::puts("],\"bit_mismatches\":0}");std::fflush(stdout);
''' + s[end:]
replace('}else{need(std::string(argv[1])=="bench"&&argc==6,"bench args");benchmark(cpu,argv[2],argv[3],argv[4]);std::puts("{\\\"passed\\\":true}");}',
        '}else{need(std::string(argv[1])=="fixed"&&argc==7,"fixed args");benchmark(cpu,argv[2],argv[3],argv[4],std::stoi(argv[5]));std::puts("{\\\"passed\\\":true}");}')
(R/'fixed_probe.cpp').write_text(s)
