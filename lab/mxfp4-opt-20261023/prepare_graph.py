#!/usr/bin/env python3
"""Reuse the real expert graph probe and add cross-build tensor dumps."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
BASE = LAB / 'review-glm-q8-columns4-20261020'
CANDIDATE = LAB / 'review-mxfp4-20261023'


def main():
    original = LAB / 'server-study-20261023/expert_probe.cpp'
    code = original.read_text()
    code = code.replace('if(argc<6||argc>7)return 2;', 'if(argc<6||argc>8)return 2;')
    code = code.replace('scope=argc==7?argv[6]:"both"', 'scope=argc>=7?argv[6]:"both"')
    anchor = '    check_affinity();\n    double warm=now();'
    replacement = '''    check_affinity();
    if(argc==8){
        FILE *out=std::fopen(argv[7],"wb");assert(out);
        for(int step=0;step<16;++step){
            compute(step);
            for(auto *tensor : {gate,up,act,down,result}){
                const size_t bytes=ggml_nbytes(tensor);
                assert(std::fwrite(tensor->data,1,bytes,out)==bytes);
            }
        }
        assert(!std::fclose(out));
    }
    double warm=now();'''
    assert code.count(anchor) == 1
    code = code.replace(anchor, replacement)
    code = code.replace("    assert(std::getchar()=='G');", '')
    code = code.replace("    assert(std::getchar()=='Q');", '    ')
    path = ROOT / 'graph-probe.cpp'
    path.write_text(code)
    logits = (LAB / 'deepseek-logits-probe.cpp').read_text()
    logits = logits.replace('LLAMA_LOAD_MODE_MMAP', 'LLAMA_LOAD_MODE_NONE')
    (ROOT / 'logits-probe.cpp').write_text(logits)
    record = {'original_graph_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'commands': []}
    for name, base in [('baseline', BASE), ('candidate', CANDIDATE)]:
        for kind in ('graph', 'logits'):
            command = ['c++', '-O3', '-march=ivybridge', '-std=c++17', '-fopenmp', '-I' + str(base / 'source/include'), '-I' + str(base / 'source/ggml/include'), '-I' + str(base / 'source/ggml/src'), str(ROOT / (kind + '-probe.cpp')), '-L' + str(base / 'build/bin'), '-Wl,-rpath,' + str(base / 'build/bin'), '-lllama', '-lggml-cpu', '-lggml-base', '-o', str(ROOT / (kind + '-' + name))]
            subprocess.run(command, check=True)
            record['commands'].append(command)
    (ROOT / 'graph-build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
