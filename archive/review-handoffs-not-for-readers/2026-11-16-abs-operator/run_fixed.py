"""Sequential production-vs-scaffold-vs-shared operator comparison. No API restart."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    common=[LAB/'gpu-stream-20261103/runtime',LAB/'deepseek41-bringup-20261101/build/bin']
    frozen=LAB/'exploration-20261112/mx-prepack/release-candidate-a'
    assert sha(frozen/'libggml-cpu.so.0.28.0')=='33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3'
    assert sha(R/'runtime/libggml-cpu.so.0.28.0')=='120038c7e4c452714d188c9ab5e7dc99bda7deaa9506e8fe92e693cca5db06d3'
    model=LAB/'models/orcarouter--DeepSeek-V4.1-Flash-1017-abliterated-GGUF/Mix-256G'
    prefix='DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-'
    records=[]
    for tag,shard,tensor in [('iq2','00003-of-00007.gguf','blk.0.ffn_up_exps.weight'),('iq3s','00004-of-00007.gguf','blk.13.ffn_down_exps.weight')]:
        for index,label in enumerate(['frozen','off','shared','shared','off','frozen']):
            d=frozen if label=='frozen' else R/'runtime';mode=1 if label=='shared' else 0
            name=f'fixed-{tag}-{index}-{label}'
            output=R/'fixed-outputs'/name
            cmd=['python3',str(R/'guard.py'),'--name',name,'--timeout','180','--','env','LD_LIBRARY_PATH='+':'.join(map(str,[d,*common])),'numactl','--physcpubind=0-23','--interleave=0,1',str(R/'fixed_probe'),'fixed',str(model/(prefix+shard)),tensor,str(output),str(mode),str(d/'libggml-cpu.so.0.28.0')]
            subprocess.run(cmd,check=True,timeout=200)
            record={'tag':tag,'position':index,'label':label,'run':name,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir())}}
            records.append(record)
            (R/'FIXED_RECEIPT.json').write_text(json.dumps({'runs':records,'complete':len(records)==12},indent=2)+'\n')
    print('Completed twelve serial cross-library runs',flush=True)
if __name__=='__main__':main()
