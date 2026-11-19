from pathlib import Path
import csv,hashlib,json,os,subprocess
R=Path(__file__).resolve().parent;Q=R/'q8-avx1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    base=R.parent/'gpu-stream-20261103/runtime/libggml-base.so.0.28.0';out=Q/'numeric-fixed-wrapper';out.mkdir();runs=[]
    build=json.loads((Q/'BUILD.json').read_text());probe=json.loads((Q/'PROBE_BUILD.json').read_text())
    assert sha(Q/'q8_probe')==probe['binary_sha256'] and sha(base)=='2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558'
    for mode in ['control','candidate']:
        cpu=Q/mode/'runtime/libggml-cpu.so.0.28.0';assert sha(cpu)==build['variants'][mode]['CPU_sha256']
        env=os.environ.copy();env.pop('LD_PRELOAD',None);env['LD_LIBRARY_PATH']=str(cpu.parent)+':'+str(base.parent)
        cmd=[str(Q/'q8_probe'),str(cpu),str(base),str(out/(mode+'-providers.json')),'check']
        with (out/(mode+'.csv')).open('w') as o,(out/(mode+'.stderr.log')).open('w') as e:p=subprocess.run(cmd,env=env,stdout=o,stderr=e,timeout=60)
        assert p.returncode==0,(mode,p.returncode)
        rows=list(csv.DictReader((out/(mode+'.csv')).open()));assert len(rows)==4 and all(int(x['unequal_bytes'])==0 for x in rows)
        runs.append(dict(mode=mode,command=cmd,CPU_sha256=sha(cpu),base_sha256=sha(base),rows=rows))
        dis=subprocess.check_output(['objdump','-d','--disassemble=quantize_row_q8_K',str(cpu)],text=True);(Q/(mode+'-Q8-disassembly.txt')).write_text(dis)
        if mode=='candidate':assert 'vcvtps2dq' in dis
        else:assert '<quantize_row_q8_K_ref@plt>' in dis
    result=dict(passed=True,runs=runs,scope='Original check/bench helper bodies unchanged; own provider main. Fourrounding modes, fallbackincluded. Correctness only, not performance.')
    (Q/'NUMERIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
