"""Serial guarded hot-expert fixture runs; refuses a busy/shared benchmark slot."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,time
R=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=json.loads((R/'profiles/final.json').read_text());model=Path(config['model']).parent
    env=config['environment'];cpu=next(x['file'] for x in config['study_libraries'] if Path(x['file']).name.startswith('libggml-cpu.'))
    gpu=next(x['file'] for x in config['study_libraries'] if Path(x['file']).name.startswith('libggml-cuda.'))
    ids=[19] if a.pilot else [37,83]
    records=[]
    for seed in ids:
        for layer in [4,24]:
            name=f'hot-ffn-L{layer}-s{seed}'
            command=['env','LD_LIBRARY_PATH='+env['LD_LIBRARY_PATH'],'CUDA_VISIBLE_DEVICES='+env['CUDA_VISIBLE_DEVICES'],
                'numactl','--physcpubind=0-23','--interleave=0,1',str(R/'hot-ffn/probe'),str(model),str(layer),str(seed),'60',cpu,gpu]
            subprocess.run(['python3',str(R/'guard.py'),'--name',name,'--timeout','180','--',*command],check=True)
            result_path=R/'execution'/name/'stdout.log';result=json.loads(result_path.read_text())
            assert result['pass'] and all(x['numeric_diagnostic_gate_passed'] for x in result['measurements'])
            records.append({'layer':layer,'seed':seed,'result':str(result_path)})
            print(json.dumps({'layer':layer,'seed':seed,'pass':True}),flush=True)
    (R/'hot-ffn'/('PILOT.json' if a.pilot else 'VALIDATION.json')).write_text(json.dumps({'completed_unix':time.time(),'runs':records,'scope':'Actual weights;synthetic input and selected6route IDs;conditional H,not wholemodel speed'},indent=2)+'\n')
if __name__=='__main__':main()
