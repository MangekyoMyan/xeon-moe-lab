from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent
def run(script,*args):subprocess.run(['python3',str(R/script),*args],check=True)
def main():
    for profile,labels in [('gpu-sort-candidate',['native-control-r1']),('native-pool',['native-candidate-r1','native-candidate-r2']),('gpu-sort-candidate',['native-control-r2'])]:
        run('switch_and_wait.py',profile)
        if labels[0]=='native-candidate-r1':
            run('bench_cpu_wait.py','--label','native-affinity-smoke','--context','128','--tokens','16')
        for label in labels:
            run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
            run('heldout_probe.py','--label',label)
    run('analyze_heldout.py','--control','native-control-r1','native-control-r2','--candidate','native-candidate-r1','native-candidate-r2','--out','native-pool/LIVE_ABBA.json','--scope','Graph executor OpenMP vs ggml pthread/poll0,ABBA8inputsx2;cache initialization retains OpenMP;native master-last affinity;interleave;1024batch;no row-local;equal warmup each run')
if __name__=='__main__':main()
