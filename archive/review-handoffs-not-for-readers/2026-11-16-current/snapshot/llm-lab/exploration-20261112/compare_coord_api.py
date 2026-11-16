"""Explicit ABBA of the verified MX coordination binary; no automatic adoption."""
from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent
def run(script,*arguments):subprocess.run(['python3',str(R/script),*arguments],check=True)
def main():
    for profile,labels in [('gpu-sort-candidate',['coord-mx-control-r1']),('coord-mx-combined',['coord-mx-candidate-r1','coord-mx-candidate-r2']),('gpu-sort-candidate',['coord-mx-control-r2'])]:
        run('switch_and_wait.py',profile)
        run('bench_api.py','--label',labels[0]+'-warm','--context','1024','--tokens','8')
        for label in labels:run('heldout_probe.py','--label',label)
    run('analyze_heldout.py','--control','coord-mx-control-r1','coord-mx-control-r2','--candidate','coord-mx-candidate-r1','coord-mx-candidate-r2','--out','cpu-coordination-mx/LIVE_ABBA.json','--scope','A-B-B-A, eight fixed benchmark inputs x two; standard1024,sortedCUDA,MX RHS,frequency CPU cache; only CPU coordination changes')
if __name__=='__main__':main()
