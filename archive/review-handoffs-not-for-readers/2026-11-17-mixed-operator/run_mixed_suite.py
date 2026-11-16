"""Parent-reviewed fixed eight-process sequence using the unchanged outer guard.

Stops on the first failure and never overwrites or retries a process. No service,
API, CPU-library, model, GPU or fan setting is changed.
"""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent
S=R/'incoming/xeon_moe_abs_next_2026-11-16'
def main():
    spec=json.loads((S/'NEXT_RUN_SPEC.json').read_text())
    assert spec['schedule']==[{'index':i,'library':label,'case_order':order} for i,(label,order) in enumerate(zip('FEEFEFFE',[['C0','C6','Cplan'],['C0','C6','Cplan'],['C6','Cplan','C0'],['C6','Cplan','C0'],['Cplan','C0','C6'],['Cplan','C0','C6'],['C0','C6','Cplan'],['C0','C6','Cplan']]))]
    roots={'F':LAB/'exploration-20261112/mx-prepack/release-candidate-a','E':LAB/'optimization-20261116/abs-operator/runtime'}
    frozen=json.loads((R/'INITIAL_RUNTIME.json').read_text())['files']
    for d in roots.values():
        p=d/'libggml-cpu.so.0.28.0';assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen[str(p.relative_to(LAB))]
    for row in spec['schedule']:
        index=row['index'];name=f'iq2-mixed-20261117-p{index}'
        command=['python3',str(LAB/'optimization-20261116/abs-operator/guard.py'),'--name',name,'--timeout','200','--','python3','-B',str(S/'next_experiment/run_one_mixed.py'),'--harness',str(R/'build/mixed_probe'),'--cpu-library',str(roots[row['library']]/'libggml-cpu.so.0.28.0'),'--base-library',str(LAB/'gpu-stream-20261103/runtime/libggml-base.so.0.28.0'),'--library-dir',str(LAB/'deepseek41-bringup-20261101/build/bin'),'--shard',str(LAB/'models/orcarouter--DeepSeek-V4.1-Flash-1017-abliterated-GGUF/Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00003-of-00007.gguf'),'--case-dir',str(S/'next_experiment/case'),'--run-root',str(R/'mixed-run'),'--process-index',str(index)]
        # guard owns timeout/process-group termination; do not add a second
        # independent timeout capable of orphaning its child group.
        subprocess.run(command,check=True)
    print('Eight guarded processes completed; no automatic reruns',flush=True)
if __name__=='__main__':main()
