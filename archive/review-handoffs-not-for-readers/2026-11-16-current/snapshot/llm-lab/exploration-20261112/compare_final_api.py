"""Direct baseline-to-selected-profile ABBA on a previously unused input set."""
from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parent
def run(script,*args):subprocess.run(['python3',str(R/script),*args],check=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    assert args.candidate.replace('-','').isalnum() and args.candidate!='baseline'
    assert hashlib.sha256((R/'profiles/baseline.json').read_bytes()).hexdigest()=='e808d438d4ad4d852febb405dbf3d083cdf57f10e56e023ee6186eae43d340bd'
    selection=R/'FINAL_SELECTION.json'
    if selection.exists():raise RuntimeError('A final selection already exists; inspect before any repeat')
    selection.write_text(json.dumps({'candidate_profile':args.candidate,'candidate_config_sha256':hashlib.sha256((R/'profiles'/(args.candidate+'.json')).read_bytes()).hexdigest(),'baseline_config_sha256':'e808d438d4ad4d852febb405dbf3d083cdf57f10e56e023ee6186eae43d340bd','fresh_prompt_sha256':hashlib.sha256((R/'FRESH_PROMPTS.json').read_bytes()).hexdigest()},indent=2)+'\n')
    for profile,labels in [('baseline',['final-control-r1']),(args.candidate,['final-candidate-r1','final-candidate-r2']),('baseline',['final-control-r2'])]:
        run('switch_and_wait.py',profile)
        if labels[0]=='final-control-r1':run('fresh_probe.py','--label','final-inputs','--freeze-only')
        for label in labels:
            run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
            run('fresh_probe.py','--label',label)
    run('analyze_heldout.py','--control','final-control-r1','final-control-r2','--candidate','final-candidate-r1','final-candidate-r2','--out','FINAL_ABBA.json','--scope','Direct study-start baseline vs selected stack; eight fresh frozen inputs x two;128generated tokens,complete native output IDs;ABBA with equal warmup;not a quality benchmark','--final')
if __name__=='__main__':main()
