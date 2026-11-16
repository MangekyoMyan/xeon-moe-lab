"""Code-identical OpenMP-all versus OpenMP decode/native batch pool."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
R=Path(__file__).resolve().parent;H=R/'hybrid-pool'
def run(name,*args):subprocess.run(['python3',str(R/name),*args],check=True)
def main():
    validation=json.loads((H/'VALIDATION.json').read_text());assert validation['passed']
    library=H/'normal/runtime/libggml-cpu.so.0.28.0';sha=hashlib.sha256(library.read_bytes()).hexdigest();assert sha==validation['library_sha256']
    release=H/'release-a';release.mkdir(exist_ok=False);shutil.copy2(library,release/library.name)
    for name in ['libggml-cpu.so','libggml-cpu.so.0']:(release/name).symlink_to(library.name)
    for mode,name in [(0,'hybrid-control'),(1,'hybrid-batch')]:
        c=json.loads((R/'profiles/final.json').read_text());c['command'][c['command'].index('--poll-batch')+1]='50'
        c['environment']['LD_LIBRARY_PATH']=str(release)+':'+c['environment']['LD_LIBRARY_PATH'];c['environment']['GGML_CPU_HYBRID_POOL']=str(mode)
        c['study_libraries']=[x for x in c['study_libraries'] if Path(x['file']).name!=library.name]+[{'file':str(release/library.name),'sha256':sha}]
        c['optimization_study'].update(profile=name,status='experimental-mixed-pool',CPU_graph_runtime='OpenMP decode12;nativebatch24/poll50' if mode else 'OpenMP both;same mixed-capablebinary')
        c['tested_on_real_model']=False;c['cpu_iq3_compact_profile']['runtime']=str(release)
        target=R/'profiles'/f'{name}.json';assert not target.exists();target.write_text(json.dumps(c,indent=2)+'\n');target.chmod(0o600)
    state={'started_unix':time.time(),'complete':False,'error':None,'labels':[]};path=H/'CONTROLLER.json';assert not path.exists()
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for profile,labels in [('hybrid-control',['hybrid-control-r1']),('hybrid-batch',['hybrid-candidate-r1','hybrid-candidate-r2']),('hybrid-control',['hybrid-control-r2'])]:
            run('switch_and_wait.py',profile)
            for label in labels:
                run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
                for context in ['1024','4096']:run('bench_api.py','--label',label+'-prefill','--context',context,'--tokens','16')
                run('fresh_probe.py','--label',label);state['labels'].append(label);save()
        run('analyze_heldout.py','--control','hybrid-control-r1','hybrid-control-r2','--candidate','hybrid-candidate-r1','hybrid-candidate-r2','--out','hybrid-pool/LIVE_ABBA.json','--scope','Samebinary allOpenMP vsOpenMP12decode/native24batch-poll50,ABBA,existing8inputs,128completegeneratedIDs,no arithmetic changes','--final')
        run('switch_and_wait.py','final');state['complete']=True
    except BaseException as e:state['error']=repr(e);raise
    finally:state['finished_unix']=time.time();save()
if __name__=='__main__':main()
