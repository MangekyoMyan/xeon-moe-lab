"""Verify the dedicated dispatcher with no GPU expert hits before changing placement."""
from pathlib import Path
import hashlib,json,re,subprocess,time
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn'
def run(name,*args):subprocess.run(['python3',str(R/name),*args],check=True)
def main():
    state={'started_unix':time.time(),'complete':False,'error':None,'labels':[]};path=OUT/'MISS_CONTROLLER.json';assert not path.exists()
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        run('prepare_hot_profiles.py')
        for profile,labels in [('hot-disabled',['hot-disabled-r1']),('hot-miss',['hot-miss-r1','hot-miss-r2']),('hot-disabled',['hot-disabled-r2'])]:
            run('switch_and_wait.py',profile)
            for label in labels:
                run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
                run('fresh_probe.py','--label',label);state['labels'].append(label);save()
        run('analyze_heldout.py','--control','hot-disabled-r1','hot-disabled-r2','--candidate','hot-miss-r1','hot-miss-r2','--out','hot-ffn/MISS_ABBA.json','--scope','Samebinary dedicatedhostdispatcher disabled vs allmissCPU;originalGPUexpertlayers37-39 retained;8fixedinputsx2,128completeIDs;ABBA','--final')
        # The positive all-miss runs were stopped before the final control reload;
        # destructor counters prove real execution, not a guard silently bypassing it.
        text=(R.parent/'deepseek41-bringup-20261101/server.stderr.log').read_text(errors='replace')
        rows=[]
        for match in re.finditer(r'xeon-hot-stats layer=(\d+) calls=(\d+) H=\[([0-9,]+)\]',text):
            hist=list(map(int,match[3].split(',')));calls=int(match[2])
            if calls>=2000:rows.append({'layer':int(match[1]),'calls':calls,'H_histogram':hist})
        assert len(rows)==37 and {x['layer'] for x in rows}==set(range(37))
        assert len({x['calls'] for x in rows})==1 and all(x['H_histogram']==[x['calls'],0,0,0,0,0,0] for x in rows)
        (OUT/'MISS_DISPATCH.json').write_text(json.dumps({'passed':True,'layers':rows,'scope':'Actual destructor counters from allmiss API process; no user prompt text copied'},indent=2)+'\n')
        run('switch_and_wait.py','final');state['complete']=True
    except BaseException as e:state['error']=repr(e);raise
    finally:state['finished_unix']=time.time();save()
if __name__=='__main__':main()
