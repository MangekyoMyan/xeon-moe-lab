"""Independent current CPU / 08-only build; two TUs, no 04/09/03."""
from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent;D=R/'cpu08';MX=LAB/'exploration-20261112/mx-prepack';ROW=LAB/'exploration-20261112/row-local'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (D/'BUILD.json').exists();D.mkdir(exist_ok=True)
    mx=json.loads((MX/'release-candidate-a/BUILD.json').read_text());row=json.loads((ROW/'BUILD.json').read_text());recipe=json.loads((R/'astra62-review/08_OBJECT_RECIPE.json').read_text())
    patch=R/recipe['patch']['file'];assert sha(patch)=='cad2e24dea2bc3bfa8d3dc1dd43c2d7930159889dbd49c4f2c4dda4052cbd6a3'
    original=MX/'release-candidate-a/libggml-cpu.so.0.28.0';assert sha(original)=='33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3'
    receipt=dict(patch_sha256=sha(patch),original_CPU_sha256=sha(original),variants={})
    for variant in ['control','candidate']:
        root=D/variant;src=root/'source';src.mkdir(parents=True)
        for n,v in recipe['before_and_after'].items():
            if not v.get('original_path'):continue
            p=Path(v['original_path']);assert sha(p)==v['original_sha256'];q=src/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        for n in ['compact-kernel.inc','iq-small-kernels.inc','placement.inc']:shutil.copy2(ROW/n,src/'xeon-local'/n)
        if variant=='candidate':
            subprocess.run(['git','apply','--check',str(patch)],cwd=src,check=True);subprocess.run(['git','apply',str(patch)],cwd=src,check=True)
            for n,v in recipe['before_and_after'].items():
                if v.get('08_sha256'):assert sha(src/n)==v['08_sha256'],n
        commands=[];objs={}
        for name,cmd0,rel in [('cpu',mx['variants']['candidate']['compile'],'ggml/src/ggml-cpu/ggml-cpu.c'),('cache',row['compile'],'xeon-local/iq3-cache.cpp')]:
            cmd=list(cmd0);oldsrc=Path(cmd[cmd.index('-c')+1]);oldobj=Path(cmd[cmd.index('-o')+1]);obj=root/(name+'.o');cmd[cmd.index('-c')+1]=str(src/rel);cmd[cmd.index('-o')+1]=str(obj)
            cmd.insert(1,'-I'+str(src/'xeon-local'))
            cmd+=['-ffile-prefix-map='+str(src/n)+'='+v['original_path'] for n,v in recipe['before_and_after'].items() if v.get('original_path')]
            with (root/(name+'.build.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True,timeout=300)
            commands.append(dict(argv=cmd,original_object=str(oldobj),object_sha256=sha(obj),original_sha256=sha(oldobj)))
            if variant=='control':assert sha(obj)==sha(oldobj),'control object must reproduce: '+name
            objs[str(oldobj)]=obj
        link=mx['variants']['candidate']['link'];cmd=list(link['argv']);runtime=root/'runtime';runtime.mkdir();lib=runtime/'libggml-cpu.so.0.28.0'
        assert all(cmd.count(x)==1 for x in objs),'each changed object must be replaced once'
        for i,x in enumerate(cmd):
            if x in objs:cmd[i]=str(objs[x])
            elif x.startswith('-Wl,--dependency-file='):cmd[i]='-Wl,--dependency-file='+str(root/'link.d')
        cmd[cmd.index('-o')+1]=str(lib)
        with (root/'link.log').open('w') as log:subprocess.run(cmd,cwd=link['cwd'],stdout=log,stderr=log,check=True,timeout=120)
        if variant=='control':assert sha(lib)==sha(original),'control CPU must reproduce'
        dis=subprocess.check_output(['objdump','-d','--disassemble=quantize_row_q8_K',str(lib)],text=True);assert '<quantize_row_q8_K_ref@plt>' in dis and 'vcvtps2dq' not in dis
        (root/'Q8-wrapper.txt').write_text(dis)
        for n in ['libggml-cpu.so','libggml-cpu.so.0']:(runtime/n).symlink_to(lib.name)
        inputs={}
        for x in cmd:
            if x.endswith(('.o','.a','.so','.0.28.0')):
                p=Path(x);p=p if p.is_absolute() else Path(link['cwd'])/p
                if p.is_file():inputs[str(p.resolve())]=sha(p)
        receipt['variants'][variant]=dict(commands=commands,link=cmd,link_cwd=link['cwd'],CPU_sha256=sha(lib),link_inputs=inputs,sources={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()})
        print('08 '+variant+' '+sha(lib),flush=True)
    (D/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
