"""Build only the reviewed Q8_K wrapper change; preserve the live CPU DSO."""
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess
R=Path(__file__).resolve().parent
LAB=R.parent
OUT=R/'q8-avx1'
HY=LAB/'mxfp4-hybrid-20261102/hybrid'
CURRENT=LAB/'exploration-20261112/mx-prepack/release-candidate-a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'BUILD.json').exists();OUT.mkdir(exist_ok=True)
    patch=R/'incoming/xeon_moe_fd810_followup_2026-11-18/previous_handoff/xeon_moe_combined_handoff_2026-11-18/astra62_max_trial_2026-11-18/candidate/04-q8k-avx1.patch'
    source=HY/'engine/ggml/src/ggml-cpu/arch/x86/quants.c'
    assert sha(source)=='a01b2448f3f87325a2594c05528b6f15e5e117b7abfb28fb2714c1ec474bf5f7'
    expected='33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3'
    assert sha(CURRENT/'libggml-cpu.so.0.28.0')==expected
    db=json.loads((HY/'build-cpu-ivybridge/compile_commands.json').read_text())
    entry=next(x for x in db if x['file']==str(source));oldcc=shlex.split(entry['command'])
    assert '-mno-avx2' in oldcc and '-mno-fma' in oldcc and '-ffast-math' not in oldcc
    oldobj=Path(entry['directory'])/oldcc[oldcc.index('-o')+1]
    link=json.loads((CURRENT/'BUILD.json').read_text())['variants']['candidate']['link']
    receipt=dict(patch_sha256=sha(patch),source_sha256=sha(source),original_CPU_sha256=expected,original_object_sha256=sha(oldobj),compile_database_entry=entry,variants={})
    for variant in ['control','candidate']:
        d=OUT/variant;d.mkdir(exist_ok=True);src=d/'source/ggml/src/ggml-cpu/arch/x86/quants.c';src.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,src)
        if variant=='candidate':subprocess.run(['git','apply','--check',str(patch)],cwd=d/'source',check=True);subprocess.run(['git','apply',str(patch)],cwd=d/'source',check=True)
        obj=d/'quants.c.o';cc=[str(src) if x==str(source) else x for x in oldcc];cc[cc.index('-o')+1]=str(obj)
        # Resolve original quoted headers and normalize source location strings;
        # no arithmetic optimization flags are changed from the original object.
        cc+=['-iquote',str(source.parent),'-ffile-prefix-map='+str(src)+'='+str(source)]
        with (d/'compile.stdout.log').open('w') as out,(d/'compile.stderr.log').open('w') as err:subprocess.run(cc,cwd=entry['directory'],stdout=out,stderr=err,check=True,timeout=240)
        runtime=d/'runtime';runtime.mkdir(exist_ok=True);target=runtime/'libggml-cpu.so.0.28.0';argv=list(link['argv']);argv[argv.index('-o')+1]=str(target)
        for i,x in enumerate(argv):
            if x.startswith('-Wl,--dependency-file='):argv[i]='-Wl,--dependency-file='+str(d/'link.d')
            elif x.endswith('ggml-cpu/arch/x86/quants.c.o'):argv[i]=str(obj)
        with (d/'link.stdout.log').open('w') as out,(d/'link.stderr.log').open('w') as err:subprocess.run(argv,cwd=link['cwd'],stdout=out,stderr=err,check=True,timeout=120)
        for n in ['libggml-cpu.so','libggml-cpu.so.0']:(runtime/n).symlink_to(target.name)
        inputs={}
        for x in argv:
            if x.endswith(('.o','.a','.so','.0.28.0')):
                p=Path(x);p=p if p.is_absolute() else Path(link['cwd'])/p
                if p.is_file():inputs[str(p.resolve())]=sha(p)
        rec=dict(compile=cc,compile_cwd=entry['directory'],link=argv,link_cwd=link['cwd'],source_sha256=sha(src),object_sha256=sha(obj),CPU_sha256=sha(target),link_inputs=inputs)
        receipt['variants'][variant]=rec
        if variant=='control':
            receipt['unchanged_object_byte_equal']=sha(obj)==sha(oldobj)
            receipt['control_CPU_byte_equal_live']=sha(target)==expected
            (OUT/'CONTROL_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
            assert receipt['unchanged_object_byte_equal'] and receipt['control_CPU_byte_equal_live'],'Stop if current source/object/link identity does not reproduce'
        print(variant,rec['CPU_sha256'],flush=True)
    receipt['live_CPU_unchanged']=sha(CURRENT/'libggml-cpu.so.0.28.0')==expected
    assert receipt['live_CPU_unchanged'] and receipt['variants']['candidate']['CPU_sha256']!=expected
    (OUT/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
