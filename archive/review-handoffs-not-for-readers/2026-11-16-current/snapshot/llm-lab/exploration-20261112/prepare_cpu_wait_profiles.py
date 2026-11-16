"""Freeze a small thread/wait-policy screen on an explicitly chosen verified profile."""
from pathlib import Path
import argparse,copy,hashlib,json
R=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--base-profile',required=True);p.add_argument('--include-six',action='store_true');a=p.parse_args()
    assert a.base_profile.replace('-','').isalnum()
    raw=(R/'profiles'/(a.base_profile+'.json')).read_bytes();base=json.loads(raw)
    records=[]
    for threads in ([6] if a.include_six else [])+[12,24]:
        for policy in (['default'] if threads==6 else ['default','spin10000','passive']):
            profile=copy.deepcopy(base);name=f'wait-t{threads}-{policy}'
            cmd=profile['command'];cmd[cmd.index('-t')+1]=str(threads)
            if threads==6:cmd[cmd.index('--cpu-mask')+1]='0x1c7'
            env=profile['environment'];env.pop('OMP_WAIT_POLICY',None);env.pop('GOMP_SPINCOUNT',None)
            if policy=='spin10000':env['GOMP_SPINCOUNT']='10000'
            if policy=='passive':env['OMP_WAIT_POLICY']='PASSIVE'
            assert cmd[cmd.index('-tb')+1]=='24'
            assert cmd[cmd.index('--cpu-mask')+1]==('0x1c7' if threads==6 else '0xffffff')
            assert cmd[cmd.index('--cpu-strict')+1]=='1'
            assert cmd[cmd.index('--cpu-mask-batch')+1]=='0xffffff'
            assert cmd[cmd.index('--cpu-strict-batch')+1]=='1'
            assert env['OMP_THREAD_LIMIT']=='24' and env['OMP_DYNAMIC']=='FALSE'
            assert profile['context']==131072
            profile['optimization_study'].update(profile=name,status='awaiting-thread-policy-screen',generation_threads=threads,wait_policy=policy,base_profile=a.base_profile)
            output=R/'profiles'/(name+'.json');data=(json.dumps(profile,indent=2)+'\n').encode()
            if output.exists() and output.read_bytes()!=data:raise RuntimeError('Refusing to overwrite a previous profile: '+str(output))
            output.write_bytes(data);output.chmod(0o600)
            records.append({'name':name,'threads':threads,'policy':policy,'sha256':hashlib.sha256(data).hexdigest()})
    out=R/'cpu-wait';out.mkdir(exist_ok=True)
    result={'base_profile':a.base_profile,'base_sha256':hashlib.sha256(raw).hexdigest(),'profiles':records,'scope':'12/24 generation threads crossed with default/spin10000/passive; optional6physical balanced3+3; prefill24 unchanged; interleave NUMA','not_executed':True}
    (out/'PROFILES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
