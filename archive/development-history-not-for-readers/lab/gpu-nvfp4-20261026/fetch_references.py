from pathlib import Path
import json, urllib.request, hashlib
p=Path(__file__).resolve().parent/'research'
p.mkdir(exist_ok=True)
for repo in ['dnv2003/v100-skinny','gilby/volta-nvfp4']:
    req=urllib.request.Request('https://api.github.com/repos/'+repo+'/commits/main',headers={'User-Agent':'xeon-volta-lab'})
    with urllib.request.urlopen(req,timeout=30) as r: info=json.load(r)
    sha=info['sha']; out=p/repo.replace('/','--');out.mkdir(exist_ok=True)
    record={'repo':repo,'commit':sha,'files':{}}
    for name in ['README.md','LICENSE','NOTICE']:
        url='https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+name
        with urllib.request.urlopen(url,timeout=30) as r: data=r.read(1024*1024)
        (out/(name+'.reference')).write_bytes(data)
        record['files'][name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(repo,sha)
