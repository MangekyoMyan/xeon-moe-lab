from pathlib import Path
import json,subprocess,hashlib
R=Path(__file__).resolve().parent;OLD=R.parent/'iq-cache-next-20261103';OUT=R/'cache120'
source=(OLD/'test-cache.cpp').read_text()
old='need(!xeon_iq3_cache_create_v1((size_t(96)<<30)+1,0,1),"cap check");'
new='''need(!xeon_iq3_cache_create_v1((size_t(128)<<30)+1,0,1),"cap check");
    void * extended=xeon_iq3_cache_create_v1(size_t(120)<<30,size_t(64)<<30,12);
    need(extended,"120GiB owner supported without allocating weights");xeon_iq3_cache_free_v1(extended);'''
assert source.count(old)==1;source=source.replace(old,new)
(OUT/'test-cache.cpp').write_text(source)
cmd=json.loads((OLD/'TEST_BUILD.json').read_text())
cmd=[str(OUT/'test-cache.cpp') if a==str(OLD/'test-cache.cpp') else str(OUT/'test-cache') if a==str(OLD/'test-cache') else a.replace(str(OLD/'runtime'),str(OUT/'runtime')) for a in cmd]
subprocess.run(cmd,check=True,timeout=120)
(OUT/'TEST_BUILD.json').write_text(json.dumps({'command':cmd,'binary_sha256':hashlib.sha256((OUT/'test-cache').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((OUT/'test-cache.cpp').read_bytes()).hexdigest()},indent=2)+'\n')
