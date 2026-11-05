#!/usr/bin/env python3
"""Assemble a self-contained artifact and verify all recorded cases after relocation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

SRC=Path(__file__).resolve().parent
LAB=SRC.parent
OUT=LAB/'results/hybrid-sim-deep-20261023'
ARCHIVE=LAB.parent/'xeon-p40-v100-deep-simulation-20261023.zip'

def copy_file(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)

def include(path):
    if any(p in ['__pycache__','mpl-cache'] for p in path.parts):return False
    if path.name.startswith('.') or path.suffix in ['.pyc','.o','.so']:return False
    if path.name in ['WORK_STATE.txt','BUNDLE.json','PORTABLE_REPLAY_CHECK.json']:return False
    with path.open('rb') as f:
        if f.read(4)==b'\x7fELF':return False
    return True

def main():
    with tempfile.TemporaryDirectory(prefix='xeon-deep-bundle-') as temporary:
        stage=Path(temporary)
        for folder in [SRC,OUT,LAB/'results/expert-cache-evidence-20261022/cachedmoe/bench/results/hitrate/base_auto']:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and not path.is_symlink() and include(path):
                    copy_file(path,stage/'llm-lab'/path.relative_to(LAB))
        phase=LAB/'results/hybrid-sim-20261023/phases.tsv'
        copy_file(phase,stage/'llm-lab'/phase.relative_to(LAB))
        local_source=stage/'llm-lab/hybrid-sim-deep-20261023'
        print('Checking the relocated bundle: all 235 recorded runs',flush=True)
        subprocess.run(['python3',str(local_source/'replay_recorded.py')],check=True)
        portable=stage/'llm-lab/results/hybrid-sim-deep-20261023/REPLAY_ALL_CHECK.json'
        copy_file(portable,OUT/'PORTABLE_REPLAY_CHECK.json')
        copy_file(OUT/'PORTABLE_REPLAY_CHECK.json',portable.with_name('PORTABLE_REPLAY_CHECK.json'))
        (stage/'START_HERE_JA.txt').write_text(
            '旧Xeon + P40/V100 expert cache追加シミュレーション\n'
            '結果: llm-lab/results/hybrid-sim-deep-20261023/REPORT_JA.txt\n'
            '再現: llm-lab/results/hybrid-sim-deep-20261023/README_REPRODUCE.txt\n'
            '全235runを別directoryへ移して再計算し、全summary数値一致を確認済み。\n'
            'GPU実測値ではありません。CPU実測、公式shape、公開route、GPU仮定を区別しています。\n'
            '図は同results directoryのPNG/PDF、sourceはllm-lab/hybrid-sim-deep-20261023。\n')
        manifest=[]
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                manifest.append(digest+'  '+str(path.relative_to(stage)))
        (stage/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
        with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in sorted(stage.rglob('*')):
                if path.is_file():archive.write(path,str(path.relative_to(stage)))
        with zipfile.ZipFile(ARCHIVE) as archive:
            assert archive.testzip() is None
            members=len(archive.infolist())
    digest=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    payload={'archive':str(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'members':members,'sha256':digest,
             'portable_all_case_replay_passed':True,'simulated_runs':235,'distinct_input_configurations':198}
    (OUT/'BUNDLE.json').write_text(json.dumps(payload,indent=2)+'\n')
    ARCHIVE.with_suffix('.zip.sha256').write_text(digest+'  '+ARCHIVE.name+'\n')
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':main()
