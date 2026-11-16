"""Build only the reviewed mixed harness; never rebuild CPU libraries."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent
S=R/'incoming/xeon_moe_abs_next_2026-11-16/next_experiment'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=R/'build';out.mkdir(exist_ok=True)
    previous=LAB/'optimization-20261116/abs-operator/FIXED_BUILD.json'
    command=json.loads(previous.read_text())['argv']
    command[command.index(str(LAB/'optimization-20261116/abs-operator/fixed_probe.cpp'))]=str(S/'mixed_probe.cpp')
    command.insert(1,'-I'+str(S/'case'));command[command.index('-o')+1]=str(out/'mixed_probe')
    subprocess.run(command,check=True,timeout=180)
    receipt={'scope':__doc__,'argv':command,'source_sha256':sha(S/'mixed_probe.cpp'),'case_header_sha256':sha(S/'case/mixed_case_data.h'),'harness_sha256':sha(out/'mixed_probe'),'previous_build_receipt_sha256':sha(previous),'cpu_library_rebuilt':False}
    (out/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
