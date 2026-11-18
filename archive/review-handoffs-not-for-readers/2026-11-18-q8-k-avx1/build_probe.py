from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;Q=R/'q8-avx1';LAB=R.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    src=R/'incoming/xeon_moe_fd810_followup_2026-11-18/previous_handoff/xeon_moe_combined_handoff_2026-11-18/astra62_max_trial_2026-11-18/candidate/q8_probe.cpp'
    shutil.copy2(src,Q/'q8_probe.cpp')
    body=src.read_text().split('int main(int argc, char **argv) {')[0];assert body.endswith('}\n') and 'static void check()' in body
    (Q/'q8_cases.inc').write_text(body)
    s=LAB/'deepseek41-bringup-20261101/source';base=LAB/'gpu-stream-20261103/runtime';control=Q/'control/runtime'
    cmd=['g++','-O3','-std=c++17','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-frounding-math','-fopenmp']
    for d in [s/'ggml/include',s/'ggml/src',s/'vendor/nlohmann']:cmd+=['-I'+str(d)]
    cmd+=[str(Q/'probe_wrapper.cpp'),'-L'+str(control),'-L'+str(base),'-Wl,--no-as-needed','-lggml-cpu','-lggml-base','-ldl','-o',str(Q/'q8_probe')]
    with (Q/'PROBE_BUILD.stdout.log').open('w') as o,(Q/'PROBE_BUILD.stderr.log').open('w') as e:subprocess.run(cmd,stdout=o,stderr=e,check=True,timeout=120)
    (Q/'PROBE_BUILD.json').write_text(json.dumps(dict(command=cmd,provided_probe_unchanged=sha(src)==sha(Q/'q8_probe.cpp'),adaptation='Compile unchanged prefix before supplied main as q8_cases.inc; own main validates providers and returns0 explicitly.',sources={n:sha(Q/n) for n in ['q8_probe.cpp','q8_cases.inc','probe_wrapper.cpp']},binary_sha256=sha(Q/'q8_probe')),indent=2)+'\n')
if __name__=='__main__':main()
