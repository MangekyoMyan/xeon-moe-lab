"""Verify the measured twelve-core topology before starting a long model load."""
from pathlib import Path
import os
import sys


def main():
    command=sys.argv[1:]
    if command[:1]==['--']:command=command[1:]
    if not command:raise SystemExit('Missing model command')
    # The server normalizes underscores and accepts aliases. Refuse competing
    # forms instead of predicting parser precedence across wrappers and env.
    forbidden={'--threads','-C','-Cr','--cpu-range'}
    for token in command:
        option=token.split('=',1)[0].replace('_','-') if token.startswith('--') else token
        if option in forbidden or (option in {'--cpu-mask','--cpu-strict'} and token!=option):
            raise SystemExit('Use only the canonical validated CPU arguments; conflicting alias: '+option)
    for name in ['LLAMA_ARG_THREADS','LLAMA_ARG_CPU_MASK','LLAMA_ARG_CPU_RANGE','LLAMA_ARG_CPU_STRICT']:
        if os.environ.get(name):raise SystemExit('Remove conflicting '+name+' before using this measured profile.')
    def argument(name):
        positions=[i for i,v in enumerate(command) if v==name]
        if len(positions)!=1 or positions[0]+1>=len(command):raise SystemExit('Row-local profile requires explicit '+name)
        return command[positions[0]+1]
    if argument('-t')!='12' or argument('--cpu-strict')!='1' or argument('--cpu-mask')!='0xffffff':
        raise SystemExit('This row-local profile requires the validated12-thread strict CPU mask. Use an interleave profile after changing CPUs or thread settings.')
    if os.environ.get('OMP_DYNAMIC','').upper() not in {'FALSE','0'} or int(os.environ.get('OMP_THREAD_LIMIT','0'))<12:
        raise SystemExit('Measured profile requires OMP_DYNAMIC=FALSE and OMP_THREAD_LIMIT>=12.')
    plan=os.environ.get('GGML_CPU_IQ_PLAN','')
    if not plan or not Path(plan).is_file():raise SystemExit('A validated explicit IQ plan is required for fail-closed loading.')
    cores=set()
    for cpu in range(12):
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}')
        if not (root/f'node{0 if cpu<6 else 1}').exists():raise SystemExit('CPU/NUMA topology changed; use an interleave profile and recalibrate before this row-local profile.')
        cores.add(tuple(int((root/'topology'/key).read_text()) for key in ['physical_package_id','core_id']))
    if len(cores)!=12 or not set(range(12))<=os.sched_getaffinity(0):raise SystemExit('The twelve measured physical CPUs are not available.')
    os.execvp(command[0],command)


if __name__=='__main__':main()
