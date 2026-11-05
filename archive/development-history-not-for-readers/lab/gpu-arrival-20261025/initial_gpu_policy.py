#!/usr/bin/python3
"""Apply the agreed PG500 policy at boot; never run a GPU workload."""
import subprocess
GPU_ID = "GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3"


def run(*args):
    result = subprocess.run(args, check=True, text=True, capture_output=True, timeout=45)
    print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    return result.stdout.strip()


run("/usr/bin/nvidia-modprobe")
assert run("/usr/bin/nvidia-smi", "-i", GPU_ID, "--query-gpu=name", "--format=csv,noheader") == "Tesla PG500-216"
# Keep the GPU initialized so that the software power cap survives idle periods.
run("/usr/bin/nvidia-smi", "-i", GPU_ID, "-pm", "1")
run("/usr/bin/nvidia-smi", "-i", GPU_ID, "-pl", "150")
run("/usr/bin/nvidia-smi", "-i", GPU_ID, "-e", "1")
power = run("/usr/bin/nvidia-smi", "-i", GPU_ID,
            "--query-gpu=power.limit", "--format=csv,noheader,nounits")
assert abs(float(power) - 150.0) < 0.1, power
run("/usr/bin/nvidia-smi", "-i", GPU_ID,
    "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,power.limit,"
    "temperature.gpu,persistence_mode,ecc.mode.current,ecc.mode.pending", "--format=csv")
print("No workload started; any pending ECC change requires another reboot.", flush=True)
