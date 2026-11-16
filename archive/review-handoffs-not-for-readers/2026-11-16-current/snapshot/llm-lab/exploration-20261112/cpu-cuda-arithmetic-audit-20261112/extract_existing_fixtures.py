"""Reformat saved measurements only. Does not load a model or run benchmarks."""
from pathlib import Path
import hashlib
import json
import math

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    capacity_path = ROOT / "hot-ffn/CAPACITY.json"
    layers = {row["layer"]: row for row in json.loads(capacity_path.read_text())["layers"]}
    rows = []
    for source in sorted((ROOT / "execution").glob("hot-ffn-L*-s*/stdout.log")):
        data = json.loads(source.read_text())
        assert data["pass"]
        gpu_runs = [row for row in data["measurements"] if row["hot"] == 6]
        assert len(gpu_runs) == 1
        gpu = gpu_runs[0]
        cpu = data["CPU_error_vs_FP64_dequant_reference"]
        rows.append({
            "source": str(source),
            "source_sha256": digest(source),
            "layer": data["layer"],
            "seed": data["seed"],
            "gate_up_down_weight_types": layers[data["layer"]]["types"],
            "input_amplitude": data["input_amplitude"],
            "routes": data["routes"],
            "cpu_gate_up_activation_down_errors": cpu,
            "gpu_activation_down_errors": gpu["GPU_stage_error_vs_FP64_reference"],
            "cpu_down_error_vs_reference": cpu[3],
            "gpu_down_error_vs_reference": gpu["mixed_error_vs_FP64_reference"],
            "gpu_down_error_vs_cpu": gpu["mixed_error_vs_CPU"],
            "gpu_down_reference_nmse_divided_by_cpu": gpu["mixed_error_vs_FP64_reference"]["nmse"] / cpu[3]["nmse"],
            "gpu_vs_cpu_relative_L2": math.sqrt(gpu["mixed_error_vs_CPU"]["nmse"]),
            "serial_concurrent_byte_identical": gpu["serial_concurrent_byte_identical"],
            "gpu_scatter_byte_identical": gpu["GPU_hot_slots_byte_identical"],
        })
    assert len(rows) == 6
    result = {
        "operation": "Read and reformat six existing fixture logs; no new measurements",
        "reference": "Dequantize existing GGUF weights to F32, accumulate dot products in FP64, round projection outputs to F32; SwiGLU is F32. No activation Q8 quantization in reference. Not an original unquantized model reference.",
        "nmse_definition": "sum((result-reference)^2) / sum(reference^2); not an answer error rate or percent accuracy",
        "scope": "Two layers, three synthetic input seeds, six fixed real experts per layer, input amplitude 0.1. H=6 means all six experts computed by GPU. Does not establish whole-model quality or isolate one arithmetic operation.",
        "source_files": {str(path): digest(path) for path in [capacity_path, ROOT / "hot_ffn_probe.cpp", ROOT / "run_hot_ffn.py", ROOT / "hot-ffn/BUILD.json"]},
        "rows": rows,
    }
    destination = OUT / "EXISTING_NUMERIC_EVIDENCE.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(row["layer"], row["seed"], row["cpu_down_error_vs_reference"]["nmse"], row["gpu_down_error_vs_reference"]["nmse"], row["gpu_down_error_vs_cpu"]["nmse"])
    print(destination)


if __name__ == "__main__":
    main()
