"""Inspect the two saved F graphs and the associated debug sections. No model execution."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re


CASES = ("text_cache_budget_ja", "code_stable_grouping_python")
COMMIT = "169a2f7eab8d52071661f0da6859d38df4206f8b"


def load_case(root, case, logs, identities):
    tag = f"case={case} arm=F".encode()
    before = b"FBC_TRACE_BEGIN " + tag
    after = b"FBC_TRACE_END " + tag
    assert logs.count(before) == logs.count(after) == 1
    marked = logs.split(before, maxsplit=1)[1].split(after, maxsplit=1)[0]
    split_rows = {}
    printed_nodes = {}
    cpu_input_text = {}
    for line in marked.splitlines():
        head = re.match(rb"## SPLIT #(\d+): (\S+) # (\d+) inputs?(.*)", line)
        if head:
            index = int(head[1])
            assert index not in split_rows
            split_rows[index] = head[2].decode("ascii")
            cpu_input_text[index] = head[4]
        node = re.match(rb"node #\s*(\d+) \(\s*([^)]*)\):\s+(\S+)\s+\(", line)
        if node:
            printed_nodes[node[3].decode("ascii", "replace")] = index
    assert sorted(split_rows) == list(range(76))

    phases = {}
    for phase in ("pre", "post"):
        path = root / f"{case}-graph-{phase}.json"
        data = path.read_bytes()
        identities[path.name] = sha256(data).hexdigest()
        phases[phase] = json.loads(data)
    post = phases["post"]
    assert post["scheduler_splits"] == len(split_rows)
    before_tensors = {a["id"]: a for a in phases["pre"]["tensors"]}
    tensors = {a["id"]: a for a in post["tensors"]}
    index = {node: i for i, node in enumerate(post["nodes"])}
    nodes = {tensors[node]["name"]: tensors[node] for node in index}
    uses = defaultdict(set)
    for node in index:
        for src in tensors[node]["src"]:
            if src in tensors:
                uses[src].add(node)
    details = []
    weight_bytes = Counter()
    for layer in range(37):
        def pick(name):
            return nodes[f"{name}-{layer}"]

        norm = pick("ffn_norm")
        ids = pick("ffn_moe_topk")
        routed = [pick(key) for key in ("ffn_moe_gate", "ffn_moe_up",
                                        "ffn_moe_swiglu_limited", "ffn_moe_down")]
        shared = [pick(key) for key in ("ffn_gate", "ffn_up", "ffn_swiglu_limited", "ffn_shexp")]
        weighted = pick("ffn_moe_weighted")
        subtotal = pick("ffn_moe_out")
        merge = pick("ffn_out")
        assert [t["op_description"] for t in routed] == ["MUL_MAT_ID", "MUL_MAT_ID", "SWIGLU_CLAMP", "MUL_MAT_ID"]
        assert [t["op_description"] for t in shared] == ["MUL_MAT", "MUL_MAT", "SWIGLU_CLAMP", "MUL_MAT"]
        assert all(t["assigned_backend"] == "CPU" for t in routed)
        assert all(t["assigned_backend"] == "CUDA0" for t in [norm, ids, *shared, weighted, subtotal, merge])
        assert [weighted["op"], subtotal["op"], merge["op"]] == ["MUL", "ADD", "ADD"]
        pre_gate, pre_up = [before_tensors[t["id"]] for t in routed[:2]]
        assert pre_gate["src"][1:3] == pre_up["src"][1:3]
        assert pre_gate["src"][2] == ids["id"]
        norm_reshape = before_tensors[pre_gate["src"][1]]
        assert norm_reshape["op"] == "RESHAPE" and norm_reshape["src"][0] == norm["id"]
        assert routed[0]["src"][1:3] == routed[1]["src"][1:3]
        assert before_tensors[weighted["id"]]["src"][0] == routed[3]["id"]
        assert routed[2]["src"][:2] == [routed[0]["id"], routed[1]["id"]]
        assert routed[3]["src"][1] == routed[2]["id"]
        assert shared[0]["src"][1] == shared[1]["src"][1] == norm["id"]
        assert shared[2]["src"][:2] == [shared[0]["id"], shared[1]["id"]]
        assert shared[3]["src"][1] == shared[2]["id"]
        assert merge["src"][:2] == [subtotal["id"], shared[3]["id"]]
        assert [uses[t["id"]] for t in shared] == [
            {shared[2]["id"]}, {shared[2]["id"]}, {shared[3]["id"]}, {merge["id"]}
        ]
        pos = lambda values: [index[t["id"]] for t in values]
        assert max(pos([*routed, weighted, subtotal])) < min(pos(shared)) < index[merge["id"]]
        cpu_split = printed_nodes[routed[0]["name"]]
        gpu_split = printed_nodes[shared[-1]["name"]]
        assert gpu_split == cpu_split + 1
        assert split_rows[cpu_split] == "CPU" and split_rows[gpu_split] == "CUDA0"
        assert f"[ffn_norm-{layer} (reshaped)".encode() in cpu_input_text[cpu_split]
        assert f"[ffn_moe_topk-{layer} ".encode() in cpu_input_text[cpu_split]
        assert f"[ffn_moe_down-{layer} ".encode() in cpu_input_text[gpu_split]
        assert printed_nodes[merge["name"]] == gpu_split
        weights = [tensors[t["src"][0]] for t in (shared[0], shared[1], shared[3])]
        assert all(t["buffer_type"] == "CUDA0" and t["ne"][2:] == [1, 1] and t["nb"][2] == t["nb"][3] for t in weights)
        for t in weights:
            weight_bytes[t["type"]] += t["nb"][2]
        details.append({"layer": layer, "routed_cpu_split": cpu_split, "shared_cuda_split": gpu_split,
                        "routed_cpu_nodes": pos(routed), "shared_cuda_nodes": pos(shared), "merge_node": index[merge["id"]]})
    return {"case": case, "layer_count": len(details), "layers": details,
            "distinct_weight_tensor_bytes_by_type": dict(weight_bytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Exported 2026-11-19-fbc-h1-iq08 directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    directory = args.source / "fbc/runs/quality-F"
    log = (directory / "server.stderr.log").read_bytes()
    identities = {"server.stderr.log": sha256(log).hexdigest()}
    observations = [load_case(directory, case, log, identities) for case in CASES]
    result = {"commit": COMMIT, "inputs_under": "fbc/runs/quality-F", "sha256": identities,
              "interpretation": "One first-T1 F graph per case. Dependencies, placement and planned split order only. No hardware timing, memory traffic or performance forecast.",
              "observations": observations}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
