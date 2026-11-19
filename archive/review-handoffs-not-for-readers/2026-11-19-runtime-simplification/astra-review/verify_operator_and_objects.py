"""Read existing build objects and synthetic output. No build or inference."""
from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

study = Path(__file__).resolve().parents[1]
lab = study.parent
build = study / "build"
previous = lab / "be05-followup-20261119/cpu08"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def code_and_relocations(path):
    """Compare the arithmetic bodies, referenced data, and relocation locations."""
    obj = path.read_bytes()
    sections = {}
    for line in subprocess.check_output(["readelf", "-SW", path], text=True).splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m:
            start, size = (int(v, 16) for v in m.groups()[1:])
            sections[m.group(1)] = obj[start:start + size]
    symbols = {}
    for line in subprocess.check_output(["objdump", "-t", path], text=True).splitlines():
        p = line.split()
        if len(p) >= 5 and re.fullmatch("[0-9a-f]+", p[0]) and p[-3] in sections:
            symbols[p[-1]] = (p[-3], int(p[0], 16), int(p[-2], 16))
    rels = []
    in_text = False
    for line in subprocess.check_output(["readelf", "-rW", path], text=True).splitlines():
        if line.startswith("Relocation section "):
            in_text = "'.rela.text'" in line
        elif in_text and re.match("[0-9a-f]{16} ", line):
            addr, _, kind, _, sym, sign, delta = line.split()
            rels.append((int(addr, 16), kind, sym, sign + delta))
    result = {}
    for name, (section, start, size) in symbols.items():
        if not (name.startswith(("_ZL11compact_dot", "_Z14iq_compact_dot")) or
                (name.startswith("_ZL12add_selected") and name.endswith("._omp_fn.0"))):
            continue
        relocated = []
        for where, kind, sym, addend in rels:
            if not start <= where < start + size:
                continue
            target = None
            if sym.startswith(".LC"):
                constant_section, offset, _ = symbols[sym]
                width = int(re.fullmatch(r"\.rodata\.cst(\d+)", constant_section).group(1))
                target = sections[constant_section][offset:offset + width].hex()
            elif sym == ".rodata":
                target = digest(sections[sym])
            relocated.append((where - start, kind, sym, addend, target))
        result[name] = {"size": size, "object_code_sha256": digest(sections[section][start:start + size]),
                        "relocations_and_targets": relocated}
    assert len(result) == 4 and all(v["relocations_and_targets"] for v in result.values())
    return result


def replay_raw(raw):
    nr = 2304
    row = nr * 4
    graph = 3 * row
    assert sys.byteorder == "little" and len(raw) == 3 * 5 * graph
    for i in range(3):
        start = i * 5 * graph
        ref, mixed, changed, freed, reload = [raw[start + j * graph:start + (j + 1) * graph] for j in range(5)]
        assert mixed == ref and changed[:row] == ref[:row] and changed[2 * row:] == ref[2 * row:]
        for j in range(3):
            x = array("f", ref[j * row:(j + 1) * row])
            assert all(map(math.isfinite, x)) and any(x)
        assert not any(array("f", changed[row:2 * row] + freed + reload))
    return {"saved_F32": len(raw) // 4, "saved_graph_outputs": 15,
            "bytes": len(raw), "sha256": digest(raw)}


def main():
    historic = load(previous / "BUILD.json")["variants"]["control"]
    for p, sha in historic["link_inputs"].items():
        assert digest(Path(p).read_bytes()) == sha
    checked = load(build / "BUILD.json")
    a, b = (checked[k] for k in ("control", "ordinary"))
    assert digest(Path(a["file"]).read_bytes()) == historic["CPU_sha256"] == a["sha256"]
    assert digest(Path(b["file"]).read_bytes()) == b["sha256"] != a["sha256"]
    inputs_a, inputs_b = (v["link_inputs_sha256"] for v in (a, b))
    assert set(inputs_a) ^ set(inputs_b) == {str(build / "control-cache.o"), str(build / "ordinary-cache.o")}
    common = inputs_a.keys() & inputs_b.keys()
    assert len(common) == 21
    for path in common:
        assert digest(Path(path).read_bytes()) == inputs_a[path] == inputs_b[path]
    env = [b"GGML_CPU_IQ_ROW_LOCAL", b"GGML_CPU_IQ_AUDIT_DIR"]
    assert not any(pat in Path(path).read_bytes() for path in common for pat in env)
    for p, contains in [(build / "control-cache.o", True), (Path(a["file"]), True),
                        (build / "ordinary-cache.o", False), (Path(b["file"]), False)]:
        blob = p.read_bytes()
        assert all(blob.count(pat) == int(contains) for pat in env)
    code = code_and_relocations(build / "control-cache.o")
    assert code == code_and_relocations(build / "ordinary-cache.o")
    old_src = (previous / "control/source/xeon-local/iq3-cache.cpp").read_text()
    for line in ['#include "placement.inc"', '    xeon_placement::options placement{};',
                 '        const bool row_local=xeon_placement::bind_before_touch(o.placement,t,e->data,bytes,selected_count);',
                 '        xeon_placement::audit(o.placement,t,e->data,bytes,selected_count,row_local);']:
        assert old_src.count(line + "\n") == 1
        old_src = old_src.replace(line + "\n", "")
    old_src = old_src.replace("#include <cstring>\n", "#include <cstring>\n#include <exception>\n")
    assert old_src == (study / "current/cpu/iq3-cache.cpp").read_text()
    for filename in ["compact-kernel.inc", "iq-small-kernels.inc", "iq3-cache.h"]:
        assert (previous / "control/source/xeon-local" / filename).read_bytes() == (study / "current/cpu" / filename).read_bytes()
    compact = study / "compact-check"
    raw = (compact / "control.f32").read_bytes()
    assert raw == (compact / "ordinary.f32").read_bytes()
    for arm in ["control", "ordinary"]:
        logged = [json.loads(l) for l in (compact / (arm + ".log")).read_text().splitlines()]
        assert [r["type"] for r in logged[:3]] == ["iq2_s", "iq3_s", "iq3_xxs"]
        assert all(r["resident_pages_by_node"] == [1620, 1620] for r in logged[:3]) and logged[3]["pass"]
    return {"cpu_control": a["sha256"], "cpu_ordinary": b["sha256"],
            "historic_hashes_checked_including_output": len(historic["link_inputs"]),
            "common_named_link_inputs": len(common),
            "both_env_names_confined_to_historic_and_control_cache_object": True,
            "one_TU_source_delta_and_shared_includes_verified": True,
            "matched_relocatable_arithmetic_and_pack_worker": code,
            "synthetic_outputs_per_arm": replay_raw(raw), "pass": True}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
