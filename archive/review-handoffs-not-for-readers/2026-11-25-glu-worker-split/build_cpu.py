"""Rebuild the original Ivy Bridge CPU operator and an isolated GLU scheduling variant."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
LAB = HERE.parent
CPU = LAB / "worker-balance-20261124/cpu"
PROVENANCE = LAB / "ecc9-simplification-20261119/CPU_OBJECT_PROVENANCE.json"
OPS = "CMakeFiles/ggml-cpu.dir/ggml-cpu/ops.cpp.o"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, result):
    path.write_text(json.dumps(result, indent=2) + "\n")


def new_source(src):
    intro = '''// Split each of the six contiguous F32 rows over the 24 physical workers.
// The CPU backend can reuse an input buffer for an elementwise output.
static bool swiglu_decode_layout(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->ne[0] == 2304 && t->ne[1] == 6 &&
           t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float) &&
           t->nb[1] >= 2304 * sizeof(float) && t->nb[1] % sizeof(float) == 0;
}

static bool swiglu_independent_elements(const ggml_tensor * input, const ggml_tensor * output) {
    const uintptr_t src = reinterpret_cast<uintptr_t>(input->data);
    const uintptr_t dst = reinterpret_cast<uintptr_t>(output->data);
    if (src == dst) {
        return input->nb[1] == output->nb[1];
    }
    return src < dst ? dst - src >= ggml_nbytes(input) : src - dst >= ggml_nbytes(output);
}

'''
    name = "static void ggml_compute_forward_swiglu_clamp_f32("
    end = src.index("static void ggml_compute_forward_swiglu_clamp_f16(")
    a = src.index(name)
    body = src[a:end]
    before = """    const int dr  = (nr + nth - 1) / nth;
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);"""
    after = """    const bool split_channels = nth == 24 && src1 && swiglu_decode_layout(dst) &&
            swiglu_decode_layout(src0) && swiglu_decode_layout(src1) &&
            swiglu_independent_elements(src0, dst) && swiglu_independent_elements(src1, dst);
    const int dr  = (nr + nth - 1) / nth;
    const int ir0 = split_channels ? 0 : dr * ith;
    const int ir1 = split_channels ? nr : MIN(ir0 + dr, nr);
    const int k0  = split_channels ? 96 * ith : 0;
    const int k1  = split_channels ? k0 + 96 : nc;"""
    assert body.count(before) == 1 and body.count("for (int k = 0; k < nc; k++)") == 2
    body = body.replace(before, after).replace("for (int k = 0; k < nc; k++)","for (int k = k0; k < k1; k++)")
    return src[:a] + intro + body + src[end:]


def main():
    rec = next(o for o in read(PROVENANCE)["objects_by_arm"] if o["source"].endswith("/ops.cpp"))
    src = Path(rec["source"])
    assert sha(src) == rec["source_bytes_sha256_now"] == "69d1b0b6a8514d8c3070a2dd7576b02ec0691d5bc4bf14a382d45c37c131613d"
    assert sha(rec["object"]) == rec["object_sha256"] == "f68e39b8eca1c97f6fc88c5adef5f27cb3dba23743a2a65950f9bd40bc56353a"
    src_g = HERE/"candidate"/"ops.cpp"
    src_g.parent.mkdir(exist_ok=True)
    src_g.write_text(new_source(src.read_text()))
    (HERE/"glu-split.patch").write_text("".join(difflib.unified_diff(
        src.read_text().splitlines(keepends=True), src_g.read_text().splitlines(keepends=True),
        fromfile="a/ggml/src/ggml-cpu/ops.cpp", tofile="b/ggml/src/ggml-cpu/ops.cpp")))
    base = read(CPU/"rebuild/BUILD.json")
    assert all(sha(path) == value for path, value in base["input_objects"].items())
    report = {"original_source": str(src), "source_sha256": sha(src), "candidate_source_sha256": sha(src_g),
              "patch_sha256": sha(HERE/"glu-split.patch"), "original_object":rec["object"],
              "original_object_sha256": rec["object_sha256"], "original_cpu_sha256":base["sha256"],
              "compiler":subprocess.check_output(["c++","--version"],text=True).splitlines()[0],
              "reused_link_inputs_sha256":base["input_objects"], "arms":{}}
    for name, source in (("O",src),("G",src_g)):
        dest=HERE/"build"/name
        dest.mkdir(exist_ok=True,parents=True)
        obj=dest/"ops.cpp.o"
        cmd=list(rec["argv"])
        cmd[cmd.index("-o")+1]=str(obj)
        cmd[cmd.index("-c")+1]=str(source)
        if name=="G":
            cmd[1:1]=["-iquote", str(src.parent), f"-ffile-prefix-map={source}={src}"]
        with (dest/"compile.log").open("w") as log:
            subprocess.run(cmd, cwd=rec["cwd"], stdout=log, stderr=subprocess.STDOUT, check=True)
        if name=="O":
            assert sha(obj)==rec["object_sha256"]
        outputs = {}
        for diagnostic in (False,True):
            runtime = dest/("trace" if diagnostic else "runtime")
            runtime.mkdir(exist_ok=True)
            lib=runtime/"libggml-cpu.so.0.28.0"
            link=list(base["link"]);link[link.index(OPS)] = str(obj)
            if diagnostic:
                old = str(CPU/"rebuild/ggml-cpu.c.o")
                link[link.index(old)]=str(CPU/"trace/ggml-cpu.c.o")
            link[link.index("-o")+1]=str(lib)
            dep=next(i for i,x in enumerate(link) if x.startswith("-Wl,--dependency-file="))
            link[dep]=f"-Wl,--dependency-file={runtime}/link.d"
            subprocess.run(link,cwd=base["link_cwd"],check=True)
            for alias in ("libggml-cpu.so","libggml-cpu.so.0"):
                if not (runtime/alias).exists():
                    (runtime/alias).symlink_to(lib.name)
            if name=="O":
                assert sha(lib)==(read(CPU/"trace/BUILD.json")["sha256"] if diagnostic else base["sha256"])
            outputs[runtime.name]={"file":str(lib), "sha256":sha(lib), "link":link, "cwd":base["link_cwd"]}
        report["arms"][name]={"compile":cmd, "compile_cwd":rec["cwd"], "object":str(obj),"object_sha256":sha(obj),"outputs":outputs}
    report["diagnostic_c_object_sha256"]=sha(CPU/"trace/ggml-cpu.c.o")
    save(HERE/"BUILD.json",report)
    print({k:v["outputs"]["runtime"]["sha256"] for k,v in report["arms"].items()})


if __name__ == "__main__":
    main()
