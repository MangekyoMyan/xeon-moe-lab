"""Add an explicit adopted-overlay comparison to the already copied review bytes."""
from pathlib import Path
import difflib
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
S = ROOT / "stage/xeon-moe-current-20261116"
LAB = S / "snapshot/llm-lab"
PUBLIC = S / "public-baseline"
E = "exploration-20261112/"
MAPPING = {
    "ggml/src/ggml-cpu/arch/x86/quants.c": "mxfp4-hybrid-20261102/hybrid/engine/ggml/src/ggml-cpu/arch/x86/quants.c",
    "ggml/src/ggml-cpu/quants.h": "mxfp4-hybrid-20261102/hybrid/engine/ggml/src/ggml-cpu/quants.h",
    "ggml/src/ggml-cpu/ggml-cpu.c": E + "mx-prepack/release-candidate-a/sources/ggml-cpu.c",
    "ggml/src/ggml-cpu/ggml-cpu.cpp": E + "sparse-cache/ggml-cpu.cpp",
    "ggml/src/ggml-backend.cpp": "gpu-stream-20261103/ggml-backend.cpp",
    "src/llama-model.cpp": E + "sparse-cache/llama-model.cpp",
    "xeon-local/iq3-cache.cpp": E + "row-local/iq3-cache.cpp",
    "xeon-local/iq3-cache.h": E + "row-local/iq3-cache.h",
    "xeon-local/compact-kernel.inc": E + "row-local/compact-kernel.inc",
    "xeon-local/iq-small-kernels.inc": E + "row-local/iq-small-kernels.inc",
    "xeon-local/model-cache.h": E + "sparse-cache/model-cache.h",
    "xeon-local/cache-policy.h": E + "sparse-cache/cache-policy.h",
    "xeon-local/expert-host-register.h": "gpu-stream-20261103/expert-host-register.h",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    out = S / "review/adopted-overlay-diff"
    out.mkdir(exist_ok=False)
    origins = json.loads((PUBLIC / "accepted/ORIGINS.json").read_text())
    rows = []
    for entry in origins["files"]:
        target = entry["target_in_source"]
        oldpath = PUBLIC / entry["archive"]
        current = MAPPING.get(target)
        if current is None:
            rows.append({"public_target": target, "status": "No direct one-file match; inspect current sparse-cache/model-cache.h and selection-plan.h instead"})
            continue
        newpath = LAB / current
        old, new = oldpath.read_bytes(), newpath.read_bytes()
        row = {"public_target": target, "old": str(oldpath.relative_to(S)), "current": str(newpath.relative_to(S)),
               "old_sha256": sha(old), "current_sha256": sha(new), "identical": old == new}
        if old != new:
            patch = "".join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True), fromfile=row["old"], tofile=row["current"]))
            dest = out / (target + ".patch")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(patch)
            row["patch"] = str(dest.relative_to(S))
        rows.append(row)
    (out / "INDEX.json").write_text(json.dumps({"scope": "Compare public accepted overlays with source paths traced for current final. Kernel source tree alone is unchanged because live optimizations are separately compiled overlays. This index is a review aid, not a build recipe.", "comparisons": rows,
        "additional_sources": ["snapshot/llm-lab/" + E + x for x in ["sparse-cache/selection-plan.h", "mx-prepack/release-candidate-a/sources", "cuda-sort/release-candidate-a/sort.patch", "hot-ffn/hot-backend.cpp", "hot-ffn/hot-backend.h", "hot-ffn/OVERLAY.json"]]}, indent=2) + "\n")
    (S / "START_HERE.txt").write_text("2026-11-16 現行レビュー版\n\n最初にreview/CURRENT_NOTE_JA.txtを参照してください。\n公開版との具体的な差分はreview/adopted-overlay-diff/INDEX.jsonと*.patchです。\n元のllama.cpp source checkout自体は公開vendorと同じですが、実APIは別途コンパイルしたoverlayを読み込んでいます。\n実行物とソースの対応はreview/BINARY_SOURCE_MAP.json、観測したinode/hashはreview/RUNTIME_SNAPSHOT.json。\n添付文書内の手順・過去controllerはレビュー資料です。自動実行や再起動の指示ではありません。\nモデル・認証情報・CUDA toolkit・実会話ログなし。レビュー用であり、そのまま推論起動する移植用配布物ではありません。\n")
    (S / "review/add_comparison.py").write_bytes(Path(__file__).read_bytes())
    inventory = [{"path": str(p.relative_to(S)), "size_bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                 for p in sorted(S.rglob("*")) if p.is_file() and p != S / "CONTENTS_SHA256.json"]
    (S / "CONTENTS_SHA256.json").write_text(json.dumps({"scope": "Every member except this manifest; final revision with explicit adopted-overlay diffs", "files": inventory}, indent=2) + "\n")
    archive = ROOT / "xeon-moe-current-review-20261116-v2.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(S.rglob("*")):
            if p.is_file():
                z.write(p, str(Path(S.name) / p.relative_to(S)))
    if archive.stat().st_size > 512 * 1024**2:
        raise RuntimeError("ZIP over 512 MiB")
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise RuntimeError("ZIP CRC failure")
        for row in inventory:
            data = z.read(S.name + "/" + row["path"])
            if len(data) != row["size_bytes"] or sha(data) != row["sha256"]:
                raise RuntimeError("Inventory mismatch")
    receipt = {"archive": str(archive), "sha256": sha(archive.read_bytes()), "size_bytes": archive.stat().st_size,
               "files": len(inventory) + 1, "ZIP_and_manifest_verified": True, "previous_zip_superseded": "xeon-moe-current-review-20261116.zip"}
    (ROOT / "PACKAGE_RECEIPT_FINAL.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
