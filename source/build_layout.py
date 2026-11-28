"""Rebuild only the KV serialization object of the verified private x9ram libllama."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(source, build_dir, link_receipt, out):
    out.mkdir(parents=True, exist_ok=False)
    original = source / "src/llama-kv-cache.cpp"
    unit = out / original.name
    unit.write_bytes(original.read_bytes())
    patch = Path(__file__).with_name("physical-layout.patch")
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", str(unit), "-i", str(patch.resolve())], check=True
    )
    database = json.loads((build_dir / "compile_commands.json").read_bytes())
    row = next(r for r in database if Path(r["file"]).resolve() == original.resolve())
    cc = shlex.split(row["command"])
    obj = out / (unit.name + ".o")
    cc[cc.index("-c") + 1] = str(unit)
    cc[cc.index("-o") + 1] = str(obj)
    subprocess.run(cc, cwd=row["directory"], check=True)
    link = next(
        c
        for c in json.loads(link_receipt.read_bytes())["commands"]
        if isinstance(c, dict) and "argv" in c
    )
    args = [s for s in link["argv"] if not s.startswith("-Wl,--dependency-file=")]
    dso = out / "libllama.so.0.8.0"
    args[args.index("-o") + 1] = str(dso)
    old = next(s for s in args if s.endswith("/llama-kv-cache.cpp.o"))
    # Verify every other link input by reconstructing the installed library first.
    base = out / "base-relink.so"
    args[args.index("-o") + 1] = str(base)
    subprocess.run(args, cwd=link["cwd"], check=True)
    base_hash = sha(base)
    if base_hash != "1d1e7d722c050856ddff874b5f7b87080925555e7eb9580a7ca37fcafb2a8f26":
        raise ValueError("original linker inputs do not reproduce the deployed model library")
    base.unlink()
    args[args.index("-o") + 1] = str(dso)
    args[args.index(old)] = str(obj)
    subprocess.run(args, cwd=link["cwd"], check=True)
    for name in ("libllama.so", "libllama.so.0"):
        (out / name).symlink_to(dso.name)
    (out / "BUILD.json").write_text(
        json.dumps(
            dict(
                base_sha256=base_hash,
                source_original=sha(original),
                source_private=sha(unit),
                patch_sha256=sha(patch),
                library_sha256=sha(dso),
                compile=cc,
                link=args,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "build", "link-receipt", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    build(a.source.resolve(), a.build.resolve(), a.link_receipt.resolve(), a.output.resolve())
