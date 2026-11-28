"""Build the two private HTTP adapter translations from an existing x9ram build."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(source, build_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    units = ("server-context.cpp", "server-task.cpp")
    work = out / "src"
    for name in units:
        p = work / "tools/server" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "tools/server" / name, p)
    shutil.copyfile(Path(__file__).with_name("prefix-capture.hpp"), work / "tools/server/prefix-capture.hpp")
    patch = Path(__file__).with_name("native-stream.patch")
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch.resolve())], cwd=work, check=True
    )
    database = json.loads((build_dir / "compile_commands.json").read_text())

    def compile_one(name):
        (entry,) = [v for v in database if Path(v["file"]).name == name]
        assert sha(Path(entry["file"])) == sha(source / "tools/server" / name)
        args = shlex.split(entry["command"])
        args[args.index("-o") + 1] = str(out / (name + ".o"))
        args[args.index("-c") + 1] = str(work / "tools/server" / name)
        subprocess.run(args, cwd=entry["directory"], check=True)
        return args

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        commands = list(pool.map(compile_one, units))
    server = build_dir / "tools/server"
    archive = shlex.split(
        (server / "CMakeFiles/server-context.dir/link.txt").read_text().splitlines()[0]
    )
    static = out / "libserver-context.a"
    archive[2] = str(static)
    objects = []
    for i in range(3, len(archive)):
        original = server / archive[i]
        path = out / original.name if original.name in {n + ".o" for n in units} else original
        archive[i] = str(path)
        objects.append({"file": path.name, "sha256": sha(path)})
    subprocess.run(archive, cwd=server, check=True)
    subprocess.run(["ranlib", str(static)], check=True)
    args = shlex.split((server / "CMakeFiles/llama-server-impl.dir/link.txt").read_text())
    library = out / "libllama-server-impl.so"
    args[args.index("-o") + 1] = str(library)
    args[args.index("libserver-context.a")] = str(static)
    args = [v for v in args if not v.startswith("-Wl,--dependency-file")]
    subprocess.run(args, cwd=server, check=True)
    record = {
        "patch_sha256": sha(patch),
        "prefix_capture_sha256": sha(work / "tools/server/prefix-capture.hpp"),
        "library_sha256": sha(library),
        "original_source": {n: sha(source / "tools/server" / n) for n in units},
        "private_source": {n: sha(work / "tools/server" / n) for n in units},
        "archive_inputs": objects,
        "compile": commands,
        "link": args,
        "base_server_impl_sha256": sha(build_dir / "bin/libllama-server-impl.so"),
    }
    (out / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
    print(library, record["library_sha256"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("source", "build", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    build(a.source.resolve(), a.build.resolve(), a.output.resolve())
