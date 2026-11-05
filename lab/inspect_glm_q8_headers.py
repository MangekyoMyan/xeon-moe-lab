#!/usr/bin/env python3
"""Inspect previously fetched GGUF headers without reading any weight payload."""
import collections
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/glm-air-q8-preparation-20261020"


class Header:
    def __init__(self, path):
        self.data = path.read_bytes()
        self.pos = 0

    def unpack(self, fmt):
        result = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return result[0] if len(result) == 1 else result

    def string(self):
        size = self.unpack("Q")
        end = self.pos + size
        if end > len(self.data):
            raise ValueError("Header prefix too short")
        result = self.data[self.pos:end].decode("utf-8", errors="replace")
        self.pos = end
        return result

    def value(self, kind):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind in formats:
            return self.unpack(formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            subtype, count = self.unpack("IQ")
            for _ in range(count):
                self.value(subtype)
            return {"array_type": subtype, "count": count}
        raise ValueError(f"Unsupported GGUF metadata type {kind}")

    def parse(self):
        magic, version, tensors, kvs = self.unpack("IIQQ")
        assert magic == 0x46554747 and version == 3
        metadata = {}
        for _ in range(kvs):
            key = self.string()
            metadata[key] = self.value(self.unpack("I"))
        infos = []
        for _ in range(tensors):
            name = self.string()
            dims = [self.unpack("Q") for _ in range(self.unpack("I"))]
            kind, offset = self.unpack("IQ")
            block, size, typename = {
                0: (1, 4, "F32"), 1: (1, 2, "F16"),
                8: (32, 34, "Q8_0"), 30: (1, 2, "BF16"),
            }[kind]
            elements = math.prod(dims)
            assert elements % block == 0
            infos.append(dict(name=name, dims=dims, type=typename,
                              offset=offset, bytes=elements // block * size))
        alignment = metadata.get("general.alignment", 32)
        return dict(metadata=metadata, tensors=infos,
                    data_offset=(self.pos + alignment - 1) // alignment * alignment)


def main():
    parts = {}
    for path in sorted(OUT.glob("*.gguf.header")):
        parts[path.name] = Header(path).parse()
    all_tensors = [t for part in parts.values() for t in part["tensors"]]
    counts = collections.Counter(t["type"] for t in all_tensors)
    sizes = collections.Counter()
    inner_dims = collections.Counter()
    for t in all_tensors:
        sizes[t["type"]] += t["bytes"]
        if t["type"] == "Q8_0":
            inner_dims[t["dims"][0]] += t["bytes"]
    summary = dict(parts=parts, type_counts=dict(counts), type_bytes=dict(sizes),
                   q8_inner_dimensions_bytes=dict(inner_dims),
                   weight_bytes=sum(sizes.values()))
    (OUT / "header-inspection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "parts"}, indent=2))
    for name, part in parts.items():
        print(name, "tensors", len(part["tensors"]), "data_offset", part["data_offset"])
        print(json.dumps({k: v for k, v in part["metadata"].items()
                          if k.startswith(("glm4moe.", "split.", "general.arch"))}, indent=2))
    first = OUT / "GLM-4.5-Air-Q8_0-00001-of-00002.gguf.header"
    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        "f1451f049d87363a5aa99bee10d7db42f024f78e9c9fb1f35914f31a7fecb1ef")


if __name__ == "__main__":
    main()
