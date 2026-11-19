"""Independent prepared-input and legacy-dot equivalence check.

This calls the candidate DSO's new preparation/dot and its unchanged legacy
compact dot. It uses synthetic compact blocks (a superset of native codebook
combinations), not a model. No performance measurement is made.
"""
import argparse
import ctypes as C
import hashlib
import json
import struct
from pathlib import Path
import numpy as np


DOT = C.CFUNCTYPE(None, C.c_int, C.POINTER(C.c_float), C.c_size_t, C.c_void_p,
                 C.c_size_t, C.c_void_p, C.c_size_t, C.c_int)
TYPES = [(18, "IQ3_XXS", "xeon_iq3_compact_dot"),
         (22, "IQ2_S", "xeon_iq2s_compact_dot"),
         (21, "IQ3_S", "xeon_iq3s_compact_dot")]


def input_case(name, n, rng):
    a = rng.normal(0, .1, n).astype(np.float32)
    if name == "zeros":
        a[:] = 0
    elif name == "tied_extrema":
        for b in range(n//256):
            a[b*256:(b+1)*256] = np.clip(a[b*256:(b+1)*256], -.5, .5)
            a[b*256] = -1 if b % 2 else 1
            a[b*256+255] = -a[b*256]
    elif name == "alternating":
        a[::2], a[1::2] = np.float32(127), np.float32(-128)
    elif name == "sparse":
        a[:] = 0
        a[::137] = np.float32(-.375)
    elif name == "small":
        a *= np.float32(1e-19)
    elif name == "large":
        a *= np.float32(1e20)
    return a


def compact_weights(kind, blocks, rng):
    buf = bytearray(blocks*144)
    for b in range(blocks):
        if kind == "IQ2_S":
            lo = rng.integers(0, 3, 128, dtype=np.uint8) | (rng.integers(0, 2, 128, dtype=np.uint8) << 3)
            hi = rng.integers(0, 3, 128, dtype=np.uint8) | (rng.integers(0, 2, 128, dtype=np.uint8) << 3)
            codes = lo | (hi << 4)
        else:
            codes = rng.integers(0, 256, 128, dtype=np.uint8)
        off = b*144
        buf[off:off+128] = codes.tobytes()
        d = float(np.float16(rng.uniform(-.125, .125)))
        struct.pack_into("<f", buf, off+128, d)
        if kind == "IQ3_XXS":
            scales = (2*rng.integers(0,16,8,dtype=np.uint8)+1).tobytes()
        elif kind == "IQ2_S":
            scales = rng.integers(0,256,8,dtype=np.uint8).tobytes()
        else:
            scales = rng.integers(0,256,4,dtype=np.uint8).tobytes() + bytes(4)
        buf[off+132:off+140] = scales
    return C.create_string_buffer(bytes(buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = args.libdir/"libggml-base.so.0.28.0"
    cpu_path = args.libdir/"libggml-cpu.so.0.28.0"
    C.CDLL(str(base.resolve()), mode=C.RTLD_GLOBAL)
    cpu = C.CDLL(str(cpu_path.resolve()), mode=C.RTLD_GLOBAL)
    cpu.ggml_cpu_init()
    quant = cpu.quantize_row_q8_K
    quant.argtypes = [C.c_void_p,C.c_void_p,C.c_int64]
    quant.restype = None
    prep = cpu.xeon_iq_input_from_float
    prep.argtypes = [C.c_void_p,C.c_void_p,C.c_int64]
    prep.restype = None
    get_dot = cpu.xeon_iq_input_dot
    get_dot.argtypes, get_dot.restype = [C.c_int], C.c_void_p
    rng = np.random.default_rng(621811)
    records = []
    total = prepared_blocks = 0
    digest_old, digest_new = hashlib.sha256(), hashlib.sha256()
    for type_id, name, symbol in TYPES:
        old_dot = getattr(cpu,symbol)
        old_dot.argtypes = DOT._argtypes_
        old_dot.restype = None
        new_dot = DOT(get_dot(type_id))
        count = 0
        for blocks in [1,2,9,20]:
            n = blocks*256
            for pattern in ["zeros","normal","tied_extrema","alternating","sparse","small","large"]:
                x = input_case(pattern,n,rng)
                original, folded = C.create_string_buffer(blocks*292), C.create_string_buffer(blocks*400)
                quant(x.ctypes.data, original, n)
                prep(x.ctypes.data, folded, n)
                qbytes, fbytes = original.raw, folded.raw
                for b in range(blocks):
                    q = np.frombuffer(qbytes,dtype=np.int8,count=256,offset=b*292+4)
                    assert int(q.min()) >= -127
                    assert qbytes[b*292:b*292+4] == fbytes[b*400+384:b*400+388]
                    for g in range(8):
                        lo, hi = q[g*32:g*32+16], q[g*32+16:g*32+32]
                        signs = ((lo<0).astype(np.uint8)<<3) | ((hi<0).astype(np.uint8)<<7)
                        start = b*400+g*48
                        assert fbytes[start:start+16] == signs.tobytes()
                        assert fbytes[start+16:start+32] == np.abs(lo.astype(np.int16)).astype(np.uint8).tobytes()
                        assert fbytes[start+32:start+48] == np.abs(hi.astype(np.int16)).astype(np.uint8).tobytes()
                    prepared_blocks += 1
                for rep in range(16):
                    w = compact_weights(name,blocks,rng)
                    old, new = C.c_float(), C.c_float()
                    old_dot(n,C.byref(old),0,C.addressof(w),0,C.addressof(original),0,1)
                    new_dot(n,C.byref(new),0,C.addressof(w),0,C.addressof(folded),0,1)
                    a, b = C.string_at(C.byref(old),4), C.string_at(C.byref(new),4)
                    if a != b:
                        raise AssertionError((name,blocks,pattern,rep,old.value,new.value,a.hex(),b.hex()))
                    assert np.isfinite(old.value) and np.isfinite(new.value)
                    digest_old.update(a)
                    digest_new.update(b)
                    total += 1
                    count += 1
        records.append({"type":name,"legacy_new_dot_byte_equal_cases":count})
    result = {
        "status":"pass",
        "scope":"Independent prepared-field check and legacy compact dot vs new dot; no timing; no complete model",
        "CPU_DSO_sha256":hashlib.sha256(cpu_path.read_bytes()).hexdigest(),
        "base_DSO_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
        "seed":621811,
        "prepared_blocks_checked":prepared_blocks,
        "dot_outputs_byte_equal":total,
        "legacy_output_sha256":digest_old.hexdigest(),
        "new_output_sha256":digest_new.hexdigest(),
        "per_type":records,
        "input_patterns":["zero","normal","equal positive/negative extrema","alternating extrema","sparse","small finite","large finite"],
        "limitations":["Uses unchanged legacy compact functions in the candidate DSO","Synthetic compact blocks can exceed native codebook combinations","Prequantized minus128 fallback is not exercised by this direct F32 test","Does not establish latest runtime speed or model quality"],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
