#!/usr/bin/env python3
"""Throwaway GGUF metadata + tensor inspector (no external deps).

Used during M1 to learn the exact arch params and tensor names of the Qwen2.5
GGUFs before writing the real C++ loader. Prints scalar metadata, summarizes
large arrays, and lists tensor name/shape/type (collapsing per-layer repeats).
"""
import struct
import sys

# ggml_type -> name (only the ones we expect to meet)
GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0",
             12: "Q4_K", 14: "Q6_K"}

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        return self.f.read(n)

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def i32(self):
        return struct.unpack("<i", self.f.read(4))[0]

    def f32(self):
        return struct.unpack("<f", self.f.read(4))[0]

    def string(self):
        n = self.u64()
        return self.f.read(n).decode("utf-8", "replace")


def read_value(r, vtype):
    # 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 str,9 arr,10 u64,11 i64,12 f64
    if vtype == 4:
        return r.u32()
    if vtype == 5:
        return r.i32()
    if vtype == 6:
        return r.f32()
    if vtype == 7:
        return struct.unpack("<b", r.raw(1))[0]
    if vtype == 8:
        return r.string()
    if vtype == 10:
        return r.u64()
    if vtype == 11:
        return struct.unpack("<q", r.raw(8))[0]
    if vtype == 12:
        return struct.unpack("<d", r.raw(8))[0]
    if vtype in (0, 1):
        return struct.unpack("<b", r.raw(1))[0]
    if vtype in (2, 3):
        return struct.unpack("<h", r.raw(2))[0]
    if vtype == 9:
        elem_t = r.u32()
        count = r.u64()
        # don't materialize giant token arrays; sample first few
        sample = []
        for i in range(count):
            v = read_value(r, elem_t)
            if i < 3:
                sample.append(v)
        return f"<array type={elem_t} count={count} sample={sample}>"
    raise ValueError(f"unknown vtype {vtype}")


def main(path):
    with open(path, "rb") as f:
        r = R(f)
        magic = r.u32()
        assert magic == GGUF_MAGIC, f"bad magic {magic:#x}"
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        print(f"GGUF v{version}  tensors={n_tensors}  kv={n_kv}\n--- metadata ---")
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            val = read_value(r, vtype)
            sval = str(val)
            if len(sval) > 90:
                sval = sval[:90] + "..."
            print(f"{key} = {sval}")

        print("\n--- tensors (layer.0 + non-layer; *.N collapsed) ---")
        seen = set()
        for _ in range(n_tensors):
            name = r.string()
            ndim = r.u32()
            dims = [r.u64() for _ in range(ndim)]
            ttype = r.u32()
            _off = r.u64()
            # collapse blk.<n>.* to blk.N.* so we print each unique tensor once
            parts = name.split(".")
            key = name
            if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
                if parts[1] != "0":
                    continue
                key = name
            if key in seen:
                continue
            seen.add(key)
            tname = GGML_TYPE.get(ttype, f"type{ttype}")
            print(f"{name:35s} {tname:6s} {dims}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/home/jose/models/gguf/qwen2.5-1.5b-instruct-q4_0.gguf")
