#!/usr/bin/env python3
"""
extract_embeddings.py — recover YAMNet embeddings from a Milvus Lite `milvus_demo.db`.

Milvus Lite stores collection data in SQLite. Each stored vector is a
length-delimited protobuf float array:

    0x0A  <varint: byte-length = dim*4>  <float32 little-endian * dim>

Two gotchas that this script handles:
  * `con.text_factory = bytes` — otherwise SQLite tries to decode BLOBs as UTF-8 and corrupts them.
  * length is validated to be a multiple of 4 before interpreting the chunk as float32.

Usage:
    python extract_embeddings.py --db milvus_demo.db --out recovered_embeddings
Outputs one `<collection>_<dim>.npy` per (table, vector-dimension) found.
"""
import argparse, os, sqlite3
from collections import defaultdict
import numpy as np


def read_varint(buf, i):
    shift = val = 0
    while True:
        byte = buf[i]; i += 1
        val |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return val, i
        shift += 7


def parse_float_vectors(blob):
    """Yield float32 arrays from concatenated 0x0A length-delimited protobuf fields."""
    out, i, n = [], 0, len(blob)
    while i < n:
        if blob[i] != 0x0A:            # only field-1, wire-type-2 (length-delimited)
            i += 1
            continue
        i += 1
        length, i = read_varint(blob, i)
        if length == 0 or length % 4 != 0 or i + length > n:
            continue
        out.append(np.frombuffer(blob[i:i + length], dtype="<f4"))
        i += length
    return out


def blob_columns(con, table):
    cur = con.execute(f'SELECT * FROM "{table}" LIMIT 50')
    cols = [d[0] for d in cur.description]
    found = set()
    for row in cur.fetchall():
        for c, v in zip(cols, row):
            if isinstance(v, (bytes, bytearray)) and len(v) >= 8 and v[:1] == b"\x0a":
                found.add(c)
    return cols, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="milvus_demo.db")
    ap.add_argument("--out", default="recovered_embeddings")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"database not found: {args.db}")
    os.makedirs(args.out, exist_ok=True)

    con = sqlite3.connect(args.db)
    con.text_factory = bytes  # critical: keep BLOBs as raw bytes

    tables = [t[0].decode() if isinstance(t[0], bytes) else t[0]
              for t in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]

    total = 0
    for table in tables:
        try:
            _, bcols = blob_columns(con, table)
        except Exception:
            continue
        for col in bcols:
            by_dim = defaultdict(list)
            for (blob,) in con.execute(f'SELECT "{col}" FROM "{table}"'):
                if isinstance(blob, (bytes, bytearray)):
                    for v in parse_float_vectors(bytes(blob)):
                        by_dim[len(v)].append(v)
            for dim, arrs in by_dim.items():
                name = table.strip().replace(" ", "_")
                path = os.path.join(args.out, f"{name}_{dim}.npy")
                arr = np.vstack(arrs).astype("float32")
                np.save(path, arr)
                total += 1
                print(f"  saved {path}  shape={arr.shape}")

    print(f"done: wrote {total} array file(s) to {args.out}/")
    if total == 0:
        print("No float vectors found. Check the DB path, or pass the right table/column.")


if __name__ == "__main__":
    main()
