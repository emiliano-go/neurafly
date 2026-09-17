#!/usr/bin/env python3
"""Preprocess the FlyWire v783 proofread connections into a compact runtime
graph for neural-beam-scope.

Reads:  data/proofread_connections_783.feather
Writes: data/flywire_net.bin

Output format (little-endian):
    magic   8 bytes  b"NBSNET01"
    N       u32      neuron count
    GW, GH  u32      layout grid dims
    E       u32      edge count
    per neuron (N records):
        x u16, y u16   grid position
        band u8        audio band 0..9 (tonotopic stripe)
        inhib u8       1 if dominantly inhibitory (GABA/glut) else 0
        thr f32        firing threshold (deterministic hash jitter)
    CSR edges:
        indptr u32[N+1]
        target u32[E]
        weight f32[E]  signed (negative = inhibitory)

Run:  uv run --with pyarrow --with numpy python tools/preprocess.py
"""

import sys
import numpy as np
import pyarrow.feather as feather

SRC = "data/proofread_connections_783.feather"
DST = "data/flywire_net.bin"
TOP_K_OUT = 16          # keep strongest K outgoing edges per neuron
W_SCALE = 0.075         # signed weight per unit of normalized syn_count

def main():
    print(f"reading {SRC} ...", flush=True)
    tbl = feather.read_table(SRC)
    cols = tbl.column_names
    print("columns:", cols, flush=True)

    def col(*names):
        for n in names:
            if n in cols:
                return tbl.column(n).to_numpy()
        raise SystemExit(f"missing column, tried {names}")

    pre = col("pre_root_id", "pre_pt_root_id", "pre")
    post = col("post_root_id", "post_pt_root_id", "post")
    syn = col("syn_count", "syn_count_nt", "count").astype(np.float64)
    gaba = col("gaba_avg").astype(np.float64) if "gaba_avg" in cols else None
    ach = col("ach_avg").astype(np.float64) if "ach_avg" in cols else None
    glut = col("glut_avg").astype(np.float64) if "glut_avg" in cols else None

    # aggregate duplicate (pre,post) pairs split across neuropils
    print(f"{len(pre):,} raw rows; aggregating duplicate pairs ...", flush=True)
    order = np.lexsort((post, pre))
    pre, post, syn = pre[order], post[order], syn[order]
    if gaba is not None:
        gaba, ach, glut = gaba[order], ach[order], glut[order]
    starts = np.flatnonzero(np.r_[True, (pre[1:] != pre[:-1]) | (post[1:] != post[:-1])])
    run_id = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(pre)]))
    pre_u = pre[starts]
    post_u = post[starts]
    syn_u = np.bincount(run_id, weights=syn)
    if gaba is not None:
        # synapse-weighted NT fractions
        gaba_u = np.bincount(run_id, weights=gaba * syn) / np.maximum(syn_u, 1e-9)
        ach_u = np.bincount(run_id, weights=ach * syn) / np.maximum(syn_u, 1e-9)
        glut_u = np.bincount(run_id, weights=glut * syn) / np.maximum(syn_u, 1e-9)
    else:
        gaba_u = ach_u = glut_u = None
    print(f"{len(pre_u):,} unique pairs", flush=True)

    # node indexing
    nodes = np.unique(np.concatenate([pre_u, post_u]))
    N = len(nodes)
    print(f"{N:,} neurons", flush=True)
    pre_i = np.searchsorted(nodes, pre_u).astype(np.int32)
    post_i = np.searchsorted(nodes, post_u).astype(np.int32)

    # edge sign: fly neurochem — ACh excitatory, GABA + glutamate inhibitory
    if gaba_u is not None:
        inhib_edge = (gaba_u + glut_u) > ach_u
    else:
        inhib_edge = np.zeros(len(pre_u), bool)
    w = syn_u.copy()
    cap = np.percentile(w, 99.0)
    w = np.minimum(w, cap) / cap * W_SCALE
    w[inhib_edge] *= -1.6

    # prune: strongest TOP_K_OUT outgoing edges per presynaptic neuron
    print("pruning to top-K outgoing edges ...", flush=True)
    order2 = np.lexsort((-np.abs(w), pre_i))
    p_sorted = pre_i[order2]
    pstarts = np.flatnonzero(np.r_[True, p_sorted[1:] != p_sorted[:-1]])
    rank = np.arange(len(order2)) - np.repeat(pstarts, np.diff(np.r_[pstarts, len(order2)]))
    keep = np.zeros(len(w), bool)
    keep[order2[rank < TOP_K_OUT]] = True
    pre_i, post_i, w = pre_i[keep], post_i[keep], w[keep]
    E = len(w)
    print(f"{E:,} edges kept", flush=True)

    # drop neurons left with no edges at all
    used = np.zeros(N, bool)
    used[pre_i] = True
    used[post_i] = True
    remap = -np.ones(N, dtype=np.int32)
    remap[used] = np.arange(int(used.sum()), dtype=np.int32)
    pre_i = remap[pre_i]
    post_i = remap[post_i]
    nodes = nodes[used]
    N = len(nodes)
    print(f"{N:,} neurons after pruning", flush=True)

    # per-neuron inhibitory flag: sign of summed outgoing weight
    out_sum = np.bincount(pre_i, weights=w, minlength=N)
    inhib_n = (out_sum < 0).astype(np.uint8)

    # deterministic threshold jitter from root id
    h = (nodes % (1 << 31)).astype(np.uint64)
    h ^= h >> 12; h ^= (h << 25) & 0xFFFFFFFFFFFFFFFF; h ^= h >> 27
    u01 = ((h * np.uint64(0x2545F4914F6CDD1D)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    thr = (0.70 + u01 * 0.65).astype(np.float32)

    # layout: raster order sorted by (mean post position unknown) — use
    # root-id hash to scatter, then smooth into a square grid so the map is
    # evenly filled
    gw = int(np.ceil(np.sqrt(N)))
    gh = int(np.ceil(N / gw))
    order3 = np.argsort(h, kind="stable")
    xs = np.empty(N, np.uint16)
    ys = np.empty(N, np.uint16)
    xs[order3] = np.arange(N) % gw
    ys[order3] = np.arange(N) // gw
    band = (xs.astype(np.int64) * 24 // gw).astype(np.uint8)

    # CSR
    indptr = np.zeros(N + 1, np.uint32)
    np.add.at(indptr, pre_i + 1, 1)
    indptr = np.cumsum(indptr).astype(np.uint32)
    sorder = np.argsort(pre_i, kind="stable")
    tgt = post_i[sorder].astype(np.uint32)
    wt = w[sorder].astype(np.float32)

    import struct
    with open(DST, "wb") as f:
        f.write(b"NBSNET01")
        f.write(struct.pack("<IIII", N, gw, gh, E))
        for i in range(N):
            f.write(struct.pack("<HHBBf", int(xs[i]), int(ys[i]), int(band[i]), int(inhib_n[i]), float(thr[i])))
        f.write(indptr.tobytes())
        f.write(tgt.tobytes())
        f.write(wt.tobytes())
    print(f"wrote {DST}: N={N:,} E={E:,} grid={gw}x{gh}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
