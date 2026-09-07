#!/usr/bin/env python3
"""
F. Ablation, final table.

Pass 3 exposed a generator artifact that invalidates every earlier experiment in
which the adversary is told the row's cost centre and document type.  The
default synthetic world draws each account's five cost centres from a pool of
N_CC = 100, so a (cost centre, document type) cell holds 1.9 distinct accounts
on average and 87.6% of rows sit in a cell where one account dominates.  Knowing
(c,d) therefore almost names the account, whatever the cipher does.

A real chart is the other way round: a handful of departments and document types
shared by every account.  We re-run under a realistic pool -- 8 cost centres and
12 document types -- and report the CIPHERTEXT-FREE BASELINE alongside every
number, so the reader can see what the ciphertext actually contributes.

The clustering attack that carries the paper's design rule never reads (c,d)
(it is built on alias co-occurrence alone), so it is untouched by this artifact.
"""
import numpy as np
from collections import defaultdict
from harness import ci95
from harness_tfrs import TFRSWorld
from f_ablation import build, CELLS

SEEDS = [11, 22, 33]
POOLS = (("synthetic  (100 cc x 10 dt)", 100, 10),
         ("realistic  (8 cc x 12 dt)",     8, 12))


def measure(w, occ):
    draw = w.docs_for_occupancy(occ)
    w._last = draw
    w.docs_for_occupancy = lambda o, _d=draw: _d
    b = build(w, occ)
    da, ca, d_al, c_al, d_cc, d_dt, c_cc, c_dt = draw
    acct = np.concatenate([da, ca])
    cd = list(zip(np.concatenate([d_cc, c_cc]).tolist(),
                  np.concatenate([d_dt, c_dt]).tolist()))
    rank = {a: r for r, a in enumerate(np.argsort(-w.f).tolist())}
    grp = defaultdict(list)
    for i, k in enumerate(cd):
        grp[k].append(i)

    free = sum(max(np.bincount([int(acct[i]) for i in idxs]))
               for idxs in grp.values()) / len(acct)
    purity = free
    out = {"_free": free, "_purity": purity,
           "_ncand": float(np.mean([len({int(acct[i]) for i in idxs})
                                    for idxs in grp.values()]))}
    for c in CELLS:
        u, v, owner = b[c]
        node = np.concatenate([u, v])
        correct = 0
        for idxs in grp.values():
            cnt = defaultdict(int)
            for i in idxs:
                cnt[int(node[i])] += 1
            order = sorted(cnt, key=lambda z: (-cnt[z], z))
            cands = sorted({int(acct[i]) for i in idxs}, key=lambda a: rank[a])
            gss = {t: cands[min(r, len(cands) - 1)] for r, t in enumerate(order)}
            correct += sum(1 for i in idxs if gss[int(node[i])] == int(acct[i]))
        out[c] = correct / len(node)
    return out


print("=" * 94, flush=True)
print("F. ABLATION -- row recovery by an adversary who knows (c,d)", flush=True)
print("=" * 94, flush=True)
print("  structured TFRS chart, m=200, three seeds, generative aux, "
      "occupancy 1", flush=True)

for label, pc, pd in POOLS:
    res = [measure(TFRSWorld(s, m=200, pool_cc=pc, pool_dt=pd), 1.0)
           for s in SEEDS]
    fm, fh, _ = ci95([r["_free"] for r in res])
    print(flush=True)
    print("  {}   accounts per (c,d) cell: {:.1f}".format(
        label, np.mean([r["_ncand"] for r in res])), flush=True)
    print("  {:<32} {:>22}".format("adversary", "row recovery"), flush=True)
    print("  {:<32} {:>13.1%} +/- {:.1%}   <== reads no ciphertext".format(
        "guess the cell's best account", fm, fh), flush=True)
    for c in CELLS:
        m, h, _ = ci95([r[c] for r in res])
        gain = m - fm
        print("  {:<32} {:>13.1%} +/- {:.1%}   {:+.1%} vs baseline".format(
            c, m, h, gain), flush=True)

print(flush=True)
print("done", flush=True)
