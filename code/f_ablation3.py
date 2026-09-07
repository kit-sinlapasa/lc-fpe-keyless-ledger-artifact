#!/usr/bin/env python3
"""
F. Ablation, third pass -- the trivial baseline for row recovery.

Pass 2 left FPAE at 31.7% row recovery, which sounds like a leak.  But an
adversary who ignores the ciphertext entirely and answers "the most frequent
account that can appear in this (c,d) group" already scores well, because the
account distribution is Zipf-like and hubs dominate.  Only the excess over that
constant strategy is attributable to the ciphertext.
"""
import numpy as np
from collections import defaultdict
from harness import ci95
from harness_tfrs import TFRSWorld

SEEDS = [11, 22, 33]

for occ in (1.0, 4.0):
    T, G = [], []
    for s in SEEDS:
        w = TFRSWorld(s, m=200)
        da, ca, d_al, c_al, d_cc, d_dt, c_cc, c_dt = w.docs_for_occupancy(occ)
        acct = np.concatenate([da, ca])
        cd = list(zip(np.concatenate([d_cc, c_cc]).tolist(),
                      np.concatenate([d_dt, c_dt]).tolist()))
        rank = {a: r for r, a in enumerate(np.argsort(-w.f).tolist())}
        g = defaultdict(list)
        for i in range(len(acct)):
            g[cd[i]].append(i)
        triv = sum(sum(1 for i in idxs
                       if int(acct[i]) == min((int(acct[j]) for j in idxs),
                                              key=lambda a: rank[a]))
                   for idxs in g.values()) / len(acct)
        T.append(triv)
        G.append(float((acct == np.argmax(w.f)).mean()))
    tm, th, _ = ci95(T); gm, gh, _ = ci95(G)
    print("  occupancy {:g}: ciphertext-free baseline (best account per c,d "
          "group) = {:.1%} +/- {:.1%}   |   single global hub = {:.1%}"
          .format(occ, tm, th, gm), flush=True)
