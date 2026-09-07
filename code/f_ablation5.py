#!/usr/bin/env python3
"""F. Ablation -- clustering column re-measured under the realistic (c,d) pool,
   so both columns of the paper table come from the same worlds."""
import numpy as np
from harness import ci95
from harness_tfrs import TFRSWorld
from f_ablation import build, cluster, CELLS

SEEDS = [11, 22, 33]
for occ in (1.0, 4.0):
    res = []
    for s in SEEDS:
        w = TFRSWorld(s, m=200, pool_cc=8, pool_dt=12)
        b = build(w, occ)
        res.append({c: (cluster(w, *b[c]), len(b[c][2])) for c in CELLS})
    print("\n  occupancy {:g}   (realistic pool, 8 cc x 12 dt)".format(occ),
          flush=True)
    for c in CELLS:
        m, h, _ = ci95([r[c][0] for r in res])
        n = int(np.mean([r[c][1] for r in res]))
        note = "  <== vacuous: node = account" if c == "plain" else ""
        print("  {:<8} {:>9,} nodes   ARI {:>8.4f} +/- {:.4f}{}".format(
            c, n, m, h, note), flush=True)
print("\ndone", flush=True)
