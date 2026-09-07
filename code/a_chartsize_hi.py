#!/usr/bin/env python3
"""
Paper A, item 5, second pass: chart size where the metric has room to move.

The first pass (a_chartsize.py) found the design rule holds to m = 1200 at
occupancy 1, and ALSO found occupancy 2 pinned near zero on the structured
chart -- consistent with tab:tfrs, which needs occupancy 8-24 before the
structured chart shows any recovery at all. So occupancy 2 could not reveal a
chart-size effect even if one existed. That is the pinned-metric trap the
artefact README warns about, and this pass is the correction: the same sweep
at occupancies 8 and 24, where tab:tfrs shows the attack actually working.

m = 300 at both occupancies is a validation point against tab:tfrs
(0.0086 +/- 0.0019 at 8, 0.0227 +/- 0.0029 at 24).
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from harness import ci95
from harness_tfrs import TFRSWorld

SEEDS = [11, 22, 33, 44, 55]
SIZES = [300, 600, 1000, 1200]
DIMS, REFINE = 64, 3
PAPER = {(300, 8.0): (0.0086, 0.0019), (300, 24.0): (0.0227, 0.0029)}


def attack(w, u, v):
    n = w.Kp
    C = sp.coo_matrix((np.ones(2 * len(u)),
                       (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(n, n)).tocsr()
    C.data = np.log1p(C.data)
    idx = np.nonzero(np.asarray((C != 0).sum(axis=1)).ravel() > 0)[0]
    if len(idx) < w.m + 2:
        return float("nan")
    Cs = C[idx][:, idx]
    U, S, _ = svds(Cs.asfptype(), k=min(DIMS, len(idx) - 1), random_state=w.seed)
    X = U * S
    nn = np.linalg.norm(X, axis=1, keepdims=True)
    nn[nn == 0] = 1
    km = lambda k: KMeans(k, n_init=2, random_state=w.seed)
    lab = km(w.m).fit_predict(X / nn)
    for _ in range(REFINE):
        Hm = sp.coo_matrix((np.ones(len(idx)), (np.arange(len(idx)), lab)),
                           shape=(len(idx), w.m)).tocsr()
        Q = np.asarray((Cs @ Hm).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True)
        nn[nn == 0] = 1
        lab = km(w.m).fit_predict(Q / nn)
    return float(adjusted_rand_score(w.owner[idx], lab))


print("=" * 88, flush=True)
print("PAPER A -- CHART SIZE AT HIGH OCCUPANCY (where the attack has room)",
      flush=True)
print("=" * 88, flush=True)
for occ in (8.0, 24.0):
    print(flush=True)
    print("  OCCUPANCY {:.0f}".format(occ), flush=True)
    print("     {:>6} {:>22} {:>9} {:>11} {:>18} {:>6}".format(
        "m", "ARI (95% CI)", "upper", "floor k=1", "tab:tfrs", "secs"),
        flush=True)
    for m in SIZES:
        vals, fl = [], []
        t0 = time.time()
        for sd in SEEDS:
            w = TFRSWorld(sd, m=m)
            fl.append(float((w.k == 1).mean()))
            _, _, d, c, _, _, _, _ = w.docs_for_occupancy(occ)
            vals.append(attack(w, d, c))
        mu, h, _ = ci95(vals)
        ref = PAPER.get((m, occ))
        reftxt = "{:.4f} +/- {:.4f}".format(*ref) if ref else "---"
        print("     {:>6} {:>12.4f} +/- {:.4f} {:>9.4f} {:>10.1%} {:>18} {:>6.0f}"
              .format(m, mu, h, mu + h, np.mean(fl), reftxt, time.time() - t0),
              flush=True)
print(flush=True)
print("done", flush=True)
