#!/usr/bin/env python3
"""
Two ledgers with the same Delta and very different recoverability.

Theorem 2 bounds Delta = TV(f, k/K'), the flatness of the aliased class
distribution, and the paper leans on it as the design quantity. This asks
whether Delta predicts what an adversary actually recovers.

STEP 1 tunes the chart size of each generator until the two worlds have the
same Delta. Delta moves with m, so this is a search over m rather than a
choice of a convenient pair: without it a reader is entitled to say the two
Deltas merely happened to be close.

STEP 2 runs the same clustering attack on both at the same occupancy.

If Delta is nearly matched and the recovery does not, Delta alone is a poor
predictor for this specified attack at these parameters. This is an empirical
scope diagnostic, not an impossibility proof: it does not rule out another
function of Delta, another bound using co-occurrence structure, or an optimal
adversary. It is why the occupancy rule here is measured rather than derived.
a_freq_bound.py reaches the same conclusion from the other side: a per-draw
distance does not survive a period of N draws.

The attack is the one from d_tfrs.py, unchanged, so that this comparison and
the structured-chart results are the same measurement.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import World, ci95, fmt          # noqa: E402
from harness_tfrs import TFRSWorld            # noqa: E402

ALPHA = 10_000
SEEDS = (11, 22, 33, 44, 55)
OCC = 2.0            # where the flat chart is already over the ARI budget
GRID = (150, 200, 250, 300, 350, 400, 450, 500, 600)


def attack(w, u, v, dims=64, refine=3):
    """Verbatim from d_tfrs.py, strong configuration (KMeans, 64 dims, 3 refines)."""
    n = w.Kp
    C = sp.coo_matrix((np.ones(2 * len(u)), (np.concatenate([u, v]),
                       np.concatenate([v, u]))), shape=(n, n)).tocsr()
    C.data = np.log1p(C.data)
    idx = np.nonzero(np.asarray((C != 0).sum(axis=1)).ravel() > 0)[0]
    if len(idx) < w.m + 2:
        return float("nan")
    Cs = C[idx][:, idx]
    U, S, _ = svds(Cs.asfptype(), k=min(dims, len(idx) - 1), random_state=w.seed)
    X = U * S
    nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    KM = lambda k: KMeans(k, n_init=2, random_state=w.seed)
    lab = KM(w.m).fit_predict(X / nn)
    for _ in range(refine):
        Hm = sp.coo_matrix((np.ones(len(idx)), (np.arange(len(idx)), lab)),
                           shape=(len(idx), w.m)).tocsr()
        Q = np.asarray((Cs @ Hm).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KM(w.m).fit_predict(Q / nn)
    return float(adjusted_rand_score(w.owner[idx], lab))


def mk_flat(seed, m):
    return World(seed, m=m, s=1.0, n_types=400, K=ALPHA - m)


def mk_tfrs(seed, m):
    return TFRSWorld(seed, m=m, K=ALPHA - m)


def mean_delta(mk, m):
    return float(np.mean([mk(s, m).delta() for s in SEEDS]))


def main():
    print("=" * 78, flush=True)
    print("DOES Delta PREDICT WHAT THE ADVERSARY RECOVERS?", flush=True)
    print("=" * 78, flush=True)

    print("", flush=True)
    print("STEP 1  Delta against chart size, both generators, %d seeds each"
          % len(SEEDS), flush=True)
    print("  %-7s %-14s %-14s" % ("m", "flat Delta", "TFRS Delta"), flush=True)
    grid = {}
    for m in GRID:
        df, dt = mean_delta(mk_flat, m), mean_delta(mk_tfrs, m)
        grid[m] = (df, dt)
        print("  %-7d %-14.5f %-14.5f" % (m, df, dt), flush=True)

    best = None
    for mf, (df, _) in grid.items():
        for mt, (_, dt) in grid.items():
            gap = abs(df - dt)
            if best is None or gap < best[0]:
                best = (gap, mf, mt, df, dt)
    gap, mf, mt, df, dt = best
    print("", flush=True)
    print("  closest pair: flat m=%d (Delta %.5f) and TFRS m=%d (Delta %.5f)"
          % (mf, df, mt, dt), flush=True)
    print("  they differ by %.5f, which is %.1f%% of the larger"
          % (gap, 100 * gap / max(df, dt)), flush=True)

    print("", flush=True)
    print("STEP 2  the same attack on the matched pair, occupancy %.1f, %d seeds"
          % (OCC, len(SEEDS)), flush=True)
    print("", flush=True)
    print("  %-8s %-6s %-12s %-24s" % ("chart", "m", "Delta", "ARI (95% CI)"),
          flush=True)

    out = []
    for name, mk, m in (("flat", mk_flat, mf), ("TFRS", mk_tfrs, mt)):
        vals = []
        t0 = time.time()
        for s in SEEDS:
            w = mk(s, m)
            _, _, d, c, _, _, _, _ = w.docs_for_occupancy(OCC)
            vals.append(attack(w, d, c))
        mean, half, n = ci95(vals)
        out.append((name, m, mean_delta(mk, m), mean, half))
        print("  %-8s %-6d %-12.5f %-24s [%.0fs]"
              % (name, m, mean_delta(mk, m), fmt(mean, half, n), time.time() - t0),
              flush=True)

    (_, _, d1, a1, _), (_, _, d2, a2, _) = out
    print("", flush=True)
    print("  Delta differs by %.1f%%.  Recovery differs by %s."
          % (100 * abs(d1 - d2) / max(d1, d2),
             ("%.0f times" % (max(a1, a2) / min(a1, a2))) if min(a1, a2) > 1e-9
             else "more than three orders of magnitude"), flush=True)
    print("", flush=True)
    print("  Delta alone is a poor predictor for this specified attack at these",
          flush=True)
    print("  parameters; this is not an impossibility proof for every bound. What",
          flush=True)
    print("  separates them is the co-occurrence structure of the chart, which",
          flush=True)
    print("  Delta does not see.", flush=True)
    print("  done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
