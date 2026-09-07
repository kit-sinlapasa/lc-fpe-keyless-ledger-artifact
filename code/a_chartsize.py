#!/usr/bin/env python3
"""
Paper A, item 5: measuring chart size instead of extrapolating it.

The paper's Limitations said:

    "Chart size beyond m ~ 400 untested. Recovery grows monotonically with the
     size of the chart at fixed K; a linear extrapolation puts the budget at
     roughly a thousand accounts, but we have not measured there."

That is the last extrapolation left in Paper A, and this session has spent its
time removing exactly this kind of claim. So we measure it.

TWO OCCUPANCIES, deliberately. At occupancy 1 the adjusted Rand index is near
zero in every configuration tested so far, so a sweep run only there risks a
null result for want of room to move -- the failure that task D and task G both
hit. Occupancy 1 answers the question the design rule asks ("does the rule
still hold at m = 1000?"). Occupancy 2, where the rule is already known to
fail, is where chart size can actually show its effect.

THE MECHANISM is worth reporting alongside the outcome. The budget is
K = alpha - m, so growing the chart at fixed field width shrinks the budget
AND divides it among more accounts. Accounts held at the floor k(a)=1 are the
ones an adversary can pick out, because a single alias appears in the
co-occurrence graph exactly as the account does. We report the floor fraction
next to the recovery figure so the reader can see which one drives the other.

Attack is the STRONG configuration (full k-means, 64 dimensions, three
refinement rounds), because the paper judges safety by the upper end of the
interval of the strongest adversary it has.
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
SIZES = [200, 300, 400, 600, 800, 1000, 1200]
DIMS, REFINE = 64, 3


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
print("PAPER A -- CHART SIZE, MEASURED", flush=True)
print("=" * 88, flush=True)
print("  alpha = 10^4, K = alpha - m, structured TFRS chart, strong attack, "
      "{} seeds".format(len(SEEDS)), flush=True)

# ------------------------------------------------------------- structure first
print(flush=True)
print("  1. WHAT GROWING THE CHART DOES TO THE ALLOCATION", flush=True)
print("     {:>6} {:>8} {:>9} {:>12} {:>16}".format(
    "m", "K'", "mean k", "floor k=1", "Delta"), flush=True)
struct = {}
for m in SIZES:
    fl, mk, dl = [], [], []
    for sd in SEEDS:
        w = TFRSWorld(sd, m=m)
        fl.append(float((w.k == 1).mean()))
        mk.append(float(w.k.mean()))
        dl.append(w.delta())
    struct[m] = (np.mean(fl), np.mean(mk), np.mean(dl))
    w = TFRSWorld(11, m=m)
    d_m, d_h, _ = ci95(dl)
    print("     {:>6} {:>8,} {:>9.1f} {:>11.1%} {:>10.4f} +/- {:.4f}".format(
        m, w.Kp, np.mean(mk), np.mean(fl), d_m, d_h), flush=True)

# ------------------------------------------------------------------ the sweep
for occ in (1.0, 2.0):
    print(flush=True)
    print("  2. RECOVERY AT OCCUPANCY {:.0f}{}".format(
        occ, "  (the design rule)" if occ == 1.0
        else "  (rule already known to fail; room to move)"), flush=True)
    print("     {:>6} {:>22} {:>9} {:>11} {:>8}".format(
        "m", "ARI (95% CI)", "upper", "floor k=1", "secs"), flush=True)
    for m in SIZES:
        vals = []
        t0 = time.time()
        for sd in SEEDS:
            w = TFRSWorld(sd, m=m)
            _, _, d, c, _, _, _, _ = w.docs_for_occupancy(occ)
            vals.append(attack(w, d, c))
        mu, h, _ = ci95(vals)
        up = mu + h
        flag = "" if up < 0.05 else ("  <== OVER 0.05" if occ == 1.0 else "")
        print("     {:>6} {:>12.4f} +/- {:.4f} {:>9.4f} {:>10.1%} {:>8.0f}{}"
              .format(m, mu, h, up, struct[m][0], time.time() - t0, flag),
              flush=True)

print(flush=True)
print("done", flush=True)
