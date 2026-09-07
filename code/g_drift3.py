#!/usr/bin/env python3
"""
G, third pass -- resolve the trend, and use an instrument that can see it.

Pass 2 showed that pass 1's "statistical distance and recovery are decoupled"
was a floor artifact: at occupancy 1 ARI is near zero whatever we do, so there
was no room to degrade.  At occupancy 16 ARI does rise with drift, monotonically
across five settings, but only from 0.0282 to 0.0343 -- and at three seeds the
end points' intervals still overlap.  This pass runs seven seeds to say whether
that trend is real.

Pass 2's frequency instrument was useless: scoring the ten most frequent aliases
gives a statistic with a 38-57 point interval.  Delta is a property of the whole
alias distribution, so we measure it with one:

    SPEARMAN( rows landing on an alias , frequency of the account owning it )

over all K' aliases.  A correct allocation makes every alias equally likely, so
the correlation is zero; drift makes over-allocated accounts' aliases sparse and
under-allocated ones' aliases crowded, so it moves away from zero.  Nine
thousand aliases instead of ten give it a usable variance.
"""
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from harness import ci95, alloc_hamilton
from harness_tfrs import TFRSWorld

SEEDS = [11, 22, 33, 44, 55, 66, 77]
M, K, OCC = 200, 9800, 16.0


def reallocate(w, fhat):
    k = alloc_hamilton(fhat, K)
    w.k = k
    w.Kp = int(k.sum())
    w.start = np.concatenate([[0], np.cumsum(k)[:-1]])
    w.owner = np.repeat(np.arange(w.m), k)
    return w


def drifted(w, d):
    g = w.f.copy()
    g[w.by_cat["expense"]] *= (1.0 + d)
    return g / g.sum()


def measure(w, occ, dims=64, refine=3):
    _, _, d_al, c_al, _, _, _, _ = w.docs_for_occupancy(occ)
    n = w.Kp
    cnt = np.bincount(np.concatenate([d_al, c_al]), minlength=n)
    rho = float(spearmanr(cnt, w.f[w.owner]).statistic)

    C = sp.coo_matrix((np.ones(2 * len(d_al)),
                       (np.concatenate([d_al, c_al]),
                        np.concatenate([c_al, d_al]))), shape=(n, n)).tocsr()
    C.data = np.log1p(C.data)
    idx = np.nonzero(np.asarray((C != 0).sum(axis=1)).ravel() > 0)[0]
    Cs = C[idx][:, idx]
    U, S, _ = svds(Cs.asfptype(), k=min(dims, len(idx) - 1), random_state=w.seed)
    X = U * S
    nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    KM = lambda kk: KMeans(kk, n_init=2, random_state=w.seed)
    lab = KM(w.m).fit_predict(X / nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(len(idx)), (np.arange(len(idx)), lab)),
                          shape=(len(idx), w.m)).tocsr()
        Q = np.asarray((Cs @ H).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KM(w.m).fit_predict(Q / nn)
    return float(adjusted_rand_score(w.owner[idx], lab)), rho


print("=" * 96, flush=True)
print("G pass 3 -- seven seeds, and an instrument with usable variance",
      flush=True)
print("=" * 96, flush=True)
print("  structured TFRS chart, m={}, K={}, occupancy {:g}, seven seeds"
      .format(M, K, OCC), flush=True)
print(flush=True)
print("  {:>8} {:>12} {:>24} {:>26}".format(
    "delta", "TV(f,fhat)", "clustering ARI", "Spearman(count, f(owner))"),
    flush=True)

rows = []
for d in (0.0, 0.25, 0.5, 1.0, 2.0):
    TV, AR, RH = [], [], []
    for s in SEEDS:
        w = TFRSWorld(s, m=M, K=K)
        fhat = w.f.copy() if d == 0.0 else drifted(w, d)
        TV.append(0.5 * float(np.abs(w.f - fhat).sum()))
        reallocate(w, fhat)
        a, r = measure(w, OCC)
        AR.append(a); RH.append(r)
    tm, _, _ = ci95(TV); am, ah, _ = ci95(AR); rm, rh, _ = ci95(RH)
    rows.append((tm, am, ah, rm, rh))
    print("  {:>8.2f} {:>12.4f} {:>15.4f} +/- {:.4f} {:>17.4f} +/- {:.4f}"
          .format(d, tm, am, ah, rm, rh), flush=True)

print(flush=True)
lo, hi = rows[0], rows[-1]
sep = (lo[1] + lo[2]) < (hi[1] - hi[2])
print("  ARI separated end to end at 95%: {}   ({:.4f}+{:.4f} vs {:.4f}-{:.4f})"
      .format("YES" if sep else "NO", lo[1], lo[2], hi[1], hi[2]), flush=True)
sepr = abs(lo[3]) + lo[4] < abs(hi[3]) - hi[4]
print("  Spearman separated end to end at 95%: {}".format(
    "YES" if sepr else "NO"), flush=True)
print(flush=True)
print("done", flush=True)
