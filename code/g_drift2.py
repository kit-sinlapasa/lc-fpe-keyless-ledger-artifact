#!/usr/bin/env python3
"""
G, second pass -- the two controls the first pass needs.

Pass 1 found Delta rising from 0.0025 to 0.2223 while clustering ARI stayed
inside noise of zero.  Taken at face value that says statistical distance and
recovery are decoupled.  But ARI at occupancy 1 is near zero for every
configuration we have ever measured, so there is no room for it to degrade:
that reading could be a floor artifact, the same failure that made the TFRS
chart look 100x safer in task D.

CONTROL 1: repeat the drift sweep at occupancy 16, where the structured chart
gives a clearly non-zero ARI (0.021) and degradation would be visible.

CONTROL 2: measure the attack the theorem is actually about.  Delta is the
statistical distance between the aliased distribution and uniform, which is
exactly what a FREQUENCY matcher exploits -- clustering exploits co-occurrence
instead and has no particular reason to notice Delta.  We score the ten
most frequent aliases against the ten most frequent accounts, and report the
chance level alongside, because hub accounts own most aliases and would win
that test without any leakage at all.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from harness import ci95, alloc_hamilton
from harness_tfrs import TFRSWorld

SEEDS = [11, 22, 33]
M, K = 200, 9800
OCC = 16.0


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
    al = np.concatenate([d_al, c_al])

    # --- frequency attack, with its chance level
    cnt = np.bincount(al, minlength=n)
    top_acct = set(np.argsort(-w.f)[:10].tolist())
    hit = sum(1 for x in np.argsort(-cnt)[:10] if w.owner[x] in top_acct) / 10
    chance = float(np.mean([w.owner[x] in top_acct for x in range(n)]))

    # --- clustering
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
    ari = float(adjusted_rand_score(w.owner[idx], lab))
    return ari, hit, chance


print("=" * 100, flush=True)
print("G pass 2 -- does drift move any attack, at an occupancy where it could?",
      flush=True)
print("=" * 100, flush=True)
print("  structured TFRS chart, m={}, K={}, occupancy {:g}, three seeds"
      .format(M, K, OCC), flush=True)
print(flush=True)
print("  {:>8} {:>12} {:>22} {:>18} {:>10}".format(
    "delta", "TV(f,fhat)", "clustering ARI", "freq top-10", "chance"),
    flush=True)

for d in (0.0, 0.25, 0.5, 1.0, 2.0):
    TV, AR, HI, CH = [], [], [], []
    for s in SEEDS:
        w = TFRSWorld(s, m=M, K=K)
        fhat = w.f.copy() if d == 0.0 else drifted(w, d)
        TV.append(0.5 * float(np.abs(w.f - fhat).sum()))
        reallocate(w, fhat)
        a, h, c = measure(w, OCC)
        AR.append(a); HI.append(h); CH.append(c)
    tm, _, _ = ci95(TV); am, ah, _ = ci95(AR)
    hm, hh, _ = ci95(HI); cm, _, _ = ci95(CH)
    print("  {:>8.2f} {:>12.4f} {:>13.4f} +/- {:.4f} {:>10.0%} +/- {:.0%} "
          "{:>10.0%}".format(d, tm, am, ah, hm, hh, cm), flush=True)

print(flush=True)
print("done", flush=True)
