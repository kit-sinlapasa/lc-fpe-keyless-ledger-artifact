#!/usr/bin/env python3
"""
Refined scan in the critical low-occupancy band, 3 seeds per point.

Round 1 showed the clustering attack is far stronger than the AUC
distinguisher implied, so the R/K' < 3 rule derived from AUC is too loose.
Also: at occupancy ~1 'recovery' read 13.6% while ARI read 0.003 -- the
recovery metric is inflated by cluster-SIZE matching against a Zipf chart,
so ARI is the trustworthy chance-corrected metric.  Report both.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def alloc_hamilton(f, K, rng):
    q = f * K
    base = np.maximum(np.floor(q).astype(np.int64), 1)
    short = K - base.sum()
    if short > 0:
        base[np.argsort(-(q - np.floor(q)))[:short]] += 1
    elif short < 0:
        order = np.argsort(q - base); i = 0
        while short < 0 and i < len(order):
            j = order[i]
            if base[j] > 1:
                base[j] -= 1; short += 1
            i += 1
    return base

def run(m, K, n_docs, seed, n_types=400, dims=32, refine=2):
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1); wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    k = alloc_hamilton(f, K, rng)
    Kp = int(k.sum()); start = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    t = rng.choice(n_types, size=n_docs, p=wt)
    da, ca = dr[t], cr[t]
    d_al = start[da] + rng.integers(0, k[da])
    c_al = start[ca] + rng.integers(0, k[ca])

    C = sp.coo_matrix((np.ones(2 * n_docs),
                       (np.concatenate([d_al, c_al]),
                        np.concatenate([c_al, d_al]))), shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)
    U, S, _ = svds(C.asfptype(), k=min(dims, Kp - 1))
    X = U * S
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    lab = KMeans(m, n_init=2, random_state=seed).fit_predict(X / n)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(Kp), (np.arange(Kp), lab)),
                          shape=(Kp, m)).tocsr()
        Q = np.asarray((C @ H).todense())
        n = np.linalg.norm(Q, axis=1, keepdims=True); n[n == 0] = 1
        lab = KMeans(m, n_init=2, random_state=seed).fit_predict(Q / n)

    conf = np.zeros((m, m)); np.add.at(conf, (lab, owner), 1.0)
    r, c = linear_sum_assignment(-conf); mp = dict(zip(r, c))
    guess = np.array([mp[l] for l in lab])
    acc = float((guess == owner).mean())
    ari = adjusted_rand_score(owner, lab)
    nl = np.repeat(np.arange(m), k)[rng.permutation(Kp)]
    cn = np.zeros((m, m)); np.add.at(cn, (nl, owner), 1.0)
    r2, c2 = linear_sum_assignment(-cn); mp2 = dict(zip(r2, c2))
    nacc = float((np.array([mp2[l] for l in nl]) == owner).mean())
    return 2 * n_docs / Kp, acc, nacc, ari

m, K = 300, 9700
print("=" * 80)
print("CRITICAL BAND SCAN  --  3 seeds per point")
print("=" * 80)
print(f"  {'docs':>9} {'occupancy':>10} | {'recovery':>9} {'null':>7} | "
      f"{'ARI':>7} {'ARI sd':>7}  regime")
for nd in (2_500, 5_000, 8_000, 12_000, 18_000, 26_000):
    res = [run(m, K, nd, s) for s in (11, 22)]
    occ = res[0][0]
    acc = np.mean([r[1] for r in res]); nac = np.mean([r[2] for r in res])
    ari = np.mean([r[3] for r in res]); sd = np.std([r[3] for r in res])
    reg = ("safe" if ari < 0.05 else "leaking" if ari < 0.20
           else "SERIOUS" if ari < 0.50 else "BROKEN")
    print(f"  {nd:>9,} {occ:>10.2f} | {acc:>8.2%} {nac:>6.2%} | "
          f"{ari:>7.3f} {sd:>7.3f}  {reg}")
