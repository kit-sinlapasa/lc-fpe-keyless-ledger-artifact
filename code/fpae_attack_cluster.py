#!/usr/bin/env python3
"""
FULL CLUSTERING ATTACK on FPAE  --  converts the co-occurrence AUC into an
actual chart-of-accounts RECOVERY RATE.

Adversary pipeline:
  1. observe the aliased double-entry ledger (documents = pairs of codes)
  2. build the sparse alias x alias co-occurrence matrix
  3. spectral embedding (truncated SVD) -> row-normalise
  4. k-means into m clusters
  5. iterative refinement: re-project co-occurrence onto current clusters,
     re-cluster  (co-clustering / EM style)
  6. match clusters to accounts optimally (Hungarian on the confusion matrix)
Metric: fraction of aliases placed in the correct account, plus ARI, plus
recovery restricted to the top-10 accounts.  Null = size-respecting random
assignment put through the same Hungarian matching.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(20260903)


def alloc_hamilton(f, K):
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


def build_ledger(m, K, n_docs, n_types=400):
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1) ** 1.0; wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    k = alloc_hamilton(f, K)
    Kp = int(k.sum()); start = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    t = rng.choice(n_types, size=n_docs, p=wt)
    da, ca = dr[t], cr[t]
    d_al = start[da] + rng.integers(0, k[da])
    c_al = start[ca] + rng.integers(0, k[ca])
    return f, k, Kp, owner, d_al, c_al


def rownorm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def cluster_attack(Kp, m, d_al, c_al, dims=64, refine=3, seed=0):
    C = sp.coo_matrix((np.ones(len(d_al) * 2),
                       (np.concatenate([d_al, c_al]),
                        np.concatenate([c_al, d_al]))),
                      shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)                     # damp heavy hitters
    U, S, _ = svds(C.asfptype(), k=min(dims, Kp - 1))
    E = rownorm(U * S)
    lab = KMeans(m, n_init=4, random_state=seed).fit_predict(E)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(Kp), (np.arange(Kp), lab)),
                          shape=(Kp, m)).tocsr()
        Q = rownorm(np.asarray((C @ H).todense()))
        lab = KMeans(m, n_init=4, random_state=seed).fit_predict(Q)
    return lab


def score(lab, owner, m, f, k):
    conf = np.zeros((m, m))
    np.add.at(conf, (lab, owner), 1.0)
    r, c = linear_sum_assignment(-conf)
    mapping = dict(zip(r, c))
    guess = np.array([mapping[l] for l in lab])
    acc = float((guess == owner).mean())
    top10 = np.argsort(-f)[:10]; mt = np.isin(owner, top10)
    acc10 = float((guess[mt] == owner[mt]).mean())
    return acc, acc10, adjusted_rand_score(owner, lab)


def null_score(owner, m, f, k, Kp):
    lab = np.repeat(np.arange(m), k)
    lab = lab[rng.permutation(Kp)]
    return score(lab, owner, m, f, k)


m, K = 300, 9700
print("=" * 88)
print("FULL CLUSTERING ATTACK  --  chart-of-accounts recovery rate")
print("=" * 88)
print(f"  m = {m} accounts, K = {K} aliases, spectral(64) + k-means + 3 refinements")
print()
print(f"  {'docs':>10} {'lines/alias':>12} | {'recovery':>9} {'null':>7} {'lift':>6} |"
      f" {'top10 rec':>10} {'null':>7} | {'ARI':>7}")
for nd in (5_000, 20_000, 50_000, 100_000, 200_000, 500_000, 2_000_000):
    f, k, Kp, owner, d_al, c_al = build_ledger(m, K, nd)
    lab = cluster_attack(Kp, m, d_al, c_al)
    acc, acc10, ari = score(lab, owner, m, f, k)
    n_acc, n_acc10, n_ari = null_score(owner, m, f, k, Kp)
    occ = 2 * nd / Kp
    print(f"  {nd:>10,} {occ:>12.2f} | {acc:>8.2%} {n_acc:>6.2%} "
          f"{acc/max(n_acc,1e-9):>5.1f}x | {acc10:>9.2%} {n_acc10:>6.2%} | {ari:>7.3f}")

print()
print("=" * 88)
print("  recovery = fraction of aliases mapped to their TRUE account after")
print("  optimal cluster->account matching.  null = size-respecting random")
print("  labelling scored the same way.  ARI: 0 = chance, 1 = perfect.")
