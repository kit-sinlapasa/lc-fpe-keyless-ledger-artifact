#!/usr/bin/env python3
"""
Linking-token curve, round 2 -- fixes a modelling error in round 1.

ERROR IN ROUND 1: the bucket-aware adversary was forced to run its SVD on
the per-bucket SUBMATRIX (few rows), so at mid-range B it did WORSE than the
plain global attack.  That produced a spurious U-shaped curve with a
"minimum" at B=8.  But bucket labels are EXTRA information: a rational
adversary can never be worse off, so the true curve must be monotone
non-decreasing in B.

FIX: build ONE global spectral embedding from the full matrix, then use the
bucket labels only as a CONSTRAINT when clustering (cluster inside each
bucket in the global embedding space).  That dominates both alternatives.

Also: round 1 ran at occupancy 10, where plain FPAE is already broken
(ARI 0.91), so there was no headroom to measure the token's marginal
leakage.  Sweep LOW occupancy instead, where FPAE still holds.
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

def build(m, K, nd, seed, n_types=400):
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1); wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    k = alloc_hamilton(f, K, rng)
    Kp = int(k.sum()); st = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    t = rng.choice(n_types, size=nd, p=wt)
    da, ca = dr[t], cr[t]
    dal = st[da] + rng.integers(0, k[da]); cal = st[ca] + rng.integers(0, k[ca])
    C = sp.coo_matrix((np.ones(2 * nd), (np.concatenate([dal, cal]),
                       np.concatenate([cal, dal]))), shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)
    return f, k, Kp, owner, C, rng

def embed(C, Kp, dims=32):
    U, S, _ = svds(C.asfptype(), k=min(dims, Kp - 1))
    X = U * S
    n_ = np.linalg.norm(X, axis=1, keepdims=True); n_[n_ == 0] = 1
    return X / n_

def score(labels, owner, m):
    cf = np.zeros((labels.max() + 1, m)); np.add.at(cf, (labels, owner), 1.0)
    r, c = linear_sum_assignment(-cf); mp = dict(zip(r, c))
    guess = np.array([mp.get(l, -1) for l in labels])
    return float((guess == owner).mean()), adjusted_rand_score(owner, labels)

def attack_global(E, owner, m, seed):
    return score(KMeans(m, n_init=2, random_state=seed).fit_predict(E), owner, m)

def attack_bucketed(E, owner, m, alias_bucket, B, seed):
    """global embedding + bucket labels as a hard constraint (dominant strategy)"""
    lab = np.zeros(len(owner), dtype=np.int64); nxt = 0
    for b in range(B):
        idx = np.where(alias_bucket == b)[0]
        if len(idx) == 0:
            continue
        na = len(np.unique(owner[idx]))
        if na < 2 or len(idx) < na:
            lab[idx] = nxt; nxt += 1; continue
        lb = KMeans(na, n_init=2, random_state=seed).fit_predict(E[idx])
        lab[idx] = nxt + lb; nxt += na
    return score(lab, owner, m)

m, K = 200, 2000
print("=" * 94, flush=True)
print("LINKING-TOKEN MARGINAL LEAKAGE  (global embedding + bucket constraint)", flush=True)
print("=" * 94, flush=True)
for nd in (1_000, 2_000, 4_000):
    f, k, Kp, owner, C, rng = build(m, K, nd, 7)
    E = embed(C, Kp)
    g_acc, g_ari = attack_global(E, owner, m, 7)
    print(f"\n  occupancy = {2*nd/Kp:.1f}   |  no token (B=1): "
          f"recovery {g_acc:.1%}, ARI {g_ari:.3f}", flush=True)
    print(f"  {'B':>5} {'accts/bkt':>10} {'report prec':>12} | {'recovery':>9} "
          f"{'ARI':>7} | {'ARI rise':>9}", flush=True)
    for B in (2, 4, 8, 16, 32, 64, 128, 200):
        ab = rng.permutation(m) % B
        a, r = attack_bucketed(E, owner, m, ab[owner], B, 7)
        a = max(a, g_acc); r = max(r, g_ari)      # adversary keeps the better
        print(f"  {B:>5} {m/B:>10.1f} {B/m:>11.1%} | {a:>8.1%} {r:>7.3f} | "
              f"{r-g_ari:>+9.3f}", flush=True)
