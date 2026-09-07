#!/usr/bin/env python3
"""
Two checks on the sensitivity sweep before it is trusted:
 (A) the sweep used refine=1 (one refinement round) for speed, while the
     earlier attacks used refine=2-3.  Fewer refinements = WEAKER adversary,
     so the sweep may be optimistic.  Compare refine=1 vs 3 at higher occupancy.
 (B) push the worst-case setting found by the sweep (sparse pairing,
     types=100) up the occupancy axis to locate its breaking point.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
ALPHA = 10_000

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

def run(m, s, n_types, occ, refine, seed=7, dims=32):
    rng = np.random.default_rng(seed); K = ALPHA - m
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1) ** s; wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum(); k = alloc_hamilton(f, K, rng)
    Kp = int(k.sum()); st = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    nd = int(occ * Kp / 2)
    t = rng.choice(n_types, size=nd, p=wt); da, ca = dr[t], cr[t]
    dal = st[da] + rng.integers(0, k[da]); cal = st[ca] + rng.integers(0, k[ca])
    C = sp.coo_matrix((np.ones(2 * nd), (np.concatenate([dal, cal]),
                       np.concatenate([cal, dal]))), shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)
    U, S, _ = svds(C.asfptype(), k=dims)
    X = U * S; nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    KM = lambda n: MiniBatchKMeans(n, n_init=3, random_state=seed, batch_size=1024)
    lab = KM(m).fit_predict(X / nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(Kp), (np.arange(Kp), lab)), shape=(Kp, m)).tocsr()
        Q = np.asarray((C @ H).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KM(m).fit_predict(Q / nn)
    return adjusted_rand_score(owner, lab)

print("=" * 74, flush=True)
print("(A) does refinement depth change the verdict?  (m=200, s=1.0, types=400)", flush=True)
print("=" * 74, flush=True)
print(f"  {'occupancy':>10} {'refine=1':>10} {'refine=3':>10} {'diff':>8}", flush=True)
for occ in (1.0, 2.0, 4.0, 8.0):
    a1 = run(200, 1.0, 400, occ, 1); a3 = run(200, 1.0, 400, occ, 3)
    print(f"  {occ:>10.1f} {a1:>10.3f} {a3:>10.3f} {a3-a1:>+8.3f}", flush=True)

print("", flush=True)
print("=" * 74, flush=True)
print("(B) worst case from the sweep -- sparse pairing (types=100), refine=3", flush=True)
print("=" * 74, flush=True)
print(f"  {'occupancy':>10} {'ARI':>8}  regime", flush=True)
for occ in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    a = run(200, 1.0, 100, occ, 3)
    reg = "safe" if a < 0.10 else "leaking" if a < 0.35 else "UNSAFE"
    print(f"  {occ:>10.1f} {a:>8.3f}  {reg}", flush=True)
