#!/usr/bin/env python3
"""
SENSITIVITY ANALYSIS -- is the design rule "occupancy <= 1" robust across the
parameter space, or does it shift with the shape of the chart of accounts?

Four parameters that our results depend on:
  m         size of the chart of accounts
  s         Zipf exponent of account frequency (skew)
  n_types   number of distinct transaction types = pairing sparsity
  occ       occupancy R/K' (rows per alias)

Baseline: alpha = 10^4 (4-digit codes), K = alpha - m, m=200, s=1.0,
n_types=400.  One-factor-at-a-time sweep around it, at occupancy 1 and 2.

MiniBatchKMeans is used for speed; a validation row compares it against full
KMeans at the baseline so the approximation is checked, not assumed.
"""
import numpy as np, scipy.sparse as sp, time
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans, MiniBatchKMeans
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

def run(m, s, n_types, occ, seed=7, dims=32, refine=1, fast=True):
    rng = np.random.default_rng(seed)
    K = ALPHA - m
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1) ** s; wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    k = alloc_hamilton(f, K, rng)
    Kp = int(k.sum()); st = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    delta = 0.5 * np.abs(f - k / Kp).sum()          # analytic flatness
    nd = int(occ * Kp / 2)
    t = rng.choice(n_types, size=nd, p=wt)
    da, ca = dr[t], cr[t]
    dal = st[da] + rng.integers(0, k[da]); cal = st[ca] + rng.integers(0, k[ca])
    C = sp.coo_matrix((np.ones(2 * nd), (np.concatenate([dal, cal]),
                       np.concatenate([cal, dal]))), shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)
    U, S, _ = svds(C.asfptype(), k=min(dims, Kp - 1))
    X = U * S; nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    KM = (lambda n: MiniBatchKMeans(n, n_init=3, random_state=seed,
                                    batch_size=1024)) if fast else \
         (lambda n: KMeans(n, n_init=2, random_state=seed))
    lab = KM(m).fit_predict(X / nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(Kp), (np.arange(Kp), lab)), shape=(Kp, m)).tocsr()
        Q = np.asarray((C @ H).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KM(m).fit_predict(Q / nn)
    return Kp, delta, adjusted_rand_score(owner, lab)

print("=" * 96, flush=True)
print("SENSITIVITY ANALYSIS   alpha=10^4, K=alpha-m", flush=True)
print("=" * 96, flush=True)

t0 = time.time()
_, d_f, a_f = run(200, 1.0, 400, 1.0, fast=True)
_, d_s, a_s = run(200, 1.0, 400, 1.0, fast=False)
print(f"\n[validation @ baseline, occ=1]  MiniBatch ARI={a_f:.3f}  "
      f"full KMeans ARI={a_s:.3f}  diff={abs(a_f-a_s):.3f}  "
      f"({time.time()-t0:.0f}s)", flush=True)

def sweep(title, cases):
    print(f"\n--- {title} ---", flush=True)
    print(f"  {'setting':>22} {'K prime':>8} {'Delta':>8} | {'ARI occ=1':>10} "
          f"{'ARI occ=2':>10}  verdict", flush=True)
    for label, (m, s, nt) in cases:
        Kp, d, a1 = run(m, s, nt, 1.0)
        _, _, a2 = run(m, s, nt, 2.0)
        v = "safe" if a1 < 0.10 else "leaking" if a1 < 0.35 else "UNSAFE"
        print(f"  {label:>22} {Kp:>8} {d:>8.4f} | {a1:>10.3f} {a2:>10.3f}  {v}",
              flush=True)

sweep("chart size m", [(f"m={m}", (m, 1.0, 400)) for m in (50, 100, 200, 400)])
sweep("skew (Zipf s)", [(f"s={s}", (200, s, 400)) for s in (0.6, 0.8, 1.0, 1.3, 1.6)])
sweep("pairing sparsity", [(f"types={n}", (200, 1.0, n))
                          for n in (100, 200, 400, 800, 1600)])
print("\ndone", flush=True)
