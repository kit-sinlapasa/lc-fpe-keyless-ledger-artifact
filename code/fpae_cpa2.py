#!/usr/bin/env python3
"""CPA vs plain FPAE -- vectorised, unbuffered, smaller K for speed."""
import sys, numpy as np, scipy.sparse as sp
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

def model(m, n_types, rng):
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1); wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    return dr, cr, wt, cnt / cnt.sum()

def bundles(m, k, dr, cr, wt, rng):
    """per (a,b) -> (first_bin, n_bins), weight-balanced greedy least-loaded."""
    pw = {}
    for t in range(len(wt)):
        for a, b in ((dr[t], cr[t]), (cr[t], dr[t])):
            pw[(int(a), int(b))] = pw.get((int(a), int(b)), 0.0) + wt[t]
    by = {}
    for (a, b), w in pw.items():
        by.setdefault(a, []).append((b, w))
    fb, nb = {}, {}
    for a, lst in by.items():
        ka = int(k[a]); tot = sum(w for _, w in lst); cap = tot / ka
        lst = [lst[i] for i in rng.permutation(len(lst))]
        lst.sort(key=lambda x: -x[1])
        load = np.zeros(ka); nxt = 0
        for b, w in lst:
            if cap > 0 and w > cap:
                n = min(ka, max(1, int(round(w / cap))))
                fb[(a, b)] = nxt; nb[(a, b)] = n
                for i in range(n):
                    load[(nxt + i) % ka] += w / n
                nxt = (nxt + n) % ka
            else:
                j = int(np.argmin(load)); fb[(a, b)] = j; nb[(a, b)] = 1
                load[j] += w
    return fb, nb

def sim(m, K, nd, seed, mode, n_types=400, dims=32, refine=2):
    rng = np.random.default_rng(seed)
    dr, cr, wt, f = model(m, n_types, rng)
    k = alloc_hamilton(f, K, rng)
    Kp = int(k.sum()); st = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    t = rng.choice(n_types, size=nd, p=wt)
    da, ca = dr[t], cr[t]
    if mode == "fpae":
        dal = st[da] + rng.integers(0, k[da])
        cal = st[ca] + rng.integers(0, k[ca])
    else:
        fb, nbm = bundles(m, k, dr, cr, wt, rng)
        fbd = np.array([fb[(int(dr[i]), int(cr[i]))] for i in range(n_types)])
        nbd = np.array([nbm[(int(dr[i]), int(cr[i]))] for i in range(n_types)])
        fbc = np.array([fb[(int(cr[i]), int(dr[i]))] for i in range(n_types)])
        nbc = np.array([nbm[(int(cr[i]), int(dr[i]))] for i in range(n_types)])
        dal = st[da] + (fbd[t] + rng.integers(0, nbd[t])) % k[da]
        cal = st[ca] + (fbc[t] + rng.integers(0, nbc[t])) % k[ca]
    cnt = np.bincount(np.concatenate([dal, cal]), minlength=Kp)
    delta = 0.5 * np.abs(cnt / cnt.sum() - 1.0 / Kp).sum()
    C = sp.coo_matrix((np.ones(2 * nd), (np.concatenate([dal, cal]),
                       np.concatenate([cal, dal]))), shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)
    U, S, _ = svds(C.asfptype(), k=min(dims, Kp - 1))
    X = U * S; n_ = np.linalg.norm(X, axis=1, keepdims=True); n_[n_ == 0] = 1
    lab = KMeans(m, n_init=2, random_state=seed).fit_predict(X / n_)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(Kp), (np.arange(Kp), lab)), shape=(Kp, m)).tocsr()
        Q = np.asarray((C @ H).todense())
        n_ = np.linalg.norm(Q, axis=1, keepdims=True); n_[n_ == 0] = 1
        lab = KMeans(m, n_init=2, random_state=seed).fit_predict(Q / n_)
    cf = np.zeros((m, m)); np.add.at(cf, (lab, owner), 1.0)
    r, c = linear_sum_assignment(-cf); mp = dict(zip(r, c))
    acc = float((np.array([mp[l] for l in lab]) == owner).mean())
    nl = np.repeat(np.arange(m), k)[rng.permutation(Kp)]
    cn = np.zeros((m, m)); np.add.at(cn, (nl, owner), 1.0)
    r2, c2 = linear_sum_assignment(-cn); mp2 = dict(zip(r2, c2))
    nac = float((np.array([mp2[l] for l in nl]) == owner).mean())
    return 2 * nd / Kp, delta, acc, nac, adjusted_rand_score(owner, lab)

m, K = 200, 2000
print("=" * 92, flush=True)
print("CPA vs plain FPAE   (m=200 accounts, K=2000 aliases)", flush=True)
print("=" * 92, flush=True)
print(f"  {'docs':>8} {'occ':>6} | {'D fpae':>8} {'rec':>7} {'ARI':>7} |"
      f" {'D cpa':>8} {'rec':>7} {'ARI':>7} | {'null':>6}", flush=True)
for nd in (5_000, 15_000, 50_000, 150_000):
    o, d1, a1, nl1, r1 = sim(m, K, nd, 7, "fpae")
    _, d2, a2, nl2, r2 = sim(m, K, nd, 7, "cpa")
    print(f"  {nd:>8,} {o:>6.1f} | {d1:>8.4f} {a1:>6.1%} {r1:>7.3f} |"
          f" {d2:>8.4f} {a2:>6.1%} {r2:>7.3f} | {nl1:>5.1%}", flush=True)
