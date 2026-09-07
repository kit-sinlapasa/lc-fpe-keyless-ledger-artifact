#!/usr/bin/env python3
"""
Does exposing the full permuted triple help or hurt the clustering adversary?

Our clustering experiments modelled the observable as the ALIAS alone.  The
deployed construction encrypts <a_hat, c, d> jointly, so what an adversary
actually sees is one opaque symbol per distinct (a_hat, c, d) triple.  We
claimed in the manuscript that this is conservative -- more classes to merge,
so a weaker adversary.  That claim was never tested, and it sits uneasily
beside our own CPA result, where adding structure HELPED the adversary.

So: run the same spectral-clustering attack under both observables and compare
the ARI of recovering the ACCOUNT.

  observable A : alias                (what we simulated)
  observable B : (alias, c, d) triple (what deployment exposes)

Both are scored against the same ground truth: the account of each row.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

N_CC, N_DT = 40, 6


def alloc_hamilton(f, K):
    m = len(f)
    q = f * K
    base = np.maximum(np.floor(q).astype(np.int64), 1)
    short = int(K - base.sum())
    if short > 0:
        base[np.argsort(-(q - np.floor(q)))[:short]] += 1
    while short < 0:
        cand = np.nonzero(base > 1)[0]
        take = min(-short, cand.size)
        base[cand[np.argsort((q - base)[cand])[:take]]] -= 1
        short += take
    return base


def cluster(rows_u, rows_v, n_nodes, owner, m, seed, dims=32, refine=2):
    """rows_u, rows_v: the two node ids co-occurring in each document."""
    C = sp.coo_matrix((np.ones(2 * len(rows_u)),
                       (np.concatenate([rows_u, rows_v]),
                        np.concatenate([rows_v, rows_u]))),
                      shape=(n_nodes, n_nodes)).tocsr()
    C.data = np.log1p(C.data)
    keep = np.asarray((C != 0).sum(axis=1)).ravel() > 0
    idx = np.nonzero(keep)[0]
    Cs = C[idx][:, idx]
    d = min(dims, len(idx) - 1)
    U, S, _ = svds(Cs.asfptype(), k=d, random_state=seed)
    X = U * S
    n_ = np.linalg.norm(X, axis=1, keepdims=True); n_[n_ == 0] = 1
    KM = lambda n: MiniBatchKMeans(n, n_init=3, random_state=seed,
                                   batch_size=1024)
    lab = KM(m).fit_predict(X / n_)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(len(idx)), (np.arange(len(idx)), lab)),
                          shape=(len(idx), m)).tocsr()
        Q = np.asarray((Cs @ H).todense())
        n_ = np.linalg.norm(Q, axis=1, keepdims=True); n_[n_ == 0] = 1
        lab = KM(m).fit_predict(Q / n_)
    full = np.full(n_nodes, -1)
    full[idx] = lab
    return full, idx


def run(m, K, n_docs, n_cc, n_dt, seed, n_types=250):
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1); wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    k = alloc_hamilton(f, K)
    Kp = int(k.sum()); st = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner_alias = np.repeat(np.arange(m), k)

    # each account is used with a limited set of (c, d) -- the correlation
    # that made the conditional-domain attack work
    cc_of = [rng.choice(N_CC, size=n_cc, replace=False) for _ in range(m)]
    dt_of = [rng.choice(N_DT, size=n_dt, replace=False) for _ in range(m)]

    t = rng.choice(n_types, size=n_docs, p=wt)
    da, ca = dr[t], cr[t]
    d_al = st[da] + rng.integers(0, k[da])
    c_al = st[ca] + rng.integers(0, k[ca])
    d_cc = np.array([cc_of[x][rng.integers(n_cc)] for x in da])
    d_dt = np.array([dt_of[x][rng.integers(n_dt)] for x in da])
    c_cc = np.array([cc_of[x][rng.integers(n_cc)] for x in ca])
    c_dt = np.array([dt_of[x][rng.integers(n_dt)] for x in ca])

    # observable A: alias only
    labA, idxA = cluster(d_al, c_al, Kp, owner_alias, m, seed)
    ariA = adjusted_rand_score(owner_alias[idxA], labA[idxA])

    # observable B: (alias, c, d) triple -> one opaque symbol each
    tripd = (d_al * N_CC + d_cc) * N_DT + d_dt
    tripc = (c_al * N_CC + c_cc) * N_DT + c_dt
    uniq, inv = np.unique(np.concatenate([tripd, tripc]), return_inverse=True)
    nd = len(tripd)
    bd, bc = inv[:nd], inv[nd:]
    owner_trip = np.empty(len(uniq), dtype=np.int64)
    owner_trip[bd] = owner_alias[d_al]
    owner_trip[bc] = owner_alias[c_al]
    labB, idxB = cluster(bd, bc, len(uniq), owner_trip, m, seed)
    ariB = adjusted_rand_score(owner_trip[idxB], labB[idxB])

    return Kp, len(uniq), 2 * n_docs / Kp, ariA, ariB


print("=" * 84, flush=True)
print("OBSERVABLE: alias alone  vs  full (alias, c, d) triple", flush=True)
print("=" * 84, flush=True)
print("  ARI of recovering the ACCOUNT, same ledger, same attack", flush=True)
print(flush=True)
print("  {:>6} {:>8} {:>9} {:>7} | {:>9} {:>9} {:>9}".format(
    "docs", "aliases", "triples", "occ", "ARI alias", "ARI triple", "delta"),
    flush=True)
m, K = 100, 2000
for nd in (5_000, 15_000, 40_000):
    for n_cc, n_dt in ((2, 2), (5, 3)):
        a = []
        for seed in (11, 22):
            Kp, nt, occ, x, y = run(m, K, nd, n_cc, n_dt, seed)
            a.append((x, y))
        xa = float(np.mean([p[0] for p in a]))
        ya = float(np.mean([p[1] for p in a]))
        print("  {:>6,} {:>8} {:>9} {:>7.1f} | {:>9.3f} {:>9.3f} {:>+9.3f}   "
              "(cc x dt = {}x{})".format(nd, Kp, nt, occ, xa, ya, ya - xa,
                                         n_cc, n_dt), flush=True)
print(flush=True)
print("  positive delta => the deployed observable is WORSE for us than what", flush=True)
print("  we simulated, i.e. the manuscript's 'conservative' claim is false.", flush=True)
