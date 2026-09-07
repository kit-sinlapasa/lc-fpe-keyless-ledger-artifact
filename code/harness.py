#!/usr/bin/env python3
"""
Multi-seed experiment harness.

Why this exists: 22 of our 24 scripts ran a single seed, and those that
averaged (trials=300) averaged only the SAMPLING randomness -- the chart of
accounts and the pairing structure were one fixed instance.  Every headline
number was therefore conditional on one synthetic world.

Here the seed drives the whole world: chart frequencies, which accounts pair
with which, cost-centre and document-type assignments, and the rows.  Results
are reported as mean with a 95% t-interval.

Everything downstream should import from this module rather than re-deriving
its own allocation or ledger generator.

Reproducibility note (2026-09-06): the spectral step calls scipy's `svds`,
whose ARPACK start vector is random unless `random_state` is given. Every
`svds` call in this codebase now passes the world's seed, so a fixed seed
gives a bit-identical ARI. Before this, three repeats of the same seed at
occupancy 1 gave 0.0221 / 0.0248 / 0.0230, which is why earlier runs of the
same tables disagree in the third decimal.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.stats import t as student_t
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

N_CC, N_DT = 100, 10


# ------------------------------------------------------------------ allocation
def alloc_opt(f, K):
    """Exact minimiser of sum_a |f(a)K - k(a)| subject to sum k = K, k >= 1.

    The objective is separable and each term |q_a - k_a| is convex in k_a, so
    starting from k = 1 and handing out the remaining K - m units one at a
    time to whichever account has the smallest marginal cost
    |q_a-(k_a+1)| - |q_a-k_a| is optimal (a standard greedy for separable
    convex resource allocation). Largest-remainder with a floor is NOT
    optimal: for f=(.46,.36,.09,.09), K=10 it returns (4,3,2,1) with L1 2.4
    where (5,3,1,1) achieves 1.2, because the floor has already pushed the
    two small quotas above their targets and the remainder rule still feeds
    them. The paper's optimality claim is stated for this allocator."""
    import heapq
    m = len(f)
    if K < m:
        raise ValueError("infeasible: K={} < m={}".format(K, m))
    q = np.asarray(f, dtype=float) * K
    k = np.ones(m, dtype=np.int64)
    heap = [(abs(q[i] - 2) - abs(q[i] - 1), i) for i in range(m)]
    heapq.heapify(heap)
    for _ in range(int(K - m)):
        _, i = heapq.heappop(heap)
        k[i] += 1
        heapq.heappush(heap, (abs(q[i] - (k[i] + 1)) - abs(q[i] - k[i]), i))
    assert int(k.sum()) == K and k.min() >= 1
    return k


def alloc_hamilton(f, K):
    """Kept under its old name so every experiment script keeps running, but
    it now returns the exact minimiser above. The largest-remainder code it
    replaced is alloc_hamilton_legacy, retained for provenance; on the
    structured chart at m=200 and m=300 the two agree exactly, and on the
    flat chart they differ only where the floor is active (skew >= 1.3)."""
    return alloc_opt(f, K)


def alloc_hamilton_legacy(f, K):
    """Largest-remainder with a floor of one, balanced to budget in both
       directions.  Repeats clawback until sum(k) == K exactly.  Superseded:
       not L1-optimal once the floor binds (see alloc_opt)."""
    m = len(f)
    if K < m:
        raise ValueError("infeasible: K={} < m={}".format(K, m))
    q = f * K
    base = np.maximum(np.floor(q).astype(np.int64), 1)
    short = int(K - base.sum())
    if short > 0:
        base[np.argsort(-(q - np.floor(q)))[:short]] += 1
    while short < 0:
        cand = np.nonzero(base > 1)[0]
        if cand.size == 0:
            raise RuntimeError("clawback exhausted")
        take = min(-short, cand.size)
        base[cand[np.argsort((q - base)[cand])[:take]]] -= 1
        short += take
    assert int(base.sum()) == K
    return base


# ------------------------------------------------------------------ the world
class World:
    """One synthetic accounting world. Everything varies with the seed."""

    def __init__(self, seed, m=300, s=1.0, n_types=400, K=None,
                 n_cc=5, n_dt=3, alpha=10_000,
                 pool_cc=N_CC, pool_dt=N_DT):
        self.rng = np.random.default_rng(seed)
        self.seed, self.m, self.s = seed, m, s
        self.K = (alpha - m) if K is None else K

        # transaction types define which accounts pair with which
        self.dr = self.rng.integers(0, m, n_types)
        self.cr = self.rng.integers(0, m, n_types)
        bad = self.dr == self.cr
        self.cr[bad] = (self.cr[bad] + 1) % m
        w = 1.0 / np.arange(1, n_types + 1) ** s
        self.wt = w / w.sum()

        cnt = np.zeros(m)
        np.add.at(cnt, self.dr, self.wt)
        np.add.at(cnt, self.cr, self.wt)
        self.f = cnt / cnt.sum()

        self.k = alloc_hamilton(self.f, self.K)
        self.Kp = int(self.k.sum())
        self.start = np.concatenate([[0], np.cumsum(self.k)[:-1]])
        self.owner = np.repeat(np.arange(m), self.k)

        # each account touches only a few cost centres / document types
        self.cc_of = [self.rng.choice(pool_cc, n_cc, replace=False)
                      for _ in range(m)]
        self.dt_of = [self.rng.choice(pool_dt, n_dt, replace=False)
                      for _ in range(m)]
        self.n_cc, self.n_dt = n_cc, n_dt

    def delta(self):
        """analytic flatness TV(f, k/K')"""
        return 0.5 * float(np.abs(self.f - self.k / self.Kp).sum())

    def docs(self, n_docs):
        """returns (debit account, credit account, debit alias, credit alias,
                    debit cc, debit dt, credit cc, credit dt)"""
        r = self.rng
        t = r.choice(len(self.wt), size=n_docs, p=self.wt)
        da, ca = self.dr[t], self.cr[t]
        d_al = self.start[da] + r.integers(0, self.k[da])
        c_al = self.start[ca] + r.integers(0, self.k[ca])
        d_cc = np.array([self.cc_of[x][r.integers(self.n_cc)] for x in da])
        d_dt = np.array([self.dt_of[x][r.integers(self.n_dt)] for x in da])
        c_cc = np.array([self.cc_of[x][r.integers(self.n_cc)] for x in ca])
        c_dt = np.array([self.dt_of[x][r.integers(self.n_dt)] for x in ca])
        return da, ca, d_al, c_al, d_cc, d_dt, c_cc, c_dt

    def docs_for_occupancy(self, occ):
        return self.docs(int(occ * self.Kp / 2))


# ------------------------------------------------------------------ attack
def cluster_ari(world, u, v, n_nodes, node_owner, dims=32, refine=1):
    """spectral embedding + k-means + refinement; returns ARI vs node_owner"""
    C = sp.coo_matrix((np.ones(2 * len(u)),
                       (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(n_nodes, n_nodes)).tocsr()
    C.data = np.log1p(C.data)
    seen = np.asarray((C != 0).sum(axis=1)).ravel() > 0
    idx = np.nonzero(seen)[0]
    if len(idx) < world.m + 2:
        return float("nan")
    Cs = C[idx][:, idx]
    U, S, _ = svds(Cs.asfptype(), k=min(dims, len(idx) - 1), random_state=world.seed)
    X = U * S
    nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    KM = lambda n: MiniBatchKMeans(n, n_init=3, random_state=world.seed,
                                   batch_size=1024)
    lab = KM(world.m).fit_predict(X / nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(len(idx)), (np.arange(len(idx)), lab)),
                          shape=(len(idx), world.m)).tocsr()
        Q = np.asarray((Cs @ H).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KM(world.m).fit_predict(Q / nn)
    return float(adjusted_rand_score(node_owner[idx], lab))


# ------------------------------------------------------------------ statistics
def ci95(vals):
    """mean and half-width of the 95% t-interval"""
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(a[0]), float("nan"), 1
    h = student_t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return float(a.mean()), float(h), n


def fmt(mean, half, n):
    if n <= 1 or not np.isfinite(half):
        return "{:.4f} (n=1)".format(mean)
    return "{:.4f} +/- {:.4f}".format(mean, half)


def repeat(fn, seeds):
    """run fn(seed) over seeds, return (mean, ci_halfwidth, n, raw)"""
    raw = [fn(s) for s in seeds]
    m, h, n = ci95(raw)
    return m, h, n, raw
