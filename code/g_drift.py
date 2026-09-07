#!/usr/bin/env python3
"""
G. The frequency-drift term.

Theorem 2 bounds the residual non-uniformity of the aliased distribution by

    Delta = TV(f, k/K')  <=  TV(f, f-hat) + m/K          (m/2K, floor inactive)

where f is the true account-frequency profile of the encrypted period and f-hat
is the estimate the allocation was built from.  The paper calls TV(f, f-hat)
"measurable from history" and then never measures it, quoting only the second
term (m/2K ~ 0.010 at m=200, K=9800).  A deployer does not get f.  They get a
sample of some past period, and the ledger they encrypt is a different period.
This measures what that costs, from two sources:

  SAMPLING    f-hat is the empirical profile of n historical rows.
  DRIFT       the business mix moves between the estimation period and the
              encrypted one: expense-side archetypes are reweighted by a factor
              (1+delta) and the profile renormalised.

For each we report TV(f, f-hat), the realised Delta, the bound, and -- the part
that actually decides anything -- whether the clustering attack gets better.
A bound on statistical distance is not a bound on recovery; the paper's budget
is denominated in ARI, so drift has to be checked there too.
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


def reallocate(w, fhat):
    """Rebuild the alias layer from fhat while the rows still come from w.f."""
    k = alloc_hamilton(fhat, K)
    w.k = k
    w.Kp = int(k.sum())
    w.start = np.concatenate([[0], np.cumsum(k)[:-1]])
    w.owner = np.repeat(np.arange(w.m), k)
    return w


def delta(w):
    """TV between the aliased distribution and uniform on the alias support."""
    return 0.5 * float(np.abs(w.f - w.k / w.Kp).sum())


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def attack(w, occ, dims=64, refine=3):
    r = w.rng
    _, _, d_al, c_al, _, _, _, _ = w.docs_for_occupancy(occ)
    n = w.Kp
    C = sp.coo_matrix((np.ones(2 * len(d_al)),
                       (np.concatenate([d_al, c_al]),
                        np.concatenate([c_al, d_al]))), shape=(n, n)).tocsr()
    C.data = np.log1p(C.data)
    idx = np.nonzero(np.asarray((C != 0).sum(axis=1)).ravel() > 0)[0]
    if len(idx) < w.m + 2:
        return float("nan")
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
    return float(adjusted_rand_score(w.owner[idx], lab))


def sampled(w, n, rng):
    """Empirical profile of n historical rows drawn from the true profile."""
    c = rng.multinomial(n, w.f).astype(float)
    return c / c.sum()


def drifted(w, d):
    """Expense-side accounts grow by (1+d); everything renormalises."""
    g = w.f.copy()
    g[w.by_cat["expense"]] *= (1.0 + d)
    return g / g.sum()


def run(make_fhat, label, values, do_attack=True):
    print(flush=True)
    print("  {}".format(label), flush=True)
    print("  {:>12} {:>16} {:>16} {:>14} {:>20}".format(
        "setting", "TV(f,fhat)", "Delta realised", "bound", "ARI at occ 1"),
        flush=True)
    for v in values:
        TVs, DEs, BDs, ARs = [], [], [], []
        for s in SEEDS:
            w = TFRSWorld(s, m=M, K=K)
            rng = np.random.default_rng(1000 + s)
            fhat = make_fhat(w, v, rng)
            t = tv(w.f, fhat)
            reallocate(w, fhat)
            TVs.append(t); DEs.append(delta(w))
            BDs.append(t + M / K)
            if do_attack:
                ARs.append(attack(w, 1.0))
        tm, th, _ = ci95(TVs); dm, dh, _ = ci95(DEs); bm, _, _ = ci95(BDs)
        if do_attack:
            am, ah, _ = ci95(ARs)
            a = "{:.4f} +/- {:.4f}".format(am, ah)
        else:
            a = "--"
        ok = "" if dm <= bm else "   <== BOUND VIOLATED"
        print("  {:>12} {:>9.4f} +/- {:.4f} {:>9.4f} +/- {:.4f} {:>14.4f} "
              "{:>20}{}".format(str(v), tm, th, dm, dh, bm, a, ok), flush=True)


print("=" * 104, flush=True)
print("G. FREQUENCY DRIFT -- the TV(f, f-hat) term of Theorem 2", flush=True)
print("=" * 104, flush=True)
print("  structured TFRS chart, m={}, K={}, three seeds".format(M, K),
      flush=True)
print("  allocation term for reference:  m/2K = {:.4f}   m/K = {:.4f}".format(
    M / (2 * K), M / K), flush=True)

run(lambda w, v, r: w.f.copy(), "perfect estimate (f-hat = f)", ["exact"])
run(lambda w, n, r: sampled(w, n, r),
    "SAMPLING: f-hat from n historical rows",
    [1_000, 10_000, 100_000, 1_000_000])
run(lambda w, d, r: drifted(w, d),
    "DRIFT: expense side reweighted by (1+delta)",
    [0.1, 0.25, 0.5, 1.0, 2.0])

print(flush=True)
print("done", flush=True)
