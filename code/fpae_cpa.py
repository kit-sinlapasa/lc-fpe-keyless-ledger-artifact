#!/usr/bin/env python3
"""
CPA -- Counter-Partitioned Aliasing:  a candidate fix for the co-occurrence
clustering attack on FPAE.

WHY the attack works: j = PRF(key, doc||line) mod k(a) is independent of the
counter-account, so every alias of account a sees the SAME counter mix ->
all its aliases have statistically identical profiles -> they cluster.

CPA: give each alias j of account a its own BUNDLE B_j of counter-accounts,
so the profiles of sibling aliases are DISJOINT and cannot be clustered.
Bundles are weight-balanced (greedy least-loaded over a keyed shuffle) so the
flat frequency histogram that FPAE bought is preserved.

Heavy counter-accounts (weight > 1/k(a)) cannot fit in one bundle; they are
given several dedicated bundles and split randomly across them (a local
fallback to plain FPAE for that pair).

Measured: (1) flatness Delta, (2) clustering recovery + ARI, vs plain FPAE.
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


def build_model(m, n_types, rng):
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1); wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    return dr, cr, wt, cnt / cnt.sum()


def cpa_bundles(m, k, dr, cr, wt, rng):
    """For each account a, assign its counter-accounts to k(a) bundles,
       weight-balanced.  Returns dict (a, b) -> list of bundle indices."""
    pair_w = {}
    for t in range(len(wt)):
        for a, b in ((dr[t], cr[t]), (cr[t], dr[t])):
            pair_w[(a, b)] = pair_w.get((a, b), 0.0) + wt[t]
    by_a = {}
    for (a, b), w in pair_w.items():
        by_a.setdefault(a, []).append((b, w))
    assign = {}
    for a, lst in by_a.items():
        ka = int(k[a]); tot = sum(w for _, w in lst)
        cap = tot / ka                                  # target load per bundle
        order = [lst[i] for i in rng.permutation(len(lst))]
        order.sort(key=lambda x: -x[1])
        load = np.zeros(ka); nxt = 0
        for b, w in order:
            if w > cap and cap > 0:                     # heavy: dedicate bundles
                nb = min(ka, max(1, int(round(w / cap))))
                bins = [(nxt + i) % ka for i in range(nb)]
                nxt = (nxt + nb) % ka
                assign[(a, b)] = bins
                for x in bins:
                    load[x] += w / nb
            else:
                j = int(np.argmin(load))
                assign[(a, b)] = [j]
                load[j] += w
    return assign


def simulate(m, K, n_docs, seed, mode, n_types=400, dims=32, refine=2):
    rng = np.random.default_rng(seed)
    dr, cr, wt, f = build_model(m, n_types, rng)
    k = alloc_hamilton(f, K, rng)
    Kp = int(k.sum()); start = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    t = rng.choice(n_types, size=n_docs, p=wt)
    da, ca = dr[t], cr[t]

    if mode == "fpae":
        d_al = start[da] + rng.integers(0, k[da])
        c_al = start[ca] + rng.integers(0, k[ca])
    else:                                               # cpa
        asg = cpa_bundles(m, k, dr, cr, wt, rng)
        d_al = np.empty(n_docs, dtype=np.int64)
        c_al = np.empty(n_docs, dtype=np.int64)
        for i in range(n_docs):
            a, b = da[i], ca[i]
            ba = asg[(a, b)]; bb = asg[(b, a)]
            d_al[i] = start[a] + ba[rng.integers(len(ba))]
            c_al[i] = start[b] + bb[rng.integers(len(bb))]

    # flatness of the REALISED alias histogram
    cnt = np.bincount(np.concatenate([d_al, c_al]), minlength=Kp)
    g = cnt / cnt.sum()
    delta = 0.5 * np.abs(g - 1.0 / Kp).sum()

    C = sp.coo_matrix((np.ones(2 * n_docs),
                       (np.concatenate([d_al, c_al]),
                        np.concatenate([c_al, d_al]))), shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)
    U, S, _ = svds(C.asfptype(), k=min(dims, Kp - 1))
    X = U * S
    nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    lab = KMeans(m, n_init=2, random_state=seed).fit_predict(X / nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(Kp), (np.arange(Kp), lab)),
                          shape=(Kp, m)).tocsr()
        Q = np.asarray((C @ H).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KMeans(m, n_init=2, random_state=seed).fit_predict(Q / nn)

    conf = np.zeros((m, m)); np.add.at(conf, (lab, owner), 1.0)
    r, c = linear_sum_assignment(-conf); mp = dict(zip(r, c))
    acc = float((np.array([mp[l] for l in lab]) == owner).mean())
    ari = adjusted_rand_score(owner, lab)
    return 2 * n_docs / Kp, delta, acc, ari


m, K = 300, 2000          # smaller K so CPA bundling is meaningful and fast
print("=" * 90)
print("CPA vs plain FPAE  --  does counter-partitioned aliasing kill the")
print("clustering attack, and what does it cost in flatness?")
print("=" * 90)
print(f"  m = {m}, K = {K}")
print()
print(f"  {'docs':>8} {'occ':>7} | {'FPAE Delta':>11} {'FPAE rec':>9} {'FPAE ARI':>9}"
      f" | {'CPA Delta':>10} {'CPA rec':>8} {'CPA ARI':>8}")
for nd in (10_000, 30_000, 100_000):
    o1, d1, a1, r1 = simulate(m, K, nd, 7, "fpae")
    o2, d2, a2, r2 = simulate(m, K, nd, 7, "cpa")
    print(f"  {nd:>8,} {o1:>7.1f} | {d1:>11.4f} {a1:>8.2%} {r1:>9.3f}"
          f" | {d2:>10.4f} {a2:>7.2%} {r2:>8.3f}")
