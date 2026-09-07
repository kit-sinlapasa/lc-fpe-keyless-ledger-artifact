#!/usr/bin/env python3
"""
E. Multi-line vouchers.

Every experiment so far used two-line entries: one debit, one credit, one
co-occurrence edge per document.  Real vouchers are not like that.  A purchase
invoice posts several expense lines, input VAT, and one payable; a payroll run
posts salary, social security and withholding against one bank credit.

This matters because an n-line document contributes C(n,2) edges, not one.
At a FIXED occupancy (rows per alias) the number of documents falls as 1/n but
the edge count rises:

    edges = D * C(n,2) = (R/n) * n(n-1)/2 = R(n-1)/2

so moving from n=2 to a realistic mix roughly triples or quadruples the
adversary's evidence for the same number of journal rows.  We predicted the
occupancy threshold would fall; this measures by how much.

Line-count distribution is anchored on one hub side (the payment or the
receivable) with the remaining lines drawn from the archetype's other category,
which is how vouchers are actually built.
"""
import numpy as np
import scipy.sparse as sp
from itertools import combinations
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from harness import ci95
from harness_tfrs import TFRSWorld

LINE_DIST = {2: 0.45, 3: 0.25, 4: 0.15, 5: 0.10, 6: 0.05}


def gen_multiline(w, occ, mean_lines_dist):
    """Returns (u, v) alias-edge lists over all within-document pairs."""
    r = w.rng
    ns = np.array(list(mean_lines_dist.keys()))
    ps = np.array(list(mean_lines_dist.values()), dtype=float)
    ps = ps / ps.sum()
    mean_n = float((ns * ps).sum())
    target_rows = occ * w.Kp
    n_docs = max(1, int(target_rows / mean_n))

    U, V = [], []
    rows = 0
    t = r.choice(len(w.arch), size=n_docs, p=w.arch_w)
    nl = r.choice(ns, size=n_docs, p=ps)
    for i in range(n_docs):
        ds, cs, _ = w.arch[int(t[i])]
        n = int(nl[i])
        # the hub side stays a single line; the other side is split
        if ds[0] == "hub":
            anchor, spread, k_spread = ds, cs, n - 1
        elif cs[0] == "hub":
            anchor, spread, k_spread = cs, ds, n - 1
        else:
            anchor, spread, k_spread = ds, cs, n - 1
        a_acc = int(w._pick(anchor, 1)[0])
        s_acc = w._pick(spread, max(1, k_spread))
        accts = np.concatenate([[a_acc], s_acc])
        al = w.start[accts] + r.integers(0, w.k[accts])
        rows += len(al)
        for x, y in combinations(al.tolist(), 2):
            U.append(x); V.append(y)
    return np.array(U), np.array(V), rows, n_docs, mean_n


def attack(w, u, v, dims=64, refine=3):
    n = w.Kp
    C = sp.coo_matrix((np.ones(2 * len(u)),
                       (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(n, n)).tocsr()
    C.data = np.log1p(C.data)
    idx = np.nonzero(np.asarray((C != 0).sum(axis=1)).ravel() > 0)[0]
    if len(idx) < w.m + 2:
        return float("nan")
    Cs = C[idx][:, idx]
    U, S, _ = svds(Cs.asfptype(), k=min(dims, len(idx) - 1), random_state=w.seed)
    X = U * S
    nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    KM = lambda k: KMeans(k, n_init=2, random_state=w.seed)
    lab = KM(w.m).fit_predict(X / nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(len(idx)), (np.arange(len(idx)), lab)),
                          shape=(len(idx), w.m)).tocsr()
        Q = np.asarray((Cs @ H).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KM(w.m).fit_predict(Q / nn)
    return float(adjusted_rand_score(w.owner[idx], lab))


SEEDS = [11, 22, 33]
print("=" * 90, flush=True)
print("E. MULTI-LINE VOUCHERS on the structured chart (m=300, K=9700)",
      flush=True)
print("=" * 90, flush=True)
mean_n = sum(k * v for k, v in LINE_DIST.items())
print("  line distribution {}  mean {:.2f} lines/voucher".format(
    LINE_DIST, mean_n), flush=True)
print(flush=True)
print("  {:>5} {:<14} {:>9} {:>10} {:>22} {:>8}".format(
    "occ", "voucher size", "docs", "edges", "ARI (95% CI)", "upper"), flush=True)

for occ in (1, 2, 4, 8, 16, 24):
    for label, dist in (("2-line", {2: 1.0}), ("realistic mix", LINE_DIST)):
        A, E, D = [], [], []
        for sd in SEEDS:
            w = TFRSWorld(sd, m=300)
            u, v, rows, nd, mn = gen_multiline(w, occ, dist)
            E.append(len(u)); D.append(nd)
            A.append(attack(w, u, v))
        m, h, _ = ci95(A)
        up = m + h
        flag = "" if up < 0.10 else "  <== OVER BUDGET"
        print("  {:>5} {:<14} {:>9,.0f} {:>10,.0f} {:>12.4f} +/- {:.4f} {:>8.4f}{}"
              .format(occ, label, np.mean(D), np.mean(E), m, h, up, flag),
              flush=True)
    print(flush=True)
print("done", flush=True)
