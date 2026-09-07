#!/usr/bin/env python3
"""
F. Ablation.

LC-FPE stacks two mechanisms on the account code:

    DOMAIN LIFTING  encrypt <a,c,d> jointly  -- motivated by the 10^6 FF1 floor
    FPAE            expand a into k(a) frequency-proportional aliases

Which one buys security?  The four cells differ only in what equality class one
row exposes.  FF1 under a fixed tweak is a deterministic bijection, so it only
relabels; the adversary works on the equality pattern and we may measure on the
plaintext identities directly.

    cell      node identity        nodes           true owner
    plain     a                    m               itself
    lift      (a, c, d)            m * n_cc * n_dt a
    FPAE      a-hat                K'              a
    LC-FPE    (a-hat, c, d)        <= K'*n_cc*n_dt a

TWO attacks, because they bind different cells (the same discipline as task C,
where adversary strength was non-monotone in adversary complexity):

  1. CO-OCCURRENCE CLUSTERING, uniform across all four cells: build the graph on
     observable equality classes, cluster into m groups, score ARI against the
     account that owns each class.  For 'plain' every node is its own account so
     ARI is 1.0 -- a genuine total break, not a degenerate metric.

  2. FREQUENCY MATCHING with GENERATIVE aux: the adversary holds the chart's
     true frequency vector w.f (a property of the business, independent of the
     sampled rows -- NOT the empirical histogram of the very rows under attack).
     Score: how many of the ten most frequent observable classes belong to the
     ten most frequent accounts.  FPAE flattens the class histogram by
     construction, so this attack is expected to die on the aliased cells; the
     question is whether hub accounts leak anyway through owning most aliases.
"""
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from harness import ci95
from harness_tfrs import TFRSWorld

SEEDS = [11, 22, 33]
CELLS = ("plain", "lift", "FPAE", "LC-FPE")


def build(w, occ):
    """One world's rows, as the four observable node-id streams plus owners."""
    da, ca, d_al, c_al, d_cc, d_dt, c_cc, c_dt = w.docs_for_occupancy(occ)
    streams = {
        "plain":  (da, ca),
        "lift":   (list(zip(da.tolist(), d_cc.tolist(), d_dt.tolist())),
                   list(zip(ca.tolist(), c_cc.tolist(), c_dt.tolist()))),
        "FPAE":   (d_al, c_al),
        "LC-FPE": (list(zip(d_al.tolist(), d_cc.tolist(), d_dt.tolist())),
                   list(zip(c_al.tolist(), c_cc.tolist(), c_dt.tolist()))),
    }
    out = {}
    for name, (ds, cs) in streams.items():
        ids, owner = {}, []
        def nid(tok, acct):
            if tok not in ids:
                ids[tok] = len(ids); owner.append(int(acct))
            return ids[tok]
        u = [nid(t, a) for t, a in zip(list(ds), da.tolist())]
        v = [nid(t, a) for t, a in zip(list(cs), ca.tolist())]
        out[name] = (np.array(u), np.array(v), np.array(owner))
    return out


def cluster(w, u, v, owner, dims=64, refine=3):
    n = len(owner)
    if n <= w.m + 2:                      # fewer classes than accounts
        return float(adjusted_rand_score(owner, np.arange(n)))
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
    return float(adjusted_rand_score(owner[idx], lab))


def freq_top(w, u, v, owner, t=10):
    """Of the t most frequent observable classes, how many belong to accounts in
       the top t of the chart's TRUE frequency vector?  Aux is generative."""
    cnt = np.bincount(np.concatenate([u, v]), minlength=len(owner))
    top_nodes = np.argsort(-cnt)[:t]
    top_accts = set(np.argsort(-w.f)[:t].tolist())
    return sum(1 for x in top_nodes if owner[x] in top_accts) / t


def row(w, occ):
    b = build(w, occ)
    return {c: (cluster(w, *b[c]), freq_top(w, *b[c]), len(b[c][2]))
            for c in CELLS}


if __name__ == "__main__":
    print("=" * 92, flush=True)
    print("F. ABLATION -- what does each mechanism buy?", flush=True)
    print("=" * 92, flush=True)
    print("  structured TFRS chart, m=200, three seeds, correlation 5x3", flush=True)

    for occ in (1.0, 4.0):
        res = [row(TFRSWorld(s, m=200), occ) for s in SEEDS]
        print(flush=True)
        print("  occupancy {:g}".format(occ), flush=True)
        print("  {:<8} {:>9} {:>22} {:>26}".format(
            "cell", "nodes", "clustering ARI", "freq top-10 hit rate"), flush=True)
        for c in CELLS:
            A = [r[c][0] for r in res]; F = [r[c][1] for r in res]
            am, ah, _ = ci95(A); fm, fh, _ = ci95(F)
            n = int(np.mean([r[c][2] for r in res]))
            print("  {:<8} {:>9,} {:>13.4f} +/- {:.4f} {:>17.0%} +/- {:.0%}".format(
                c, n, am, ah, fm, fh), flush=True)

    print(flush=True)
    print("done", flush=True)
