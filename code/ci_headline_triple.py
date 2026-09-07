#!/usr/bin/env python3
"""
Paper A, A-P0.7: the headline clustering table under the DEPLOYED observable.

Every clustering figure in the paper (tab:ari, fig:ari, tab:sens, the chart
size and drift sweeps) treats the ALIAS as the node identity. What a deployed
adversary holds is the FF1 output on the lifted triple; within one
cryptographic partition FF1 is a fixed permutation of the lifted domain, so
two rows carry the same ciphertext triple exactly when their aliased triples
(a_hat, c, d) coincide. The deployed observable is therefore the equality
pattern of (alias, cost centre, document type), which splits every alias into
as many nodes as (c, d) combinations it appears with.

tab:obs measured that split once, on a small chart (m=100, K=2000, two
seeds). The reviews asked for the headline table itself to be rerun under the
deployed observable, with the alias-only numbers relabelled as an oracle upper
bound. This script does that:

  * same worlds, same document draws and same attack code as tab:ari
    (World(seed, m=300, s=1.0, n_types=400) and docs_for_occupancy), so the
    two observables are PAIRED per seed;
  * both adversary configurations of tab:ari: weak (MiniBatch k-means, 32
    dimensions, one refinement) and strong (full k-means, 64 dimensions,
    three refinements);
  * for each cell: ARI under the alias oracle, ARI under the triple, the
    paired difference with its 95% t-interval, and the verdict against the
    0.05 budget read on the triple's upper end.

The ARI is always scored against the ACCOUNT that owns the node, so the two
observables are compared on the same question: how much of the chart does the
adversary recover.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from harness import World, ci95, N_CC, N_DT

SEEDS = [11, 22, 33, 44, 55]
BUDGET = 0.05


def attack(w, u, v, n_nodes, owner, dims, refine, fast):
    """Spectral embedding + k-means + refinement on an arbitrary node set;
    identical to ci_decisive.attack except that the node count and the
    node -> account map are parameters instead of w.Kp and w.owner."""
    C = sp.coo_matrix((np.ones(2 * len(u)),
                       (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(n_nodes, n_nodes)).tocsr()
    C.data = np.log1p(C.data)
    idx = np.nonzero(np.asarray((C != 0).sum(axis=1)).ravel() > 0)[0]
    if len(idx) < w.m + 2:
        return float("nan")
    Cs = C[idx][:, idx]
    U, S, _ = svds(Cs.asfptype(), k=min(dims, len(idx) - 1), random_state=w.seed)
    X = U * S
    nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1
    KM = ((lambda k: MiniBatchKMeans(k, n_init=3, random_state=w.seed,
                                     batch_size=1024))
          if fast else (lambda k: KMeans(k, n_init=2, random_state=w.seed)))
    lab = KM(w.m).fit_predict(X / nn)
    for _ in range(refine):
        H = sp.coo_matrix((np.ones(len(idx)), (np.arange(len(idx)), lab)),
                          shape=(len(idx), w.m)).tocsr()
        Q = np.asarray((Cs @ H).todense())
        nn = np.linalg.norm(Q, axis=1, keepdims=True); nn[nn == 0] = 1
        lab = KM(w.m).fit_predict(Q / nn)
    return float(adjusted_rand_score(owner[idx], lab))


def triple_nodes(w, d_al, c_al, d_cc, d_dt, c_cc, c_dt):
    """Collapse (alias, c, d) to one node id per distinct triple; the node's
    owner is the account of its alias. This is the ciphertext equality
    pattern of one partition under one FF1 tweak."""
    td = (d_al * N_CC + d_cc) * N_DT + d_dt
    tc = (c_al * N_CC + c_cc) * N_DT + c_dt
    uniq, inv = np.unique(np.concatenate([td, tc]), return_inverse=True)
    nd = len(td)
    u, v = inv[:nd], inv[nd:]
    owner = np.empty(len(uniq), dtype=np.int64)
    owner[u] = w.owner[d_al]
    owner[v] = w.owner[c_al]
    return u, v, len(uniq), owner


def cell(occ, dims, refine, fast, seeds):
    a_alias, a_trip, n_trip = [], [], []
    for sd in seeds:
        w = World(sd, m=300, s=1.0, n_types=400)
        _, _, d_al, c_al, d_cc, d_dt, c_cc, c_dt = w.docs_for_occupancy(occ)
        a_alias.append(attack(w, d_al, c_al, w.Kp, w.owner, dims, refine, fast))
        u, v, n, owner = triple_nodes(w, d_al, c_al, d_cc, d_dt, c_cc, c_dt)
        n_trip.append(n)
        a_trip.append(attack(w, u, v, n, owner, dims, refine, fast))
    return np.asarray(a_alias), np.asarray(a_trip), float(np.mean(n_trip))


CFGS = [("weak (MB,32,r1)", 32, 1, True),
        ("strong (KM,64,r3)", 64, 3, False)]
# (occupancy, config index, seeds) in increasing cost; the occupancy-103
# strong cell uses three seeds exactly as the tab:ari entry it pairs with
PLAN = [(0.5, 0, SEEDS), (1.0, 0, SEEDS), (2.0, 0, SEEDS), (4.0, 0, SEEDS),
        (1.0, 1, SEEDS), (2.0, 1, SEEDS),
        (16.0, 0, SEEDS), (103.0, 0, SEEDS), (103.0, 1, SEEDS[:3])]

if __name__ == "__main__":
    print("=" * 100, flush=True)
    print("PAPER A -- HEADLINE CLUSTERING TABLE UNDER THE DEPLOYED OBSERVABLE "
          "(m=300, K=9700, 400 tx types)", flush=True)
    print("=" * 100, flush=True)
    print("  alias  = oracle: node identity is the alias (what tab:ari measured)",
          flush=True)
    print("  triple = deployed: node identity is the (alias, c, d) equality class "
          "= ciphertext equality under one tweak", flush=True)
    print("  delta  = triple - alias, PAIRED over seeds (same worlds, same rows); "
          "budget {} read on the triple's upper end".format(BUDGET), flush=True)
    print(flush=True)
    print("  {:>5} {:<18} {:>2} | {:>18} {:>18} {:>20} | {:>8} {:>8}  {}".format(
        "occ", "attack", "n", "ARI alias (95%)", "ARI triple (95%)",
        "delta (paired 95%)", "nodes", "upper", "verdict"), flush=True)
    print("  " + "-" * 118, flush=True)
    for occ, ci, seeds in PLAN:
        label, dims, refine, fast = CFGS[ci]
        t0 = time.time()
        a, b, nt = cell(occ, dims, refine, fast, seeds)
        ma, ha, na = ci95(a)
        mb, hb, nb = ci95(b)
        md, hd, ndl = ci95(b - a)
        up = mb + (hb if np.isfinite(hb) else 0.0)
        verdict = "safe" if up < BUDGET else "over budget"
        print("  {:>5.1f} {:<18} {:>2} | {:>8.4f} +/- {:.4f} {:>8.4f} +/- {:.4f} "
              "{:>+9.4f} +/- {:.4f} | {:>8.0f} {:>8.4f}  {}   [{:.0f}s]".format(
                  occ, label, len(seeds), ma, ha, mb, hb, md, hd, nt, up, verdict,
                  time.time() - t0), flush=True)
        print("        per-seed alias : " + " ".join("{:.4f}".format(x) for x in a),
              flush=True)
        print("        per-seed triple: " + " ".join("{:.4f}".format(x) for x in b),
              flush=True)

    print(flush=True)
    print("  negative delta in every cell => the alias-only tables are an oracle "
          "upper bound on the deployed adversary", flush=True)
    print("done", flush=True)
