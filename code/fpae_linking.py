#!/usr/bin/env python3
"""
LINKING-TOKEN TRADE-OFF CURVE.

LC-FPE step 3 stores a cross-period linking token  T(a) = PRF(K_link, a) mod B
so the key holder can follow one account across periods (the FF1 tweak = (e,p)
otherwise re-permutes every code each period).

But the token is stored on every row, so a snapshot adversary READS IT TOO.
Two consequences, both worsening as B grows:
  (1) the token directly reveals which aliases share a bucket  ->  the
      clustering problem shrinks from m accounts to m/B accounts per bucket
  (2) it lets the adversary align statistics across partitions

Effect (1) dominates and is what we measure: run the clustering attack with
the bucket partition GIVEN to the adversary, sweeping B.

Utility side: grouping by token sums over the whole bucket, so cross-period
report precision ~ 1 / bucket_size.
  B = 1  -> no leakage, no cross-period reporting at all
  B = m  -> perfect reporting, token *is* the account identity = total break
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

def build(m, K, nd, seed, n_types=400):
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1); wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    k = alloc_hamilton(f, K, rng)
    Kp = int(k.sum()); st = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)
    t = rng.choice(n_types, size=nd, p=wt)
    da, ca = dr[t], cr[t]
    dal = st[da] + rng.integers(0, k[da]); cal = st[ca] + rng.integers(0, k[ca])
    C = sp.coo_matrix((np.ones(2 * nd), (np.concatenate([dal, cal]),
                       np.concatenate([cal, dal]))), shape=(Kp, Kp)).tocsr()
    C.data = np.log1p(C.data)
    return f, k, Kp, owner, C, rng

def attack(C, Kp, owner, m, buckets, alias_bucket, seed, dims=32):
    """cluster inside every bucket; adversary is handed the bucket labels."""
    correct = 0; labels = np.full(Kp, -1)
    nxt = 0
    for b in range(buckets):
        idx = np.where(alias_bucket == b)[0]
        accts = np.unique(owner[idx])
        if len(idx) < 2 or len(accts) < 2:
            labels[idx] = nxt; nxt += 1
            correct += len(idx) if len(accts) == 1 else 0
            continue
        sub = C[idx][:, :]
        d = min(dims, len(idx) - 1)
        try:
            U, S, _ = svds(sub.asfptype(), k=d)
        except Exception:
            labels[idx] = nxt; nxt += 1; continue
        X = U * S; n_ = np.linalg.norm(X, axis=1, keepdims=True); n_[n_ == 0] = 1
        lb = KMeans(len(accts), n_init=2, random_state=seed).fit_predict(X / n_)
        cf = np.zeros((len(accts), len(accts)))
        pos = {a: i for i, a in enumerate(accts)}
        np.add.at(cf, (lb, np.array([pos[o] for o in owner[idx]])), 1.0)
        r, c = linear_sum_assignment(-cf); mp = dict(zip(r, c))
        guess = np.array([accts[mp[l]] for l in lb])
        correct += int((guess == owner[idx]).sum())
        labels[idx] = nxt + lb; nxt += len(accts)
    return correct / Kp, adjusted_rand_score(owner, labels)

m, K, nd = 200, 2000, 10_000
f, k, Kp, owner, C, rng = build(m, K, nd, 7)
print("=" * 88, flush=True)
print(f"LINKING-TOKEN TRADE-OFF   m={m} accounts, K={K} aliases, "
      f"occupancy={2*nd/Kp:.1f}", flush=True)
print("=" * 88, flush=True)
print(f"  {'B (buckets)':>12} {'accts/bucket':>13} {'report prec':>12} |"
      f" {'recovery':>9} {'ARI':>7}  regime", flush=True)
for B in (1, 2, 4, 8, 16, 32, 64, 128, 200):
    ab_acct = rng.permutation(m) % B          # keyed bucket assignment
    alias_bucket = ab_acct[owner]
    acc, ari = attack(C, Kp, owner, m, B, alias_bucket, 7)
    per = m / B
    reg = ("safe" if ari < 0.10 else "leaking" if ari < 0.35
           else "SERIOUS" if ari < 0.70 else "BROKEN")
    print(f"  {B:>12} {per:>13.1f} {1/per:>11.1%} | {acc:>8.1%} {ari:>7.3f}  {reg}",
          flush=True)
