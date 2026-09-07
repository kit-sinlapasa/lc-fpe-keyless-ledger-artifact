#!/usr/bin/env python3
"""
FIX to attack 1's top-10 baseline, plus ATTACK 2 (co-occurrence re-merging).

Baseline fix: acc10 averages over ALIASES of top-10 accounts, so the
random-assignment baseline must be alias-weighted:
      base10 = sum_{a in top10} k(a)^2  /  ( K' * sum_{a in top10} k(a) )
(the previous script used the unweighted mean of k(a)/K', which is wrong.)

ATTACK 2: double-entry ledgers pair accounts.  Two aliases of the SAME
account should co-occur with the same counter-account mix.  The adversary
builds each alias's co-occurrence profile and tests whether same-account
pairs are more similar than different-account pairs (AUC of a same-account
classifier).  AUC 0.5 = no signal;  AUC 1.0 = perfect re-merging.
"""
import numpy as np
rng = np.random.default_rng(20260903)

def zipf_chart(m, s):
    w = 1.0 / np.arange(1, m + 1) ** s
    return w / w.sum()

def alloc_hamilton(f, K):
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


# ------------------------------------------------------------ attack 1 (fixed)
def ml_attack(f, k, R, trials=60):
    Kp = int(k.sum()); g = f / k
    owner = np.repeat(np.arange(len(f)), k)
    gal = np.repeat(g, k); gal = gal / gal.sum()
    slot_g = np.repeat(g, k)
    slot_owner_sorted = owner[np.lexsort((rng.random(Kp), -slot_g))]
    top10 = np.argsort(-f)[:10]; mtop = np.isin(owner, top10)
    acc = acc10 = 0.0
    for _ in range(trials):
        c = rng.multinomial(R, gal)
        order = np.lexsort((rng.random(Kp), -c))
        guess = np.empty(Kp, dtype=np.int64); guess[order] = slot_owner_sorted
        acc += (guess == owner).mean()
        acc10 += (guess[mtop] == owner[mtop]).mean()
    base = float((k.astype(float) ** 2).sum()) / Kp ** 2
    kt = k[top10].astype(float)
    base10 = float((kt ** 2).sum() / (Kp * kt.sum()))
    return acc / trials, acc10 / trials, base, base10


print("=" * 84)
print("ATTACK 1 (corrected baseline) -- ML frequency matching, Hamilton alloc")
print("=" * 84)
m, K = 300, 9700
f = zipf_chart(m, 1.0); kh = alloc_hamilton(f, K)
print(f"  {'R':>11} | {'alias acc':>10} {'random':>9} {'lift':>6} |"
      f" {'top10 acc':>10} {'random':>9} {'lift':>6}")
for R in (10_000, 100_000, 1_000_000, 10_000_000):
    a, a10, b, b10 = ml_attack(f, kh, R)
    print(f"  {R:>11,} | {a:>9.4%} {b:>8.4%} {a/b:>5.2f}x |"
          f" {a10:>9.4%} {b10:>8.4%} {a10/b10:>5.2f}x")


# ------------------------------------------------------------ attack 2
def make_pairing(m, n_types, rng):
    """Sparse debit/credit pairing: each transaction type picks (dr, cr)."""
    dr = rng.integers(0, m, n_types)
    cr = rng.integers(0, m, n_types)
    bad = dr == cr
    cr[bad] = (cr[bad] + 1) % m
    w = 1.0 / np.arange(1, n_types + 1) ** 1.0
    return dr, cr, w / w.sum()


def cooc_attack(m, K, n_docs, s=1.0, n_types=400, n_probe=600):
    """Build a real double-entry ledger, alias it, then test whether an
       adversary can tell same-account alias pairs from different-account
       pairs using co-occurrence profiles."""
    dr, cr, wt = make_pairing(m, n_types, rng)
    # realised account frequency implied by the pairing model
    cnt = np.zeros(m)
    np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    used = f > 0
    # allocate only over used accounts
    k = np.ones(m, dtype=np.int64)
    k[used] = alloc_hamilton(f[used], K)
    Kp = int(k.sum())
    start = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)

    # generate documents
    t = rng.choice(n_types, size=n_docs, p=wt)
    d_acc, c_acc = dr[t], cr[t]
    # alias index = PRF(doc, line) -> uniform over k(a)
    d_al = start[d_acc] + rng.integers(0, k[d_acc])
    c_al = start[c_acc] + rng.integers(0, k[c_acc])

    # co-occurrence profiles (sparse, as dict of counters via bincount on pairs)
    prof = np.zeros((Kp, 0))
    # build with a hashed feature space to stay tractable
    D = 512
    h = rng.integers(0, D, Kp)
    P = np.zeros((Kp, D))
    np.add.at(P, (d_al, h[c_al]), 1.0)
    np.add.at(P, (c_al, h[d_al]), 1.0)
    nrm = np.linalg.norm(P, axis=1); nz = nrm > 0
    Pn = np.zeros_like(P); Pn[nz] = P[nz] / nrm[nz, None]

    # sample same-account and different-account alias pairs
    cand = np.where(nz & (k[owner] >= 2))[0]
    if len(cand) < 10:
        return None
    same, diff = [], []
    for _ in range(n_probe):
        i = cand[rng.integers(len(cand))]
        sib = np.arange(start[owner[i]], start[owner[i]] + k[owner[i]])
        sib = sib[(sib != i) & nz[sib]]
        if len(sib) == 0:
            continue
        j = sib[rng.integers(len(sib))]
        same.append(float(Pn[i] @ Pn[j]))
        while True:
            q = cand[rng.integers(len(cand))]
            if owner[q] != owner[i]:
                break
        diff.append(float(Pn[i] @ Pn[q]))
    same, diff = np.array(same), np.array(diff)
    # AUC via Mann-Whitney
    allv = np.concatenate([same, diff])
    r = allv.argsort().argsort() + 1
    n1 = len(same)
    auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(diff))
    occ = (2 * n_docs) / Kp
    return auc, occ, Kp, same.mean(), diff.mean()


print()
print("=" * 84)
print("ATTACK 2 -- co-occurrence re-merging  (AUC 0.5 = no signal, 1.0 = broken)")
print("=" * 84)
print(f"  {'docs':>12} {'lines/alias':>12} {'AUC':>8} {'sim(same)':>10} "
      f"{'sim(diff)':>10}  verdict")
for nd in (5_000, 50_000, 500_000, 5_000_000):
    out = cooc_attack(300, 9700, nd)
    if out is None:
        continue
    auc, occ, Kp, ms, md = out
    verdict = ("no signal" if auc < 0.55 else
               "weak signal" if auc < 0.70 else
               "SIGNAL" if auc < 0.85 else "BROKEN")
    print(f"  {nd:>12,} {occ:>12.2f} {auc:>8.3f} {ms:>10.4f} {md:>10.4f}  {verdict}")
