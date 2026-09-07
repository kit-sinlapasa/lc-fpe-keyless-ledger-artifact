#!/usr/bin/env python3
"""
Co-occurrence attack, round 2.
  FIX: AUC used argsort-ranks, which mishandles the mass of exact ties at
       similarity 0 that dominates the low-occupancy regime.  Use average
       ranks (proper Mann-Whitney).  The AUC=0.069 row in round 1 was a
       tie-handling artifact and should have been ~0.5.
  GOAL: locate the occupancy threshold where re-merging becomes feasible.
        This is the number Claim 5 (partition granularity) needs.
"""
import numpy as np
rng = np.random.default_rng(20260903)

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

def avg_ranks(x):
    """average ranks, ties shared (proper Mann-Whitney)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r

def auc_mw(same, diff):
    allv = np.concatenate([same, diff])
    r = avg_ranks(allv)
    n1 = len(same)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(diff))

def experiment(m, K, n_docs, n_types=400, n_probe=1200, D=512):
    dr = rng.integers(0, m, n_types); cr = rng.integers(0, m, n_types)
    bad = dr == cr; cr[bad] = (cr[bad] + 1) % m
    wt = 1.0 / np.arange(1, n_types + 1) ** 1.0; wt /= wt.sum()
    cnt = np.zeros(m); np.add.at(cnt, dr, wt); np.add.at(cnt, cr, wt)
    f = cnt / cnt.sum()
    k = np.ones(m, dtype=np.int64)
    u = f > 0
    k[u] = alloc_hamilton(f[u], K)
    Kp = int(k.sum()); start = np.concatenate([[0], np.cumsum(k)[:-1]])
    owner = np.repeat(np.arange(m), k)

    t = rng.choice(n_types, size=n_docs, p=wt)
    d_acc, c_acc = dr[t], cr[t]
    d_al = start[d_acc] + rng.integers(0, k[d_acc])
    c_al = start[c_acc] + rng.integers(0, k[c_acc])

    h = rng.integers(0, D, Kp)
    P = np.zeros((Kp, D))
    np.add.at(P, (d_al, h[c_al]), 1.0)
    np.add.at(P, (c_al, h[d_al]), 1.0)
    nrm = np.linalg.norm(P, axis=1); nz = nrm > 0
    Pn = np.zeros_like(P); Pn[nz] = P[nz] / nrm[nz, None]

    cand = np.where(nz & (k[owner] >= 2))[0]
    same, diff = [], []
    for _ in range(n_probe):
        i = cand[rng.integers(len(cand))]
        sib = np.arange(start[owner[i]], start[owner[i]] + k[owner[i]])
        sib = sib[(sib != i) & nz[sib]]
        if len(sib) == 0:
            continue
        same.append(float(Pn[i] @ Pn[sib[rng.integers(len(sib))]]))
        while True:
            q = cand[rng.integers(len(cand))]
            if owner[q] != owner[i]:
                break
        diff.append(float(Pn[i] @ Pn[q]))
    same, diff = np.array(same), np.array(diff)
    return auc_mw(same, diff), (2 * n_docs) / Kp, Kp, same.mean(), diff.mean()


print("=" * 86)
print("CO-OCCURRENCE RE-MERGING vs occupancy  (avg-rank AUC; 0.5 = no signal)")
print("=" * 86)
print(f"  {'docs':>11} {'lines/alias':>12} {'AUC':>7} {'sim same':>9} "
      f"{'sim diff':>9}  verdict")
grid = [2_000, 5_000, 10_000, 20_000, 35_000, 50_000, 100_000,
        200_000, 500_000, 2_000_000]
rows = []
for nd in grid:
    auc, occ, Kp, ms, md = experiment(300, 9700, nd)
    v = ("no signal" if auc < 0.55 else "weak" if auc < 0.70 else
         "SIGNAL" if auc < 0.90 else "BROKEN")
    rows.append((occ, auc))
    print(f"  {nd:>11,} {occ:>12.2f} {auc:>7.3f} {ms:>9.4f} {md:>9.4f}  {v}")

print()
print("=" * 86)
print("THRESHOLD  (linear interpolation on the occupancy axis)")
print("=" * 86)
for target in (0.55, 0.60, 0.70, 0.90):
    hit = None
    for (o1, a1), (o2, a2) in zip(rows, rows[1:]):
        if a1 < target <= a2:
            hit = o1 + (target - a1) * (o2 - o1) / (a2 - a1)
            break
    print(f"  AUC = {target:.2f}  ->  occupancy ~ "
          f"{hit:.1f} lines/alias" if hit else
          f"  AUC = {target:.2f}  ->  not reached in the tested range")

print()
print("  DESIGN RULE:  keep  R / K'  below the AUC=0.55 occupancy")
print("  With K' = 9700 that caps rows per partition at roughly R < 10^4.5")
