#!/usr/bin/env python3
"""
Round 3 -- diagnose WHY deviation is non-monotonic in f, and fix it.

Hypothesis from round 2: the dominant deviation term is NOT the rounding
residual but the systematic mismatch between K (allocation scale) and
K' = sum k(a) (actual number of aliases).  Every alias sits at ~1/K while
the uniform reference is 1/K'.

Exact decomposition:   let n = k(a) = ceil(f K),  rho = n - fK  in [0,1)
     1/K - g(a) = rho / (K n)                      <- rounding term
     1/K' - 1/K = -(K'-K) / (K K')                 <- scale-mismatch term

Fix under test: largest-remainder (Hamilton) apportionment, which makes
sum k(a) = K exactly and directly minimises  Delta = TV(f, k/K').
"""
import numpy as np
from math import sqrt

rng = np.random.default_rng(20260903)

def zipf_chart(m, s):
    w = 1.0 / np.arange(1, m + 1) ** s
    return w / w.sum()

def alloc_ceil(f, K):
    return np.ceil(f * K).astype(np.int64)

def alloc_hamilton(f, K):
    """Largest-remainder apportionment with a floor of 1 alias per account."""
    q = f * K
    base = np.floor(q).astype(np.int64)
    base = np.maximum(base, 1)                 # every used account needs >=1 alias
    short = K - base.sum()
    if short > 0:                              # hand out the remainder
        rem = q - np.floor(q)
        order = np.argsort(-rem)
        base[order[:short]] += 1
    elif short < 0:                            # over-allocated: claw back
        order = np.argsort(q - base)           # take from the least deserving
        i = 0
        while short < 0 and i < len(order):
            j = order[i]
            if base[j] > 1:
                base[j] -= 1
                short += 1
            i += 1
    return base

def delta_of(f, k):
    Kp = int(k.sum())
    return 0.5 * np.sum(np.abs(f - k / Kp)), Kp


f = zipf_chart(300, 1.0); K = 9700
kc = alloc_ceil(f, K); kh = alloc_hamilton(f, K)

print("=" * 78)
print("1. EXACT DECOMPOSITION of |g - 1/K'|")
print("=" * 78)
Kp = int(kc.sum()); g = f / kc; p = 1.0 / Kp
rho = kc - f * K
term_round = rho / (K * kc)
term_scale = (Kp - K) / (K * Kp)
pred = np.abs(term_scale - term_round)
act = np.abs(g - p)
print(f"  K = {K}   K' = sum k = {Kp}   (overhead {Kp-K})")
print(f"  scale-mismatch term  (K'-K)/(K K') = {term_scale:.3e}")
print(f"  rounding term  rho/(K k)  range    = [{term_round.min():.3e}, "
      f"{term_round.max():.3e}]")
print(f"  => scale term is {term_scale/term_round.max():.0f}x larger than the "
      f"largest rounding term")
print(f"  max |predicted - actual| = {np.abs(pred-act).max():.3e}   "
      f"-> decomposition {'EXACT' if np.abs(pred-act).max() < 1e-15 else 'approximate'}")

print()
print("=" * 78)
print("2. FIX: largest-remainder (Hamilton) vs ceiling allocation")
print("=" * 78)
print(f"  {'m':>5} {'s':>5} {'K':>6} | {'K_ceil':>7} {'D_ceil':>9} | "
      f"{'K_ham':>7} {'D_ham':>9} | {'improve':>8}")
for m in (300, 500):
    for s in (0.8, 1.0, 1.3):
        ff = zipf_chart(m, s)
        for KK in (2000, 9700):
            a, Ka = delta_of(ff, alloc_ceil(ff, KK))
            b, Kb = delta_of(ff, alloc_hamilton(ff, KK))
            print(f"  {m:>5} {s:>5.1f} {KK:>6} | {Ka:>7} {a:>9.6f} | "
                  f"{Kb:>7} {b:>9.6f} | {a/b:>7.1f}x")

print()
print("=" * 78)
print("3. END-TO-END ATTACK with the fixed allocation")
print("=" * 78)

def attack_none(f, R, t, trials=300):
    hit = 0
    for _ in range(trials):
        c = rng.multinomial(R, f)
        perm = rng.permutation(len(f))
        obs = c[perm]
        guess = set(np.argsort(-obs)[:t])
        true = set(np.argsort(perm)[np.argsort(-f)[:t]])
        hit += len(guess & true) / t
    return hit / trials

def attack_fpae(f, k, R, t, trials=300):
    gal = np.repeat(f / k, k); owner = np.repeat(np.arange(len(f)), k)
    top = set(np.argsort(-f)[:t]); hit = 0
    for _ in range(trials):
        c = rng.multinomial(R, gal / gal.sum())
        hit += np.mean([owner[q] in top for q in np.argsort(-c)[:t]])
    return hit / trials

t = 10
ch_c = kc[np.argsort(-f)[:t]].sum() / kc.sum()
ch_h = kh[np.argsort(-f)[:t]].sum() / kh.sum()
print(f"  top-{t} identification accuracy   (chance: ceil {ch_c:.1%}, "
      f"hamilton {ch_h:.1%})")
print(f"  {'R':>10} {'no FPAE':>10} {'FPAE-ceil':>11} {'FPAE-hamilton':>15}")
for R in (1_000, 10_000, 100_000, 1_000_000, 10_000_000):
    print(f"  {R:>10,} {attack_none(f,R,t):>9.1%} "
          f"{attack_fpae(f,kc,R,t):>10.1%} {attack_fpae(f,kh,R,t):>14.1%}")

print()
print("=" * 78)
print("4. residual signal at low R: does k(a) itself leak?")
print("=" * 78)
print("   at R << K' most aliases are unseen; an account with more aliases is")
print("   more likely to appear at all -> alias-count is itself a frequency proxy")
own_h = np.repeat(np.arange(len(f)), kh)
for R in (1_000, 10_000, 100_000):
    gal = np.repeat(f / kh, kh)
    c = rng.multinomial(R, gal / gal.sum())
    seen = c > 0
    frac_seen = np.array([seen[own_h == i].mean() for i in range(len(f))])
    corr = np.corrcoef(f, frac_seen)[0, 1]
    print(f"   R = {R:>9,}  aliases seen = {seen.sum():>5}/{len(gal)}   "
          f"corr(f, fraction-of-own-aliases-seen) = {corr:+.3f}")
