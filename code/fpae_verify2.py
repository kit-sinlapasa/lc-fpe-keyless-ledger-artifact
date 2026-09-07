#!/usr/bin/env python3
"""
FPAE verification, round 2:
 (A) are the Claim-4 disagreements boundary cases?
 (B) is detectability non-monotonic in f (rounding-residual oscillation)?
 (C) does empirical Delta from a sampled ledger converge to the analytic Delta?
 (D) END-TO-END ATTACK: can an adversary identify the top accounts,
     with FPAE vs without FPAE (plain deterministic FPE)?
"""
import numpy as np
from math import sqrt

rng = np.random.default_rng(20260903)

def zipf_chart(m, s):
    w = 1.0 / np.arange(1, m + 1) ** s
    return w / w.sum()

def allocate(f, K):
    return np.ceil(f * K).astype(np.int64)


# ---------------------------------------------------------- (A)+(B)
print("=" * 78)
print("(A) Claim-4 disagreements: are they boundary cases?")
print("=" * 78)
f = zipf_chart(300, 1.0); K = 9700; R = 1_000_000
k = allocate(f, K); Kp = int(k.sum()); g = f / k; p = 1.0 / Kp
thr = sqrt(1.0 / (Kp * R))
dev = np.abs(g - p)
pred = dev > thr
sd = sqrt(R * p * (1 - p))
gal = np.repeat(g, k); owner = np.repeat(np.arange(len(f)), k)
flags = np.zeros(len(gal))
T = 400
for _ in range(T):
    c = rng.multinomial(R, gal / gal.sum())
    flags += (np.abs(c - R * p) > sd)
rate = np.array([(flags / T)[owner == i].mean() for i in range(len(f))])
emp = rate > 0.5
dis = np.where(pred != emp)[0]
print(f"  threshold on |g-1/K'| : {thr:.3e}")
print(f"  disagreements          : {len(dis)} / {len(f)}")
if len(dis):
    ratios = dev[dis] / thr
    print(f"  their |g-1/K'| / thr   : min {ratios.min():.3f}  max {ratios.max():.3f}  "
          f"median {np.median(ratios):.3f}")
    print(f"  -> all within [{ratios.min():.2f}x, {ratios.max():.2f}x] of the "
          f"threshold => BOUNDARY CASES" if ratios.max() < 2 and ratios.min() > 0.5
          else "  -> NOT purely boundary; investigate")

print()
print("(B) is detectability monotonic in f?  (rounding residual r = ceil(fK)-fK)")
resid = np.ceil(f * K) - f * K
idx = np.argsort(-f)[:14]
print(f"  {'rank':>5} {'f':>10} {'resid':>7} {'|g-1/K|/thr':>12} {'detect%':>8}")
for i in idx:
    print(f"  {int(np.where(np.argsort(-f)==i)[0][0])+1:>5} {f[i]:>9.4%} "
          f"{resid[i]:>7.3f} {dev[i]/thr:>12.3f} {rate[i]:>7.1%}")
sp = np.corrcoef(f, dev)[0, 1]
print(f"  Pearson corr(f, deviation) = {sp:+.3f}   "
      f"=> {'monotonic' if abs(sp) > 0.8 else 'NON-monotonic: driven by rounding residual'}")


# ---------------------------------------------------------- (C)
print()
print("=" * 78)
print("(C) empirical Delta (sampled ledger) vs analytic Delta")
print("=" * 78)
d_an = 0.5 * np.sum(np.abs(f - k / Kp))
print(f"  analytic Delta = {d_an:.6f}   (bound m/K = {len(f)/K:.6f})")
for Rs in (10_000, 100_000, 1_000_000, 10_000_000):
    c = rng.multinomial(Rs, gal / gal.sum())
    d_emp = 0.5 * np.sum(np.abs(c / Rs - p))
    print(f"  R = {Rs:>10,}  empirical Delta = {d_emp:.6f}   "
          f"(sampling noise inflates small R)")


# ---------------------------------------------------------- (D)
print()
print("=" * 78)
print("(D) END-TO-END ATTACK  --  identify the TOP-t accounts by frequency")
print("=" * 78)
print("    adversary knows the true chart frequencies f (auxiliary information)")
print("    and observes the ciphertext-code histogram of one partition.")
print()

def attack_no_fpae(f, R, t, trials=200):
    """Plain deterministic FPE: one ciphertext code per account.
       Adversary ranks codes by count, matches to ranked f."""
    hit = 0
    perm = None
    for _ in range(trials):
        c = rng.multinomial(R, f)
        perm = rng.permutation(len(f))          # secret code assignment
        obs = c[perm]                            # what adversary sees, indexed by code
        guess_codes = np.argsort(-obs)[:t]       # top-t codes by count
        true_codes = np.argsort(perm)[np.argsort(-f)[:t]]
        hit += len(set(guess_codes) & set(true_codes)) / t
    return hit / trials

def attack_fpae(f, k, R, t, trials=200):
    """FPAE: adversary sees alias-level counts. Best strategy is still
       'most frequent codes belong to the most frequent accounts'.
       Score each account by the counts of ITS aliases -- but the adversary
       does not know the grouping, so it can only pick top-t single codes
       and ask whether they belong to the top-t accounts."""
    gal = np.repeat(f / k, k)
    owner = np.repeat(np.arange(len(f)), k)
    top = set(np.argsort(-f)[:t])
    hit = 0
    for _ in range(trials):
        c = rng.multinomial(R, gal / gal.sum())
        picks = np.argsort(-c)[:t]
        hit += np.mean([owner[q] in top for q in picks])
    return hit / trials

t = 10
print(f"  {'R':>10} {'no FPAE (top-10 acc.)':>24} {'with FPAE':>12} {'chance':>9}")
for R2 in (1_000, 10_000, 100_000, 1_000_000):
    a0 = attack_no_fpae(f, R2, t)
    a1 = attack_fpae(f, k, R2, t)
    # chance = P(a random alias belongs to a top-t account) = sum_{top} k(a)/K'
    ch = k[np.argsort(-f)[:t]].sum() / Kp
    print(f"  {R2:>10,} {a0:>23.1%} {a1:>11.1%} {ch:>8.1%}")
