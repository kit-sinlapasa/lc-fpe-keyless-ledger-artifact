#!/usr/bin/env python3
"""
SCHEME-AWARE ADVERSARY vs FPAE.

Adversary knowledge (worst realistic case):
  * the scheme, K, and the allocation rule
  * the chart of accounts and its frequency profile f
    => therefore it can recompute k(a) EXACTLY (allocation is deterministic)
  * the observed per-code counts of one partition
Unknown: the alias map Phi and the FF1 permutation.

ATTACK 1 -- maximum-likelihood frequency matching.
  Assigning code with count c to account a has log-likelihood
      l(c,a) = c*log g(a) + (R-c)*log(1-g(a)),  g(a) = f(a)/k(a)
  l is supermodular in (c, g), so the ML assignment under the known group
  sizes k(a) is obtained by sorting codes by count and account-slots by g
  and matching in order  (O(n log n) instead of Hungarian O(n^3)).
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

def alloc_ceil(f, K):
    return np.ceil(f * K).astype(np.int64)


def ml_attack(f, k, R, trials=60):
    """Returns (per-alias accuracy, top10-alias accuracy, random baseline)."""
    Kp = int(k.sum())
    g = f / k
    owner = np.repeat(np.arange(len(f)), k)          # TRUE owner of each alias slot
    gal = np.repeat(g, k); gal = gal / gal.sum()
    # adversary's slot ordering: slots sorted by g descending
    slot_g = np.repeat(g, k)
    slot_owner_sorted = owner[np.lexsort((rng.random(Kp), -slot_g))]
    top10 = set(np.argsort(-f)[:10])

    acc, acc10 = 0.0, 0.0
    for _ in range(trials):
        c = rng.multinomial(R, gal)
        code_order = np.lexsort((rng.random(Kp), -c))     # codes by count desc
        # ML assignment: i-th highest-count code -> i-th highest-g slot
        guess = np.empty(Kp, dtype=np.int64)
        guess[code_order] = slot_owner_sorted
        acc += (guess == owner).mean()
        m10 = np.isin(owner, list(top10))
        acc10 += (guess[m10] == owner[m10]).mean()
    # random baseline respecting group sizes
    base = float((k ** 2).sum()) / Kp ** 2 * Kp / Kp
    base = float((k / Kp * k / Kp).sum())            # P(correct) for random perm
    base10 = float(sum((k[a] / Kp) for a in top10) * 0 + np.mean(
        [k[a] / Kp for a in top10]))
    return acc / trials, acc10 / trials, base, base10


print("=" * 84)
print("ATTACK 1 -- scheme-aware maximum-likelihood frequency matching")
print("=" * 84)
m, K = 300, 9700
f = zipf_chart(m, 1.0)
kh, kc = alloc_hamilton(f, K), alloc_ceil(f, K)
print(f"  chart: m={m}, Zipf(1.0), K={K}")
print()
print(f"  {'R':>11} | {'alloc':>9} | {'alias acc':>10} {'random':>9} {'lift':>7} |"
      f" {'top10 acc':>10} {'random':>9} {'lift':>7}")
for R in (10_000, 100_000, 1_000_000, 10_000_000):
    for name, k in (("hamilton", kh), ("ceiling", kc)):
        a, a10, b, b10 = ml_attack(f, k, R)
        print(f"  {R:>11,} | {name:>9} | {a:>9.4%} {b:>8.4%} "
              f"{a/b:>6.1f}x | {a10:>9.4%} {b10:>8.4%} {a10/b10:>6.1f}x")
    print()

print("=" * 84)
print("INTERPRETATION")
print("=" * 84)
print("  'random' = accuracy of a random assignment that respects the known")
print("  group sizes k(a).  lift = attack / random.  lift ~ 1.0x means the")
print("  adversary gains NOTHING from the observed counts.")
