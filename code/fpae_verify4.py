#!/usr/bin/env python3
"""
Round 4 -- two corrections to round 3:
  BUG   argsort breaks ties in index order; aliases are laid out grouped by
        account with the top accounts first, so at low R (mostly count-1 ties)
        the attack was handed free signal.  Fix: random tie-breaking.
  CHECK round 3 showed FPAE-hamilton scoring 1.9% at R=1e7 -- FAR BELOW the
        46% chance line.  Scoring below chance is not security: it means the
        signal is ANTI-correlated and an adversary who inverts the attack
        wins.  Test the inverted attack.
"""
import numpy as np
rng = np.random.default_rng(20260903)

def zipf_chart(m, s):
    w = 1.0 / np.arange(1, m + 1) ** s
    return w / w.sum()

def alloc_ceil(f, K):
    return np.ceil(f * K).astype(np.int64)

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

def rank_random_ties(c):
    """argsort descending with ties broken uniformly at random."""
    return np.lexsort((rng.random(len(c)), -c))

def attack(f, k, R, t, mode="top", trials=400):
    gal = np.repeat(f / k, k); gal = gal / gal.sum()
    owner = np.repeat(np.arange(len(f)), k)
    top = set(np.argsort(-f)[:t]); hit = 0.0
    for _ in range(trials):
        c = rng.multinomial(R, gal)
        order = rank_random_ties(c)
        picks = order[:t] if mode == "top" else order[-t:]
        hit += np.mean([owner[q] in top for q in picks])
    return hit / trials

def attack_no_fpae(f, R, t, trials=400):
    hit = 0.0
    for _ in range(trials):
        c = rng.multinomial(R, f)
        perm = rng.permutation(len(f))
        obs = c[perm]
        guess = set(rank_random_ties(obs)[:t])
        true = set(np.argsort(perm)[np.argsort(-f)[:t]])
        hit += len(guess & true) / t
    return hit / trials

f = zipf_chart(300, 1.0); K = 9700; t = 10
kc, kh = alloc_ceil(f, K), alloc_hamilton(f, K)
ch_c = kc[np.argsort(-f)[:t]].sum() / kc.sum()
ch_h = kh[np.argsort(-f)[:t]].sum() / kh.sum()

print("=" * 82)
print("TOP-10 ACCOUNT IDENTIFICATION  (random tie-breaking; chance line shown)")
print("=" * 82)
print(f"  chance = P(random alias belongs to a top-10 account):  "
      f"ceil {ch_c:.1%}   hamilton {ch_h:.1%}")
print()
print(f"  {'R':>11} {'no FPAE':>9} | {'ceil-top':>9} {'ceil-bot':>9} | "
      f"{'ham-top':>9} {'ham-bot':>9} | {'best adversary':>15}")
for R in (1_000, 10_000, 100_000, 1_000_000, 10_000_000):
    a0 = attack_no_fpae(f, R, t)
    ct, cb = attack(f, kc, R, t, "top"), attack(f, kc, R, t, "bot")
    ht, hb = attack(f, kh, R, t, "top"), attack(f, kh, R, t, "bot")
    best_c, best_h = max(ct, cb), max(ht, hb)
    print(f"  {R:>11,} {a0:>8.1%} | {ct:>8.1%} {cb:>8.1%} | "
          f"{ht:>8.1%} {hb:>8.1%} | ceil {best_c:>5.1%} / ham {best_h:>5.1%}")

print()
print("=" * 82)
print("WHY:  sign of the per-alias deviation vs account frequency")
print("=" * 82)
for name, k in (("ceiling", kc), ("hamilton", kh)):
    Kp = int(k.sum()); g = f / k; dev = g - 1.0 / Kp
    print(f"  {name:>9}:  K'={Kp:>5}  corr(f, g-1/K') = {np.corrcoef(f,dev)[0,1]:+.3f}"
          f"   mean|dev| = {np.abs(dev).mean():.3e}")
    hi = np.argsort(-f)[:5]; lo = np.argsort(-f)[-5:]
    print(f"             top-5 accounts  dev = "
          f"{', '.join(f'{d:+.2e}' for d in dev[hi])}")
    print(f"             bottom-5        dev = "
          f"{', '.join(f'{d:+.2e}' for d in dev[lo])}")
