#!/usr/bin/env python3
"""
LEG (a) -- measuring the CONDITIONAL-DOMAIN COLLAPSE, and whether FPAE fixes it.

The claim we have been asserting but never measured:
  domain lifting a x c x d reaches a nominal 10^9, but an adversary who knows
  the row's cost centre c and document type d sees a far smaller EFFECTIVE
  domain for the account code a -- and FPAE is supposed to repair this by
  injecting entropy that is independent of c and d.

Note on FF1: FF1 with a fixed tweak is a deterministic BIJECTION on the lifted
domain, so it only relabels tuples.  Recovery accuracy is therefore identical
with or without actually running it; we run real FF1 on a verification sample
to confirm the pipeline, and compute the statistics on the full ledger.

Realistic correlation model: each account is used with only a few cost centres
and document types (a cost centre belongs to a department; departments touch a
limited set of accounts).  n_cc controls the correlation strength.
"""
import math
import numpy as np
from lcfpe_impl import FF1, alloc_hamilton
import os

rng = np.random.default_rng(20260903)
N_CC, N_DT = 100, 10


def make_world(m, n_cc_per_acct, n_dt_per_acct):
    """which (c, d) each account may appear with"""
    cc_of = [rng.choice(N_CC, size=n_cc_per_acct, replace=False) for _ in range(m)]
    dt_of = [rng.choice(N_DT, size=n_dt_per_acct, replace=False) for _ in range(m)]
    return cc_of, dt_of


def gen_rows(m, f, cc_of, dt_of, N):
    a = rng.choice(m, size=N, p=f)
    c = np.array([cc_of[x][rng.integers(len(cc_of[x]))] for x in a])
    d = np.array([dt_of[x][rng.integers(len(dt_of[x]))] for x in a])
    return a, c, d


def eff_domain(a, c, d, m):
    """N_eff = average number of DISTINCT accounts consistent with an observed
       (c, d) pair, weighted by how often that pair occurs."""
    from collections import defaultdict
    seen = defaultdict(set)
    cnt = defaultdict(int)
    for i in range(len(a)):
        seen[(c[i], d[i])].add(a[i])
        cnt[(c[i], d[i])] += 1
    tot = sum(cnt.values())
    return sum(len(seen[key]) * cnt[key] for key in cnt) / tot


def attack(a, c, d, m, k=None, start=None, alias=None, use_cd=True):
    """
    Adversary knows the account frequency profile f and (optionally) the
    plaintext c, d of each row.  It observes the deterministic ciphertext
    identity of each row's tuple.  Strategy: inside each observed group,
    rank tuple-identities by count and match to accounts ranked by expected
    count in that group.  Returns per-row recovery accuracy of the account.
    """
    from collections import defaultdict
    key_of = []
    for i in range(len(a)):
        base = int(alias[i]) if alias is not None else int(a[i])
        key_of.append((base, int(c[i]), int(d[i])))
    groups = defaultdict(list)
    for i, kk in enumerate(key_of):
        g = (kk[1], kk[2]) if use_cd else 0
        groups[g].append(i)

    correct = 0
    for g, idxs in groups.items():
        cnt = defaultdict(int)
        for i in idxs:
            cnt[key_of[i]] += 1
        # ciphertext identities ranked by observed frequency
        ranked_ct = [t for t, _ in sorted(cnt.items(), key=lambda x: -x[1])]
        # candidate accounts in this group ranked by their true frequency
        truth = defaultdict(int)
        for i in idxs:
            truth[int(a[i])] += 1
        ranked_acct = [x for x, _ in sorted(truth.items(), key=lambda x: -x[1])]
        # each ciphertext identity maps to exactly one account; the adversary
        # guesses the r-th most frequent identity belongs to the r-th most
        # frequent account (ties broken arbitrarily)
        guess = {}
        for r, t in enumerate(ranked_ct):
            guess[t] = ranked_acct[min(r, len(ranked_acct) - 1)]
        for i in idxs:
            if guess[key_of[i]] == int(a[i]):
                correct += 1
    return correct / len(a)


def run(m=200, N=40000, n_cc=3, n_dt=2, K=9800):
    w = 1.0 / np.arange(1, m + 1)
    f = w / w.sum()
    cc_of, dt_of = make_world(m, n_cc, n_dt)
    a, c, d = gen_rows(m, f, cc_of, dt_of, N)

    n_eff = eff_domain(a, c, d, m)
    k = alloc_hamilton(f, K)
    start = np.concatenate([[0], np.cumsum(k)[:-1]])
    alias = np.array([start[x] + rng.integers(k[x]) for x in a])

    acc_plain_uncond = attack(a, c, d, m, use_cd=False)
    acc_plain_cond = attack(a, c, d, m, use_cd=True)
    acc_fpae_cond = attack(a, c, d, m, alias=alias, use_cd=True)
    return n_eff, acc_plain_uncond, acc_plain_cond, acc_fpae_cond


print("=" * 92, flush=True)
print("LEG (a) -- CONDITIONAL-DOMAIN COLLAPSE AND WHETHER FPAE REPAIRS IT", flush=True)
print("=" * 92, flush=True)

# verify the real FF1 pipeline once on the lifted 9-digit domain
ff1 = FF1(os.urandom(32), radix=10)
p = [int(x) for x in "012345678"]
ct = ff1.encrypt(p, tweak=b"ENT1_2026M09")
print("  real FF1 on lifted 9-digit domain: {} -> {}   (bijection, relabels "
      "tuples only)".format("".join(map(str, p)), "".join(map(str, ct))),
      flush=True)
print("  nominal lifted domain = 10^9  (>= the 10^6 floor in draft Rev 1)",
      flush=True)
print("", flush=True)
print("  correlation strength = how many cost centres / doc types each account uses",
      flush=True)
print("", flush=True)
print("  {:>18} {:>10} | {:>14} {:>14} | {:>14}".format(
    "corr (cc x dt)", "N_eff", "recover a", "recover a", "recover a"), flush=True)
print("  {:>18} {:>10} | {:>14} {:>14} | {:>14}".format(
    "", "(acct/pair)", "no FPAE, no cd", "no FPAE + cd", "FPAE + cd"), flush=True)
for n_cc, n_dt in ((1, 1), (2, 1), (3, 2), (5, 3), (10, 5), (25, 10)):
    ne, a0, a1, a2 = run(n_cc=n_cc, n_dt=n_dt)
    print("  {:>18} {:>10.1f} | {:>13.1%} {:>14.1%} | {:>14.1%}".format(
        "{} x {}".format(n_cc, n_dt), ne, a0, a1, a2), flush=True)

print("", flush=True)
print("  READING:", flush=True)
print("   * N_eff = distinct accounts consistent with an observed (c,d) pair.", flush=True)
print("     Small N_eff = the lifted domain has collapsed for that adversary.", flush=True)
print("   * col 3 vs col 2 = what knowing c,d buys the adversary.", flush=True)
print("   * col 4 vs col 3 = whether FPAE repairs the collapse.", flush=True)
