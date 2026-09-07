#!/usr/bin/env python3
"""
LEG (a), second pass -- the conditional-domain result, with the two controls the
first pass lacked.

The first pass fixed N_CC = 100 cost centres and gave each account five of them,
so a (c,d) pair was consistent with about two accounts and knowing (c,d) almost
named the account by itself.  It also ranked candidate accounts by their
EMPIRICAL count inside the very group under attack, which hands the adversary a
per-cell histogram it would not possess.

This pass:
  * sweeps the cost-centre POOL, not just how many each account uses, so the
    realistic regime (a handful of departments shared by every account) is
    covered;
  * ranks candidates by the chart's GENERATIVE frequency vector;
  * reports the CIPHERTEXT-FREE BASELINE -- an adversary that answers "the most
    common account in this (c,d) cell" and never looks at the ciphertext.

Only the excess over that baseline is attributable to the encryption.
"""
import numpy as np
from collections import defaultdict
from lcfpe_impl import alloc_hamilton

SEEDS = [11, 22, 33, 44, 55]
M, N, K = 200, 40_000, 9_800


def world(seed, pool_cc, pool_dt, n_cc, n_dt, include_aux=False):
    r = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, M + 1)
    f = w / w.sum()
    cc_of = [r.choice(pool_cc, min(n_cc, pool_cc), replace=False)
             for _ in range(M)]
    dt_of = [r.choice(pool_dt, min(n_dt, pool_dt), replace=False)
             for _ in range(M)]
    a = r.choice(M, size=N, p=f)
    c = np.array([cc_of[x][r.integers(len(cc_of[x]))] for x in a])
    d = np.array([dt_of[x][r.integers(len(dt_of[x]))] for x in a])
    k = alloc_hamilton(f, K)
    start = np.concatenate([[0], np.cumsum(k)[:-1]])
    alias = np.array([start[x] + r.integers(k[x]) for x in a])
    result = (f, a, c, d, alias)
    return result + (compatibility(cc_of, dt_of),) if include_aux else result


def compatibility(cc_of, dt_of):
    """Public posting permissions, constructed without evaluation labels."""
    out = defaultdict(list)
    for account, (centres, types) in enumerate(zip(cc_of, dt_of)):
        for cc in centres:
            for dd in types:
                out[int(cc), int(dd)].append(account)
    return dict(out)


def baseline_predictions(f, c, d, compatible):
    """Context-only MAP under the public generative profile and permissions.

    Each account uses equally many centres/types in these experiments, so
    their conditional likelihood factors cancel. No test labels enter here.
    """
    best = {cell: min(accounts, key=lambda a: (-f[a], a))
            for cell, accounts in compatible.items() if accounts}
    return np.asarray([best[int(cc), int(dd)] for cc, dd in zip(c, d)])


def groups(a, c, d, use_cd):
    g = defaultdict(list)
    for i in range(len(a)):
        g[(int(c[i]), int(d[i])) if use_cd else 0].append(i)
    return g


def n_eff(a, c, d):
    g = groups(a, c, d, True)
    return sum(len({int(a[i]) for i in idxs}) * len(idxs)
               for idxs in g.values()) / len(a)


def oracle_baseline(a, c, d):
    """Retrospective empirical-majority control, NOT an executable attacker."""
    g = groups(a, c, d, True)
    return sum(int(np.bincount([int(a[i]) for i in idxs]).max())
               for idxs in g.values()) / len(a)


def attack(f, a, c, d, alias=None, use_cd=True, compatible=None):
    if use_cd and compatible is None:
        raise ValueError("public compatibility is required; do not infer it from labels")
    rank = {x: r for r, x in enumerate(np.argsort(-f).tolist())}
    key = [(int(alias[i]) if alias is not None else int(a[i]),
            int(c[i]), int(d[i])) for i in range(len(a))]
    correct = 0
    for cell, idxs in groups(a, c, d, use_cd).items():
        cnt = defaultdict(int)
        for i in idxs:
            cnt[key[i]] += 1
        order = sorted(cnt, key=lambda z: (-cnt[z], z))
        cands = sorted(compatible[cell] if use_cd else range(len(f)),
                       key=lambda x: rank[x])
        gss = {t: cands[min(r, len(cands) - 1)] for r, t in enumerate(order)}
        correct += sum(1 for i in idxs if gss[key[i]] == int(a[i]))
    return correct / len(a)


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))



if __name__ == "__main__":
    print("=" * 100, flush=True)
    print("LEG (a) pass 2 -- conditional-domain collapse, with a ciphertext-free "
          "baseline", flush=True)
    print("=" * 100, flush=True)
    print("  m=200, 40k rows, five seeds, aux = the chart's generative frequency "
          "vector", flush=True)

    for pool_cc, pool_dt, tag in ((100, 10, "sparse pool  (100 cc x 10 dt)"),
                                  (8, 12, "realistic pool (8 cc x 12 dt)")):
        print(flush=True)
        print("  {}".format(tag), flush=True)
        print("  {:>10} {:>8} {:>12} | {:>13} {:>13} {:>13}".format(
            "cc x dt", "N_eff", "no-cipher", "no FPAE,no cd", "no FPAE +cd",
            "FPAE +cd"), flush=True)
        for n_cc, n_dt in ((1, 1), (3, 2), (5, 3), (10, 5), (25, 10)):
            R = [[], [], [], [], []]
            for s in SEEDS:
                f, a, c, d, al, aux = world(s, pool_cc, pool_dt, n_cc, n_dt,
                                           include_aux=True)
                R[0].append(n_eff(a, c, d))
                R[1].append(float(np.mean(baseline_predictions(f, c, d, aux) == a)))
                R[2].append(attack(f, a, c, d, use_cd=False))
                R[3].append(attack(f, a, c, d, use_cd=True, compatible=aux))
                R[4].append(attack(f, a, c, d, alias=al, use_cd=True, compatible=aux))
            v = [ci(x) for x in R]
            print("  {:>10} {:>5.1f}+-{:.1f} {:>7.1%}+-{:.1%} | "
                  "{:>6.1%}+-{:.1%} {:>6.1%}+-{:.1%} {:>6.1%}+-{:.1%}"
                  .format("{}x{}".format(n_cc, n_dt),
                          v[0][0], v[0][1], v[1][0], v[1][1],
                          v[2][0], v[2][1], v[3][0], v[3][1],
                          v[4][0], v[4][1]), flush=True)
    print(flush=True)
    print("done", flush=True)
