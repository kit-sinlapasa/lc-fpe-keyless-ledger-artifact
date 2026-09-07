#!/usr/bin/env python3
"""
F. Ablation, second pass -- the two controls the first pass was missing.

CONTROL 1: chance baseline for the frequency column.  FPAE gives a hub account
thousands of aliases, so a high-count alias belongs to a hub for reasons of
sheer mass, not leakage.  The null is the fraction of NODES owned by the ten
most frequent accounts; only the excess over that null is a leak.

CONTROL 2: a non-circular (c,d)-knowing adversary.  Rank observable classes by
count inside each true (c,d) group, match to accounts ranked by the chart's
generative frequency vector w.f -- never by the empirical histogram of the rows
under attack.  This is the realistic adversary: cost centre and document type
are low-entropy business facts an insider knows.

The first pass also reported clustering ARI = 1.0 for 'plain'.  That is true but
vacuous: with no aliasing every node IS an account, so the clustering metric has
nothing to recover.  Row recovery is the attack that binds there.
"""
import numpy as np
from collections import defaultdict
from harness import ci95
from harness_tfrs import TFRSWorld
from f_ablation import build, CELLS

SEEDS = [11, 22, 33]


def freq_top(w, owner, u, v, t=10):
    cnt = np.bincount(np.concatenate([u, v]), minlength=len(owner))
    top_accts = set(np.argsort(-w.f)[:t].tolist())
    hit = sum(1 for x in np.argsort(-cnt)[:t] if owner[x] in top_accts) / t
    null = float(np.mean([owner[x] in top_accts for x in range(len(owner))]))
    return hit, null


def recover_cd(w, owner, u, v, occ):
    """Row recovery when the adversary knows each row's (c,d).  Aux is w.f."""
    da, ca, d_al, c_al, d_cc, d_dt, c_cc, c_dt = w._last
    acct = np.concatenate([da, ca])
    cd = list(zip(np.concatenate([d_cc, c_cc]).tolist(),
                  np.concatenate([d_dt, c_dt]).tolist()))
    node = np.concatenate([u, v])
    rank_of = {a: r for r, a in enumerate(np.argsort(-w.f).tolist())}
    g = defaultdict(list)
    for i in range(len(node)):
        g[cd[i]].append(i)
    correct = 0
    for _, idxs in g.items():
        cnt = defaultdict(int)
        for i in idxs:
            cnt[int(node[i])] += 1
        # observed classes, most frequent first
        order = sorted(cnt, key=lambda z: (-cnt[z], z))
        # candidate accounts present in this group, by GENERATIVE frequency
        cands = sorted({int(acct[i]) for i in idxs}, key=lambda a: rank_of[a])
        guess = {t: cands[min(r, len(cands) - 1)] for r, t in enumerate(order)}
        correct += sum(1 for i in idxs if guess[int(node[i])] == int(acct[i]))
    return correct / len(node)


print("=" * 96, flush=True)
print("F. ABLATION pass 2 -- chance baseline and a non-circular (c,d) adversary",
      flush=True)
print("=" * 96, flush=True)
print("  structured TFRS chart, m=200, three seeds, correlation 5x3", flush=True)

for occ in (1.0, 4.0):
    acc = {c: ([], [], []) for c in CELLS}
    for s in SEEDS:
        w = TFRSWorld(s, m=200)
        # build() calls docs_for_occupancy; stash the raw draw for recover_cd
        orig = w.docs_for_occupancy
        def wrapped(o, _w=w, _f=orig):
            _w._last = _f(o)
            return _w._last
        w.docs_for_occupancy = wrapped
        b = build(w, occ)
        for c in CELLS:
            u, v, owner = b[c]
            h, n = freq_top(w, owner, u, v)
            acc[c][0].append(h); acc[c][1].append(n)
            acc[c][2].append(recover_cd(w, owner, u, v, occ))
    print(flush=True)
    print("  occupancy {:g}".format(occ), flush=True)
    print("  {:<8} {:>26} {:>12} {:>28}".format(
        "cell", "freq top-10 hit", "chance", "row recovery, knows c,d"),
        flush=True)
    for c in CELLS:
        hm, hh, _ = ci95(acc[c][0]); nm = float(np.mean(acc[c][1]))
        rm, rh, _ = ci95(acc[c][2])
        print("  {:<8} {:>17.0%} +/- {:.0%} {:>11.0%} {:>19.1%} +/- {:.1%}".format(
            c, hm, hh, nm, rm, rh), flush=True)

print(flush=True)
print("done", flush=True)
