#!/usr/bin/env python3
"""
Re-run the headline security numbers through the multi-seed harness so that
every one of them carries a 95% confidence interval.

Numbers re-derived here (all previously single-seed):
  1. Delta = TV(f, k/K')                     -- flatness
  2. clustering ARI vs occupancy             -- the design rule rests on this
  3. conditional-domain recovery, +/- FPAE   -- the headline security result
"""
import sys
import numpy as np
from collections import defaultdict
from harness import World, cluster_ari, repeat, fmt, N_CC, N_DT

SEEDS = [11, 22, 33, 44, 55]


# ------------------------------------------------------------------ 1. Delta
def delta_at(m, s):
    return lambda seed: World(seed, m=m, s=s).delta()


print("=" * 78, flush=True)
print("1. FLATNESS  Delta = TV(f, k/K')      seeds = {}".format(len(SEEDS)),
      flush=True)
print("=" * 78, flush=True)
print("  {:>5} {:>5} | {:>22} {:>12}".format("m", "s", "Delta (95% CI)",
                                             "bound m/K"), flush=True)
for m in (200, 300, 500):
    for s in (1.0, 1.6):
        mu, h, n, _ = repeat(delta_at(m, s), SEEDS)
        print("  {:>5} {:>5.1f} | {:>22} {:>12.4f}".format(
            m, s, fmt(mu, h, n), m / (10_000 - m)), flush=True)


# ------------------------------------------------------------------ 2. ARI
def ari_at(occ, m=300, s=1.0, n_types=400):
    def run(seed):
        w = World(seed, m=m, s=s, n_types=n_types)
        _, _, d_al, c_al, _, _, _, _ = w.docs_for_occupancy(occ)
        return cluster_ari(w, d_al, c_al, w.Kp, w.owner)
    return run


print("", flush=True)
print("=" * 78, flush=True)
print("2. CLUSTERING ARI vs OCCUPANCY   (m=300, K=9700, 400 tx types)",
      flush=True)
print("=" * 78, flush=True)
print("  {:>10} | {:>24}  regime".format("occupancy", "ARI (95% CI)"),
      flush=True)
for occ in (0.5, 1.0, 2.0, 4.0):
    mu, h, n, raw = repeat(ari_at(occ), SEEDS)
    hi = mu + (h if np.isfinite(h) else 0)
    reg = ("safe" if hi < 0.10 else "leaking" if hi < 0.35 else "UNSAFE")
    print("  {:>10.1f} | {:>24}  {}".format(occ, fmt(mu, h, n), reg),
          flush=True)


# ------------------------------------------------------------------ 3. leg (a)
def eff_domain(a, c, d):
    seen, cnt = defaultdict(set), defaultdict(int)
    for i in range(len(a)):
        seen[(c[i], d[i])].add(a[i]); cnt[(c[i], d[i])] += 1
    tot = sum(cnt.values())
    return sum(len(seen[x]) * cnt[x] for x in cnt) / tot


def attack_cd(a, c, d, key, use_cd):
    groups = defaultdict(list)
    for i in range(len(a)):
        groups[(int(c[i]), int(d[i])) if use_cd else 0].append(i)
    correct = 0
    for _, idxs in groups.items():
        cnt = defaultdict(int)
        for i in idxs:
            cnt[key[i]] += 1
        ranked_ct = [x for x, _ in sorted(cnt.items(), key=lambda z: -z[1])]
        truth = defaultdict(int)
        for i in idxs:
            truth[int(a[i])] += 1
        ranked_ac = [x for x, _ in sorted(truth.items(), key=lambda z: -z[1])]
        guess = {tk: ranked_ac[min(r, len(ranked_ac) - 1)]
                 for r, tk in enumerate(ranked_ct)}
        correct += sum(1 for i in idxs if guess[key[i]] == int(a[i]))
    return correct / len(a)


def lega(n_cc, n_dt, n_rows=40_000, m=200):
    def run(seed):
        w = World(seed, m=m, n_cc=n_cc, n_dt=n_dt)
        da, ca, d_al, c_al, d_cc, d_dt, c_cc, c_dt = w.docs(n_rows // 2)
        a = np.concatenate([da, ca])
        c = np.concatenate([d_cc, c_cc])
        d = np.concatenate([d_dt, c_dt])
        al = np.concatenate([d_al, c_al])
        raw = [(int(x), int(c[i]), int(d[i])) for i, x in enumerate(a)]
        ali = [(int(al[i]), int(c[i]), int(d[i])) for i in range(len(a))]
        return (eff_domain(a, c, d),
                attack_cd(a, c, d, raw, False),
                attack_cd(a, c, d, raw, True),
                attack_cd(a, c, d, ali, True))
    return run


print("", flush=True)
print("=" * 78, flush=True)
print("3. CONDITIONAL-DOMAIN ADVERSARY   (m=200)", flush=True)
print("=" * 78, flush=True)
print("  {:>7} {:>16} | {:>16} {:>16} {:>16}".format(
    "cc x dt", "N_eff (CI)", "no FPAE,no cd", "no FPAE + cd", "FPAE + cd"),
    flush=True)
for n_cc, n_dt in ((1, 1), (3, 2), (5, 3), (10, 5)):
    res = [lega(n_cc, n_dt)(s) for s in SEEDS]
    cols = list(zip(*res))
    out = []
    for col in cols:
        a = np.asarray(col, float)
        from scipy.stats import t as st
        h = st.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
        out.append((a.mean(), h))
    print("  {:>7} {:>9.2f}+/-{:.2f} | {:>10.1%}+/-{:.1%} {:>10.1%}+/-{:.1%} "
          "{:>10.1%}+/-{:.1%}".format(
              "{}x{}".format(n_cc, n_dt), out[0][0], out[0][1],
              out[1][0], out[1][1], out[2][0], out[2][1],
              out[3][0], out[3][1]), flush=True)

print("", flush=True)
print("done", flush=True)
