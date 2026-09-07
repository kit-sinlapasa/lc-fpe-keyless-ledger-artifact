#!/usr/bin/env python3
"""
FPAE — numerical verification of Claims 2, 3, 4.

Claim 2  budget:      sum_a k(a) <= K + m - 1
Claim 3  flatness:    Delta = TV(f, k/K')  and  Delta < m/K
                      head factor (1/K)/g <= 1 + 1/(f(a)K)
Claim 4  detection:   alias of account a is frequency-safe iff
                      |g - 1/K'| <= sqrt(1/(K' R))   <=>  R < f(a)^2 K^3  (head)
"""
import numpy as np
from math import sqrt

rng = np.random.default_rng(20260903)


# ---------------------------------------------------------------- chart of accounts
def zipf_chart(m, s):
    """Zipf(s) frequency profile over m accounts, descending."""
    w = 1.0 / np.arange(1, m + 1) ** s
    return w / w.sum()


def allocate(f, K):
    """k(a) = ceil(f(a) * K).  max(1, .) is redundant for f(a) > 0."""
    return np.ceil(f * K).astype(np.int64)


# ---------------------------------------------------------------- Claim 2
def check_claim2(m_list, s_list, K_list):
    worst_slack = None
    rows = []
    for m in m_list:
        for s in s_list:
            f = zipf_chart(m, s)
            for K in K_list:
                k = allocate(f, K)
                tot, bound = int(k.sum()), K + m - 1
                ok = tot <= bound
                slack = bound - tot
                rows.append((m, s, K, tot, bound, ok, slack))
                if worst_slack is None or slack < worst_slack:
                    worst_slack = slack
    allok = all(r[5] for r in rows)
    return allok, worst_slack, rows


def claim2_tightness():
    """m=3, f=1/3 each, K=4  ->  sum k = 6 = K+m-1 exactly."""
    f = np.array([1 / 3, 1 / 3, 1 / 3])
    k = allocate(f, 4)
    return int(k.sum()), 4 + 3 - 1


# ---------------------------------------------------------------- Claim 3
def claim3(f, K):
    k = allocate(f, K)
    Kp = int(k.sum())
    g = f / k                                   # per-alias probability, one per account
    # exact Delta over the alias support (k(a) copies of g(a))
    delta_exact = 0.5 * np.sum(k * np.abs(g - 1.0 / Kp))
    # closed form:  Delta = TV(f, k/Kp)
    delta_closed = 0.5 * np.sum(np.abs(f - k / Kp))
    bound = len(f) / K
    # head factor check
    head = f >= 1.0 / K
    lhs = (1.0 / K) / g[head]
    rhs = 1.0 + 1.0 / (f[head] * K)
    return dict(Kp=Kp, delta_exact=delta_exact, delta_closed=delta_closed,
                bound=bound, head_ok=bool(np.all(lhs <= rhs + 1e-12)),
                head_max_factor=float(lhs.max()) if head.any() else float("nan"),
                n_head=int(head.sum()), n_tail=int((~head).sum()))


# ---------------------------------------------------------------- Claim 4
def claim4(f, K, R, trials=400, z=1.0):
    """
    Monte-Carlo: draw R rows from the alias distribution, run a per-alias
    detector  |c - R/K'| > z*sd , and compare the empirical detection rate
    against the predicted safe/unsafe split.
    """
    k = allocate(f, K)
    Kp = int(k.sum())
    g_acct = f / k
    # expand to per-alias probability vector
    g = np.repeat(g_acct, k)
    owner = np.repeat(np.arange(len(f)), k)
    p_unit = 1.0 / Kp
    sd = sqrt(R * p_unit * (1 - p_unit))

    predicted_unsafe = np.abs(g_acct - p_unit) > sqrt(1.0 / (Kp * R))

    flags = np.zeros(len(g))
    for _ in range(trials):
        c = rng.multinomial(R, g / g.sum())
        flags += (np.abs(c - R * p_unit) > z * sd)
    rate_alias = flags / trials
    # per-account mean detection rate
    rate_acct = np.array([rate_alias[owner == i].mean() for i in range(len(f))])
    return dict(Kp=Kp, predicted_unsafe=predicted_unsafe, rate_acct=rate_acct,
                f_min_pred=sqrt(R / K ** 3))


# ================================================================= run
print("=" * 78)
print("CLAIM 2 — budget:  sum_a k(a) <= K + m - 1")
print("=" * 78)
ok, slack, rows = check_claim2([50, 200, 300, 500, 2000],
                               [0.6, 0.8, 1.0, 1.2, 1.5],
                               [100, 500, 2000, 9700, 50000])
print(f"  configurations tested : {len(rows)}")
print(f"  all satisfy bound     : {ok}")
print(f"  smallest slack seen   : {slack}   (0 = bound attained)")
t_tot, t_bound = claim2_tightness()
print(f"  tightness case m=3,f=1/3,K=4 : sum k = {t_tot}, bound = {t_bound}"
      f"  -> {'TIGHT' if t_tot == t_bound else 'not tight'}")
alpha, m_use = 10_000, 300
f300 = zipf_chart(m_use, 1.0)
k300 = allocate(f300, alpha - m_use)
print(f"  operational: alpha=1e4, m=300, K=9700 -> sum k = {int(k300.sum())} "
      f"({'fits' if k300.sum() <= alpha else 'OVERFLOW'} in alpha)")

print()
print("=" * 78)
print("CLAIM 3 — Delta = TV(f, k/K')  and  Delta < m/K")
print("=" * 78)
print(f"{'m':>5} {'s':>5} {'K':>7} {'Delta_exact':>12} {'TV(f,k/K)':>12} "
      f"{'m/K bound':>11} {'<bnd':>5} {'headOK':>7} {'maxHeadFac':>11} {'tail':>5}")
for m in (300, 500):
    for s in (0.8, 1.0, 1.3):
        f = zipf_chart(m, s)
        for K in (500, 2000, 9700):
            r = claim3(f, K)
            print(f"{m:>5} {s:>5.1f} {K:>7} {r['delta_exact']:>12.6f} "
                  f"{r['delta_closed']:>12.6f} {r['bound']:>11.6f} "
                  f"{str(r['delta_exact'] < r['bound']):>5} "
                  f"{str(r['head_ok']):>7} {r['head_max_factor']:>11.4f} "
                  f"{r['n_tail']:>5}")

print()
print("=" * 78)
print("CLAIM 4 — protection threshold  f(a) > sqrt(R / K^3)")
print("=" * 78)
f = zipf_chart(300, 1.0)
K = 9700
for R in (10_000, 100_000, 1_000_000):
    r = claim4(f, K, R, trials=300, z=1.0)
    pred = r["predicted_unsafe"]
    rate = r["rate_acct"]
    # agreement between predicted-unsafe and empirically-detected (rate > 0.5)
    emp = rate > 0.5
    agree = float((pred == emp).mean())
    print(f"  R = {R:>9,}   f_min(pred) = {r['f_min_pred']:.6%}   "
          f"predicted-unsafe = {int(pred.sum()):>3}/{len(f)}   "
          f"empirically-detected = {int(emp.sum()):>3}/{len(f)}   "
          f"agreement = {agree:.1%}")
    # where does the empirical boundary sit?
    if emp.any() and (~emp).any():
        lo = f[emp].max()
        print(f"                 highest-frequency account still detected: "
              f"f = {lo:.6%}")
