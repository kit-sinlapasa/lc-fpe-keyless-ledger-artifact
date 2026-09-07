#!/usr/bin/env python3
"""
Paper A: the combined adversary the reviews asked for (A-P0.6).

Until now the conditional-domain table measured two heuristics separately --
a ciphertext-free baseline and a rank-matching attack that reads the
ciphertext -- and reported the second as "the" FPAE figure. A rational
adversary is not bound to either: it can combine every source of evidence and
keep whichever does best. This script builds that adversary and reports

    advantage_s = max(rank_s, combined_s) - baseline_s          per seed s

reported UNCLIPPED, with a paired two-sided interval and a one-sided upper
95% confidence bound on this tested-heuristic contrast, not on an optimal
adversary or on the information in ciphertext. The old empirical-majority
baseline is retained only as an explicitly labelled oracle control.

An earlier version of this script reported max(baseline, rank, combined) -
baseline instead. That quantity cannot be negative by construction, so when
every attack fell below the baseline it collapsed to exactly 0.00 with a
degenerate interval [0, 0] -- which looks like strong evidence of no leakage
and is in fact no evidence at all, only a restatement of the definition. The
unclipped difference below can and does go negative, and its upper bound is
a real bound.

THE OBSERVABLE is the deployed one. Under one FF1 tweak two rows carry the
same ciphertext triple exactly when their aliased plaintext triples (a-hat, c,
d) coincide, so the equality classes of (alias, c, d) used here are precisely
the equality classes of the ciphertext an adversary holds. Nothing finer is
assumed.

THE COMBINED ATTACKER (per (c,d) cell, all rows in one cryptographic context)
knows: the cell's row count N_cd; the class sizes n_z of the ciphertext
equality classes in the cell; the chart's generative profile f; the public
allocation k(a); and, as in every row of the table, which accounts post to
the cell at all. Its model: account a contributes N_cd * p_a rows to the cell
(p_a = f(a) renormalised over the cell's accounts) and spreads them uniformly
over its k(a) aliases, so a class belonging to a has size ~ Poisson(lambda_a),
lambda_a = N_cd * p_a / k(a). It scores every (class, account) pair by

    log p_a + log Poisson(n_z ; lambda_a)

and assigns classes to accounts greedily in decreasing class size, never
giving an account more classes than its k(a) (which the public allocation
bounds). This uses context, profile, multiplicity and the equality pattern at
once, which is exactly the combination the reviews said was missing.

Without FPAE (k(a)=1 for every account) the same code is the classical
frequency-matching attack, since each account owns exactly one class.
"""
import math
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from leg_a2 import (world, groups, n_eff, baseline_predictions, oracle_baseline,
                    attack, ci, M, N, K)
from scipy.stats import t as student_t
from lcfpe_impl import alloc_hamilton


SEEDS = list(range(11, 11 + 25 * 11, 11))       # 25 worlds: 11, 22, ..., 275


def upper95(v):
    """One-sided upper 95% confidence bound on the mean of v."""
    a = np.asarray(v, dtype=float)
    n = len(a)
    if n < 2:
        return float("nan")
    return float(a.mean() + student_t.ppf(0.95, n - 1) * a.std(ddof=1) / np.sqrt(n))


def combined(f, a, c, d, key_alias, kvec, compatible):
    """Greedy MAP assignment of ciphertext classes to accounts per cell."""
    correct = 0
    for (cc, dd), idxs in groups(a, c, d, True).items():
        N_cd = len(idxs)
        cands = sorted(compatible[cc, dd])
        pa = np.array([f[x] for x in cands], dtype=float)
        pa /= pa.sum()
        cls = defaultdict(list)
        for i in idxs:
            cls[int(key_alias[i])].append(i)
        order = sorted(cls, key=lambda z: -len(cls[z]))
        used = defaultdict(int)
        assign = {}
        for z in order:
            n_z = len(cls[z])
            best, best_s = None, -1e300
            for x, p in zip(cands, pa):
                kx = int(kvec[x])
                if used[x] >= kx:
                    continue
                lam = N_cd * p / kx
                s = math.log(p) + (n_z * math.log(lam) - lam - math.lgamma(n_z + 1))
                if s > best_s:
                    best, best_s = x, s
            if best is None:                     # every candidate saturated
                best = cands[int(np.argmax(pa))]
            assign[z] = best
            used[best] += 1
        for z, rows in cls.items():
            correct += sum(1 for i in rows if int(a[i]) == assign[z])
    return correct / len(a)


print("=" * 118, flush=True)
print("PAPER A -- COMBINED ADVERSARY on the conditional domain (deployed observable)",
      flush=True)
print("=" * 118, flush=True)
print("  m={}, {:,} rows, K={}, {} independently generated worlds".format(
    M, N, K, len(SEEDS)), flush=True)
print("  advantage = max(rank, combined) - baseline, per seed, NOT clipped at zero;",
      flush=True)
print("  'upper' bounds only the mean tested-heuristic contrast, not optimal advantage.", flush=True)
print("  The attacker has no fitted hyperparameters -- the Poisson model and the",
      flush=True)
print("  greedy assignment are fixed a priori -- so there is no tuning set to hold out.",
      flush=True)

records = []
for pool_cc, pool_dt, tag in ((8, 12, "realistic pool (8 cc x 12 dt)"),
                              (100, 10, "sparse pool (100 cc x 10 dt)")):
    print(flush=True)
    print("  {}".format(tag), flush=True)
    print("  {:>7} {:>6} | {:>9} {:>9} {:>9} | {:>19} {:>19} | {:>8}".format(
        "cc x dt", "N_eff", "no-cipher", "rank-atk", "combined",
        "rank - base (95%)", "comb - base (95%)", "upper"), flush=True)
    print("  " + "-" * 114, flush=True)
    for n_cc, n_dt in ((1, 1), (3, 2), (5, 3), (10, 5), (25, 10)):
        ne, b0, r1, cb, d_rank, d_comb, adv, oracle = [], [], [], [], [], [], [], []
        for s_ in SEEDS:
            f, a, c, d, al, aux = world(s_, pool_cc, pool_dt, n_cc, n_dt,
                                       include_aux=True)
            kvec = alloc_hamilton(f, K)
            ne.append(n_eff(a, c, d))
            base = float(np.mean(baseline_predictions(f, c, d, aux) == a))
            oracle.append(oracle_baseline(a, c, d))
            rk = attack(f, a, c, d, alias=al, use_cd=True, compatible=aux)
            co = combined(f, a, c, d, al, kvec, aux)
            b0.append(base); r1.append(rk); cb.append(co)
            d_rank.append(rk - base)
            d_comb.append(co - base)
            adv.append(max(rk, co) - base)
        dr_m, dr_h = ci(d_rank)
        dc_m, dc_h = ci(d_comb)
        records.append(dict(pool=[pool_cc, pool_dt], cell=[n_cc, n_dt],
                            seeds=SEEDS, baseline=b0, oracle=oracle, rank=r1,
                            combined=cb, contrast=adv, upper95=upper95(adv)))
        print("  {:>7} {:>6.1f} | {:>8.1%} {:>8.1%} {:>8.1%} | "
              "{:>+7.2%} +-{:>6.2%} {:>+7.2%} +-{:>6.2%} | {:>+7.2%}".format(
                  "{}x{}".format(n_cc, n_dt), np.mean(ne), np.mean(b0),
                  np.mean(r1), np.mean(cb),
                  dr_m, dr_h, dc_m, dc_h, upper95(adv)), flush=True)
        print("           oracle control (uses test labels): {:.3%}".format(
            np.mean(oracle)), flush=True)

# the same attacker against plain lifting (no FPAE), for scale: here every
# account owns exactly one class and the Poisson model reduces to frequency
# matching, which should recover most of the cell
print(flush=True)
print("  control: the combined attacker against lifting WITHOUT FPAE "
      "(k(a)=1), realistic pool, 5x3", flush=True)
vals = []
for s_ in SEEDS:
    f, a, c, d, _, aux = world(s_, 8, 12, 5, 3, include_aux=True)
    vals.append(combined(f, a, c, d, a, np.ones(M, dtype=int), aux))
m, h = ci(vals)
print("  combined, no FPAE: {:.1%} +- {:.1%}   (rank attack in tab:lega: "
      "64.3%)".format(m, h), flush=True)
print(flush=True)
print("  These are contrasts of the specified heuristics with context-only MAP.", flush=True)
print("  A rational attacker can ignore ciphertext; a negative contrast does not", flush=True)
print("  bound optimal advantage or establish absence of information.", flush=True)
out = Path(__file__).resolve().parents[1] / "results"
if Path(__file__).resolve().parent.name == "sim":
    out = Path(__file__).resolve().parents[1] / "artifact" / "results"
out.mkdir(exist_ok=True)
(out / "a_combined_corrected.json").write_text(
    json.dumps(dict(records=records, control=vals), indent=2), encoding="utf-8")
print(flush=True)
print("done", flush=True)
