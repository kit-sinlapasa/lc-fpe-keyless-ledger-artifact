#!/usr/bin/env python3
"""Exact maximum-likelihood attack for Paper A's one-cell leakage model.

For one public (cost-centre, document-type) cell, let p_a be the conditional
posting probability of account a and let k_a be its public alias allocation.
Rows are independent and each row of account a selects one of its k_a aliases
uniformly.  Conditional on the observed class counts n_z, an assignment phi
of classes to account slots has multinomial log likelihood

    constant + sum_z n_z log(p_phi(z) / k_phi(z)).

Every account contributes exactly k_a slots, including aliases with zero
observations.  The rearrangement inequality therefore gives an exact ML
assignment: sort the K class counts and the K slot log-rates in the same
order.  This script implements that assignment without constructing a K by K
cost matrix.  Equal-count classes are ordered by a domain-separated hash;
this avoids accidentally using the contiguous plaintext alias numbers that
stand in for pseudorandom ciphertext labels in the simulator.

The result is optimal only for this explicitly stated one-cell i.i.d. model.
It does not use cross-cell co-occurrence and is not an optimal adversary for
the complete ledger leakage function.
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from leg_a2 import (K, M, N, SEEDS as BASE_SEEDS, baseline_predictions,
                    groups, n_eff, world)
from lcfpe_impl import alloc_hamilton


SEEDS = list(range(11, 11 + 25 * 11, 11))


def ci95(values):
    values = np.asarray(values, dtype=float)
    half = student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean()), float(half)


def upper95(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean() + student_t.ppf(0.95, len(values) - 1)
                 * values.std(ddof=1) / math.sqrt(len(values)))


def tie_key(label, seed):
    return hashlib.sha256(b"paper-a-exact-ml/tie/v1" + seed.to_bytes(8, "big")
                          + int(label).to_bytes(8, "big")).digest()


def exact_ml(f, owners, centres, doc_types, aliases, kvec, compatible, seed):
    """Return row accuracy of the exact ML assignment in each public cell."""
    correct = 0
    for cell, rows in groups(owners, centres, doc_types, True).items():
        candidates = sorted(compatible[cell])
        normaliser = float(sum(f[a] for a in candidates))

        slots = []
        for account in candidates:
            rate = (float(f[account]) / normaliser) / int(kvec[account])
            slots.extend((math.log(rate), account) for _ in range(int(kvec[account])))
        slots.sort(key=lambda item: (item[0], item[1]))

        classes = defaultdict(list)
        for row in rows:
            classes[int(aliases[row])].append(row)
        if len(classes) > len(slots):
            raise AssertionError("more observed classes than admissible alias slots")

        # Zero-count aliases take the lowest-rate slots.  Pair the remaining
        # positive counts and rates in the same order, which is exact by the
        # rearrangement inequality.
        positive = sorted(classes, key=lambda z: (len(classes[z]), tie_key(z, seed)))
        chosen_slots = slots[len(slots) - len(positive):]
        assignment = {z: slot[1] for z, slot in zip(positive, chosen_slots)}
        correct += sum(int(owners[row]) == assignment[z]
                       for z, class_rows in classes.items() for row in class_rows)
    return correct / len(owners)


def count_bayes(f, owners, centres, doc_types, aliases, kvec, compatible):
    """Bayes-optimal row prediction for an isolated observed class count.

    Poissonization makes an alias count conditionally Poisson with
    lambda_a=N_cd p_a/k_a.  A randomly relabelled class belongs to account a
    with prior k_a/sum(k), so the posterior score for a positive count n is

        log k_a - lambda_a + n log lambda_a - log(n!).

    The factorial term is common to candidates.  Choosing the largest score
    independently for each class minimizes expected row-wise 0-1 loss in the
    marginal count-only model.  It deliberately does not condition on the
    other observed classes or enforce their joint slot capacities.
    """
    correct = 0
    for cell, rows in groups(owners, centres, doc_types, True).items():
        candidates = sorted(compatible[cell])
        normaliser = float(sum(f[a] for a in candidates))
        classes = defaultdict(list)
        for row in rows:
            classes[int(aliases[row])].append(row)
        by_count = {}
        for count in {len(class_rows) for class_rows in classes.values()}:
            best_account, best_score = None, -float("inf")
            for account in candidates:
                ka = int(kvec[account])
                lam = len(rows) * (float(f[account]) / normaliser) / ka
                score = math.log(ka) - lam + count * math.log(lam)
                if score > best_score:
                    best_account, best_score = account, score
            by_count[count] = best_account
        for class_rows in classes.values():
            best_account = by_count[len(class_rows)]
            correct += sum(int(owners[row]) == best_account for row in class_rows)
    return correct / len(owners)


def main():
    print("=" * 112)
    print("PAPER A -- EXACT ML ATTACK IN THE ONE-CELL IID MODEL")
    print("=" * 112)
    print("  m={}, {:,} rows, K={}, {} independent worlds".format(M, N, K, len(SEEDS)))
    print("  exactness follows from sorted class-count/slot-rate matching; no fitted parameters")
    print("  scope: one cell at a time; no cross-cell co-occurrence")

    records = []
    for pool_cc, pool_dt, tag in ((8, 12, "realistic pool (8 cc x 12 dt)"),
                                  (100, 10, "sparse pool (100 cc x 10 dt)")):
        print("\n  " + tag)
        print("  {:>7} {:>6} {:>11} {:>10} {:>10} {:>18} {:>9}".format(
            "cc x dt", "N_eff", "no-cipher", "joint ML", "count Bayes",
            "Bayes - base (95%)", "upper"))
        for n_cc, n_dt in ((1, 1), (3, 2), (5, 3), (10, 5), (25, 10)):
            neff, baseline, ml, bayes, contrast = [], [], [], [], []
            for seed in SEEDS:
                f, a, c, d, aliases, aux = world(
                    seed, pool_cc, pool_dt, n_cc, n_dt, include_aux=True)
                kvec = alloc_hamilton(f, K)
                b = float(np.mean(baseline_predictions(f, c, d, aux) == a))
                score = exact_ml(f, a, c, d, aliases, kvec, aux, seed)
                bayes_score = count_bayes(f, a, c, d, aliases, kvec, aux)
                neff.append(n_eff(a, c, d))
                baseline.append(b)
                ml.append(score)
                bayes.append(bayes_score)
                contrast.append(bayes_score - b)
            cm, ch = ci95(contrast)
            row = dict(pool=[pool_cc, pool_dt], cell=[n_cc, n_dt], seeds=SEEDS,
                       n_eff=neff, baseline=baseline, exact_ml=ml,
                       count_bayes=bayes,
                       contrast=contrast, upper95=upper95(contrast))
            records.append(row)
            print("  {:>7} {:>6.1f} {:>10.1%} {:>9.1%} {:>9.1%} {:+7.2%} +- {:>5.2%} {:+8.2%}".format(
                "{}x{}".format(n_cc, n_dt), np.mean(neff), np.mean(baseline),
                np.mean(ml), np.mean(bayes), cm, ch, row["upper95"]))

    here = Path(__file__).resolve().parent
    out = (here.parent / "artifact" / "results" if here.name == "sim"
           else here.parent / "results")
    out.mkdir(exist_ok=True)
    (out / "a_exact_ml.json").write_text(
        json.dumps({"records": records}, indent=2), encoding="utf-8")
    print("\n  exact-ML self-checks")
    # A larger count must never be paired with a lower rate than a smaller
    # count; this is the exchange condition used in the proof.
    counts, rates = [0, 1, 4, 9], [-4.0, -3.0, -2.0, -1.0]
    aligned = sum(n * r for n, r in zip(counts, rates))
    crossed = sum(n * r for n, r in zip(counts, [rates[0], rates[2], rates[1], rates[3]]))
    print("  aligned likelihood dominates a crossed assignment : {}".format(aligned >= crossed))
    print("  base five-seed set is contained in evaluation set  : {}".format(
        set(BASE_SEEDS).issubset(SEEDS)))
    print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
