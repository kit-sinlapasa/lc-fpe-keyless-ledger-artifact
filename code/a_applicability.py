#!/usr/bin/env python3
"""
Paper A, item 3: the applicability checklist, and the deployment tension that
falls out of it.

Every constraint below is already somewhere in the paper. Putting them in one
place produces a result none of them states alone: the partition rule and the
history rule pull in OPPOSITE directions with firm size.

    R <= K = alpha - m          (occupancy R/K' <= 1, the design rule)
    n >= 2 K^2 S^2 / (pi m^2)   (history needed before sampling noise in f_hat
                                 stops dominating the leakage bound)

A firm posting 10^4 rows a year fits comfortably in one partition per year and
then waits over a decade for enough history. A firm posting 10^6 rows has the
history within weeks and must split the year into a hundred partitions. The
scheme is easiest to configure for the firm that has the hardest time
estimating f_hat, and vice versa.

S is not assumed here. It is computed from the same TFRS generator that
produces the paper's tables, and the script checks it against the two values
the paper quotes (8.7 at m=200, 10.2 at m=300) before using it for anything.
"""
import math
import numpy as np
from harness_tfrs import TFRSWorld

print("=" * 78, flush=True)
print("PAPER A -- APPLICABILITY CHECKLIST", flush=True)
print("=" * 78, flush=True)


def S_of(world):
    """S = sum_a sqrt(f(a)(1-f(a))), the multinomial l1 constant of eq:sampling."""
    f = world.f
    return float(np.sqrt(f * (1.0 - f)).sum())


def hist_rows(K, S, m):
    return 2.0 * K * K * S * S / (math.pi * m * m)


# ------------------------------------------------- validate S against the paper
print(flush=True)
print("  0. INSTRUMENT CHECK -- S against the values the paper quotes",
      flush=True)
print("     {:>5} {:>8} {:>10} {:>10}".format("m", "K'", "S computed",
                                              "paper"), flush=True)
for m, quoted in ((200, 8.7), (300, 10.2)):
    w = TFRSWorld(seed=11, m=m)
    S = S_of(w)
    ok = abs(S - quoted) / quoted < 0.05
    print("     {:>5} {:>8,} {:>10.2f} {:>10.1f}   {}".format(
        m, w.Kp, S, quoted, "ok" if ok else "MISMATCH"), flush=True)

# ------------------------------------------------------------- the checklist
print(flush=True)
print("  1. THE CHECKLIST, ordered by what disqualifies a deployment first",
      flush=True)
print("""
     (a) N_eff > 2        does a (cost centre, document type) cell hold more
                          than two candidate accounts? If a cell names the
                          account outright, no aliasing of the account code can
                          help and LC-FPE does not apply. Binary gate; measure
                          it on the real chart before anything else.
     (b) R <= alpha - m   partition size. The account field width alpha is
                          rarely negotiable, so this fixes how often the books
                          must be partitioned.
     (c) m <= ~400        chart size, measured. Beyond that recovery grows and
                          we have not measured the boundary.
     (d) n >= eq(histsize)  history for f_hat. The only constraint that cannot
                          be fixed by configuration -- only by waiting, or by
                          accepting the larger Delta.""", flush=True)

# --------------------------------------------------------- worked deployments
print(flush=True)
print("  2. WORKED DEPLOYMENTS", flush=True)
print("     alpha = 10^4 unless stated; K = alpha - m; R <= K", flush=True)
print(flush=True)
print("     {:>22} {:>9} {:>6} {:>9} {:>7} {:>11} {:>9}".format(
    "firm", "rows/yr", "m", "K", "parts/yr", "history rows", "years"),
    flush=True)

PROFILES = [
    ("micro (one bookkeeper)", 3_000, 120, 10_000),
    ("small (our motivating)", 12_000, 200, 10_000),
    ("mid-sized", 120_000, 300, 10_000),
    ("mid-sized, 5-digit field", 120_000, 300, 100_000),
    ("large", 1_000_000, 400, 100_000),
]

for name, rows_yr, m, alpha in PROFILES:
    w = TFRSWorld(seed=11, m=m, alpha=alpha)
    K = w.Kp
    S = S_of(w)
    parts = math.ceil(rows_yr / K)
    n_req = hist_rows(K, S, m)
    years = n_req / rows_yr
    print("     {:>22} {:>9,} {:>6} {:>9,} {:>7} {:>11.1e} {:>9.1f}".format(
        name, rows_yr, m, K, parts, n_req, years), flush=True)

# ------------------------------------------------------------- the tension
print(flush=True)
print("  3. THE TENSION", flush=True)
print("     Partitions per year rises with firm size; years of history needed "
      "falls.", flush=True)
print("     The two cross, and where they cross is the only comfortable "
      "operating point.", flush=True)
print(flush=True)
print("     {:>10} {:>10} {:>9}".format("rows/yr", "parts/yr", "yrs hist"),
      flush=True)
w = TFRSWorld(seed=11, m=300, alpha=10_000)
K, S, m = w.Kp, S_of(w), 300
n_req = hist_rows(K, S, m)
for rows_yr in (3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000):
    print("     {:>10,} {:>10} {:>9.2f}".format(
        rows_yr, math.ceil(rows_yr / K), n_req / rows_yr), flush=True)
print(flush=True)
print("     At m=300, alpha=10^4: K = {:,}, S = {:.1f}, so a clean f_hat needs "
      "{:.0f} rows.".format(K, S, n_req), flush=True)
print("     A firm reaches that in one year only once it posts ~{:,} rows a "
      "year --- by".format(int(n_req)), flush=True)
print("     which point it already needs {} partitions. There is no firm size "
      "at which".format(math.ceil(n_req / K)), flush=True)
print("     both constraints are slack at once, and a deployer must choose "
      "which to relax.", flush=True)

print(flush=True)
print("     Note the 5-digit row above. Widening the account field is the "
      "obvious way to", flush=True)
print("     cut partitions -- 13 down to 2 -- but n_req grows as K^2, so the "
      "history needed", flush=True)
print("     goes from 0.6 years to 61. A wider field buys a tighter "
      "allocation bound m/2K,", flush=True)
print("     and f_hat must then be estimated well enough to keep up with it. "
      "Widening the", flush=True)
print("     field without the history to match makes Delta worse, not better.",
      flush=True)

# ------------------------------------------------- the benchmark, checked
print(flush=True)
print("  4. OUR OWN BENCHMARK, CHECKED AGAINST THE RULE", flush=True)
w = TFRSWorld(seed=11, m=200, alpha=10_000)
print("     tab:base measures a 20,000-row ledger at m=200, K' = {:,}."
      .format(w.Kp), flush=True)
print("     occupancy = {:.2f}, which is ABOVE the design rule R/K' <= 1."
      .format(20_000 / w.Kp), flush=True)
print("     That is deliberate and it is not a security configuration: "
      "encryption cost", flush=True)
print("     is per row and independent of the partition boundary, so the "
      "throughput", flush=True)
print("     figure is unaffected. A compliant deployment splits those rows "
      "into {}".format(math.ceil(20_000 / w.Kp)), flush=True)
print("     partitions, which changes rows/s not at all and adds two more "
      "anchors.", flush=True)
print(flush=True)
print("done", flush=True)
