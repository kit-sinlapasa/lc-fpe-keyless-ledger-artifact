# Artefact: LC-FPE and the auditable encrypted ledger

Code for both papers. Everything measured in either paper is produced by a
script here, and `MANIFEST.md` maps each table and figure to the script that
produced it.

**No real accounting data is used anywhere.** Every number comes from the
synthetic generators in `code/harness.py` and `code/harness_tfrs.py`. The
company whose books motivated this work declined to have its ledger used for
evaluation, so the parameter space is swept instead of a single dataset being
reported — see Section "Evaluation" of Paper A.

---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py --list          # show which script produces which table
python run_all.py tab:drift       # regenerate one table
python run_all.py                 # regenerate everything
```

Output goes to `results/<script>.txt`. The runner prints wall-clock per script
and a summary at the end.

The independently audited Bulletproofs cross-check is a separately pinned
Rust build:

```text
cd rust/bp_audited
cargo build --release
target/release/paperb-bp-audited --quick
target/release/paperb-bp-audited --full
```

The full run's immutable per-batch output is `results/bp_audited_full.txt`.
`code/bp_audited_summary.py` derives the table totals from that file and writes
`results/bp_audited_summary.txt`; no manuscript total is transcribed by hand.

### Measured runtimes

The long Python experiment suite was run end to end for the release snapshot;
the changed security and verifier scripts were rerun again after the final
protocol fixes. The historical long-suite total was **210 minutes** on one
laptop core (Windows 11, Python 3.11.15). The audited Rust full-verifier run is
listed separately because it uses a pinned 2019 toolchain and Ristretto rather
than the Python P-256 measurement code.

All clustering figures were regenerated on 6 September 2026 after `svds` was
given an explicit `random_state`; see **Reproducibility** below. The times in
this table are from that run.

| script | artefact | seconds |
|---|---|---|
| `ff1_nist_kat.py` | correctness | 0.4 |
| `bp_frozen_heart.py` | soundness | 1.3 |
| `paperb_bench.py` | `tab:perf` | 0.9 |
| `a_impl_audit.py` | correctness | 4.1 |
| `f_ablation4.py` | `tab:ablation` | 11.9 |
| `a_applicability.py` | `tab:applic` | 5.1 |
| `baseline_bench.py` | `tab:base` | 172.7 |
| `paperb_classification.py` | `tab:class` | 10.8 |
| `leg_a2.py` | `tab:lega` | 25.7 |
| `a_combined.py` | `tab:comb` | 68.2 |
| `a_exact_ml.py` | `prop:countopt` | see result |
| `bulletproofs.py` | `tab:bp` | 28.9 |
| `observable_test.py` | `tab:obs` | 35.6 |
| `ci_headline.py` | `tab:ari` | 85.4 |
| `ci_decisive.py` | `tab:ari` | 208.8 |
| `ci_curve_ctrl.py` | `tab:ari` | 94.9 |
| `ci_headline_triple.py` | `tab:aridep` | 2,005.6 |
| `f_ablation5.py` | `tab:ablation` | 206.4 |
| `g_drift.py` | `tab:drift` | 225.7 |
| `ci_curve.py` | `fig:ari` | 567.4 |
| `g_drift3.py` | `tab:drift` | 440.5 |
| `paperb_sampling.py` | `tab:sampling` | see result | (runs the 4,096-row debit/credit sampling fixture, both consistency checks, the amount-plus-one negative test, and the C/D/K counterexample; clean rerun completed)
| `paperb_e2e.py` | `e2e` | 8.1 | small canonical record/anchor/beacon/range-proof/opening path; all rejection checks pass
| `paperb_durable.py` | durable `e2e` | see result | atomic close, restart, receipt, beacon, full small-ledger proofs and eight fault tests
| `rust/bp_audited --full` | full verifier | 676.9 prove + 55.8 verify | 20,000 rows, two full padded batches, 3,264 B, 2,523 MiB peak
| `ci_sensitivity.py` | `tab:sens` | 606.0 |
| `d_tfrs.py` | `tab:tfrs` | 1,068.1 |
| `e_multiline.py` | `tab:multiline` | 790.6 |
| `a_chartsize_hi.py` | `tab:chartsize` | 3,254* |
| `a_chartsize.py` | `tab:chartsize` | 4,370* |

\* The two chart-size scripts were run directly rather than through the runner; their times are the sums of the per-configuration seconds each script prints, not a runner wall-clock. Full k-means at $m=1200$ dominates both.

The four correctness and soundness scripts run in under five seconds between
them, so check those first: `ff1_nist_kat.py` must report 9 of 9,
`bp_frozen_heart.py` must show the *pair* — the original prover and verifier
accept the forgery, the fixed ones reject it — and
`a_impl_audit.py` must report a clean round trip.

Measured with Python 3.11.15 on Windows 11; `requirements.txt` pins the exact
library versions used. Every experiment seeds its own `numpy.random.Generator`,
so the generated worlds are fixed by the seed and do not depend on library RNG
defaults.

The clustering results are now bit-identical under the pinned environment:
every `svds` call receives the world's seed as `random_state`, and every
k-means run is seeded. Results produced before 6 September 2026 can differ in
the third decimal; the manuscripts and canonical result files use the seeded
run throughout.

---

### Checking the numbers

`check_claims.py` extracts every number the two manuscripts assert and every
number the canonical result files contain, and reports manuscript numbers
that appear in no result file:

```
python check_claims.py            # report
python check_claims.py --strict   # exit 1 if any are unsourced
```

It exists because a stale number was the most common defect across five
review rounds, and neither `pdflatex` nor a careful re-reading catches one:
the floor fractions left over from the allocator we replaced, the anchor
still quoted at 140 bytes, the prover times from before the credit-side
commitment, a cross-check quoting figures no script had produced for months.
Every one of those is exactly what this reports.

The check is deliberately blunt. It does not know which number belongs in
which cell, so it cannot catch a real number placed in the wrong row; it
catches the number that is not real anywhere. Numbers that no script can
produce -- field widths, standards identifiers, the ARI budget we chose,
quantities derived by arithmetic from other results -- are listed in the
script's WHITELIST with the reason, and each derived entry names the
arithmetic so a reader can redo it.

## What the papers claim, and where to check it

Paper A's design rule is that a partition should hold **at most one row per
alias**. The two experiments that carry it are `ci_headline.py` (`tab:ari`) and
`ci_sensitivity.py` (`tab:sens`, fourteen configurations); both give the
adversary the alias as node identity, which is an oracle. `ci_headline_triple.py`
(`tab:aridep`) reruns the headline sweep on the same worlds under the deployed
observable — the equality class of the ciphertext triple, i.e. of `(alias, c, d)`
under one tweak — paired per seed: recovery is within ±0.001 of zero up to
occupancy 4, 0.011 at 16, and converges with the oracle only at 103. The
alias-only tables are therefore an upper bound and the rule is set against it.
Everything else tests whether that rule survives a harder setting:

| question | script |
|---|---|
| does it survive a realistic chart of accounts? | `d_tfrs.py` |
| does it survive multi-line vouchers? | `e_multiline.py` |
| which mechanism actually buys the protection? | `f_ablation4.py`, `f_ablation5.py` |
| what does a stale frequency profile cost? | `g_drift.py`, `g_drift3.py` |
| what does the adversary recover from the ciphertext it actually holds, not the alias? | `ci_headline_triple.py` |
| what if the adversary knows the row's context? | `leg_a2.py` |
| what if it pools context, profile, multiplicity and the equality pattern? | `a_combined.py` |
| what is optimal when one cell reveals only an equality-class count? | `a_exact_ml.py` |
| can a given firm deploy this at all? | `a_applicability.py` |

`a_applicability.py` produces Paper A's deployment section. It starts with an
instrument check that matters more than it looks: $S$, the multinomial constant
in the history rule, is recomputed from the same TFRS generator the paper's
other tables use and must match the values the paper quotes ($8.71$ against
$8.7$ at $m=200$, $10.20$ against $10.2$ at $m=300$). Everything after that is
arithmetic on rules stated elsewhere in the paper, and the point is what the
arithmetic reveals: the partition rule $R\lelpha-m$ and the history rule
$n\ge 2K^2S^2/\pi m^2$ pull in opposite directions with firm size, so no firm
sits comfortably inside both. The script also checks the paper's own
$20{,}000$-row benchmark against the paper's own occupancy rule and reports
that it exceeds it — deliberately, because throughput does not depend on the
partition boundary.

Paper B's measured tables are `bulletproofs.py` and the pinned Rust cross-check
(`tab:bp`), `paperb_bench.py` (`tab:perf`), `paperb_sampling.py`
(`tab:sampling`) and `paperb_classification.py` (`tab:class`).

`paperb_sampling.py` carries the sampling protocol of Paper B
Section~"Public-Coin Sampling". It has four parts, and the second is the
argument: a firm that merely faces an *unpredictable* coin can re-close its
books until the sample misses the rows it wants hidden, and the script measures
how cheap that is — sixteen re-closes take the escape probability from $0.13$ to
$0.89$. The measured single-close figure, $0.128\pm0.010$, agrees with the
closed form $\exp(-S|F|/N) = 0.135$, which is what makes the measurement worth
trusting. The fix is that the anchor names the beacon round it will use. Part
three selects by monetary unit over amounts that stay committed, and every
configuration must print `yes` in the soundness column.

Unlike the clustering scripts this one is fully deterministic — the beacon, the
close timestamp, the signing key and the row values are all fixed seeds — so
everything but the wall-clock columns reproduces exactly. The population is
the construction's own commitments, not a convenient stand-in: signed amounts
with an offset in $C_i$, plus debit and credit commitments $D_i,K_i$. All
4,096 public per-row side links are checked, as are both population totals.
The old four-row counterexample still passes the global total equations and
hides both large rows from MUS, but fails the new per-row link. For each sampled
unit the proof, interval, both boundary commitments, neighbouring-row negative
control and the opened $C_i,D_i,K_i$ values must all agree. Part 4 retains the
mass-shifting diagnostic and shows that it never lowers detection below the
honest population.

`paperb_classification.py` carries Paper B Section "Verifiable Classification",
which splits the per-account balance gap in two. Substitution — claiming a row
that belongs to another account — is closed deterministically by opening the
account commitment, and the script's negative controls must both fail: a
foreign row must be rejected, and 2,000 forged blinding factors must be
rejected. Omission is only bounded, by monetary-unit sampling, and the measured
detection rates must track the closed form $1-(1-M/T)^s$ across the sweep. The
part worth reading is the last block: the bound reduces to the classical MUS
sample-size rule $s=\mathrm{RF}/(M/T)$ that auditors already use — 299 against
300 at $M/T=1\%$ — which is the paper's argument that it imposes no new audit
methodology. Deterministic; reproduces exactly apart from wall-clock.

`ff1_nist_kat.py` is a correctness check rather than a table: it
reproduces all nine FF1 sample values NIST publishes with SP 800-38G,
in both directions, across the three AES key sizes and radices 10 and 36.
Run it first if you change anything in the cipher layer.

---

## Three things a reader should know before changing anything

**1. The cost-centre pool defaults are deliberately unrealistic.**
`harness.py` and `harness_tfrs.py` default to `pool_cc=100, pool_dt=10`, the
values every experiment written before the ablation used. A real chart has a
handful of departments that every account posts to, and at these defaults a
`(cost centre, document type)` cell holds only 1.9 accounts — so knowing the
context nearly names the account by itself. `leg_a2.py` and `f_ablation4.py`
pass `pool_cc=8, pool_dt=12` explicitly, and the paper's `tab:lega` and
`tab:ablation` are measured there. The defaults are left alone so that the
occupancy and clustering experiments, none of which read the cost centre,
reproduce exactly. Changing them silently changes the random stream.


### Reproducibility of the clustering numbers

Every spectral attack in this artefact calls `scipy.sparse.linalg.svds`, whose
ARPACK iteration starts from a random vector unless `random_state` is set. Until
6 September 2026 it was not set, so a fixed seed did *not* give a fixed ARI:
three repeats of the same world at occupancy 1 returned 0.0221, 0.0248 and
0.0230. Every `svds` call now passes the world's seed, and a repeated run is
bit-identical. Tables produced before that change differ from the current ones
in the third decimal; the paper reports the seeded run throughout.

**2. Recovery numbers without a baseline are not measurements.**
Several scripts report a *ciphertext-free baseline*: what an adversary scores
by ignoring the ciphertext and using side information alone. On the realistic
pool that baseline is already 59.4%. Any recovery figure has to be read against
it — FPAE's 2.2% is *below* the baseline for the tested heuristic, which does
not mean that all ciphertext information is absent or that recovery is simply 2%.
`a_combined.py` addresses the objection that a rational adversary is not bound
to one heuristic by evaluating a prespecified per-cell Bayesian assignment of
ciphertext equality classes to accounts, using row count, class sizes, the chart
profile and the public `k(a)`. It reports the paired gain over the operational
context-only MAP baseline, while retaining the empirical-majority result only
as an oracle control. In the realistic pool the tested contrasts are -3.40,
-1.34, -0.22, -0.00 and 0.00 percentage points across the five cells; the same
attacker recovers 61.6% without FPAE in the 5x3 control. These are tested-
heuristic contrasts, not bounds on an optimal adversary or on all ciphertext
information.

**3. Measuring at a point where the metric is pinned gives a null result
regardless.** At occupancy 1 the adjusted Rand index is near zero in every
configuration we have tested, so a mechanism can look irrelevant there purely
for want of room to move. `g_drift.py` runs at occupancy 1 and finds no effect
of frequency drift; `g_drift3.py` runs the same sweep at occupancy 16 and finds
one. The paper reports the second.

---

## Superseded code is included on purpose

Paper A promises "the generator and all attack code", and several of its claims
are about what an earlier measurement got wrong: a design rule withdrawn after
calibration, a mitigation that made recovery worse, a conditional-domain figure
that turned out to be a generator artifact. The scripts behind those are in
`code/` alongside the current ones, and `MANIFEST.md` says for each what
replaced it and why. Where a superseded script disagrees with the paper, the
paper is right.

---

## Licence

Released under the MIT Licence (`LICENSE`). Change it before submission if your
institution requires otherwise.
