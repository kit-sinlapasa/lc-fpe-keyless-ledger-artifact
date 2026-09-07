# Manifest

Paper A promises to release "the generator and all attack code". All of it is
here, including the scripts whose results we later withdrew or corrected. Those
are kept rather than curated away, because several of the paper's claims are
statements about what an earlier measurement got wrong, and a reader should be
able to see the earlier measurement.

Python files live in `code/` and run from that directory so their module-name
imports resolve. The independently audited Rust cross-check is isolated in
`rust/bp_audited/` with its own lockfile and toolchain pin.

---

## Library

| file | role |
|---|---|
| `harness.py` | multi-seed experiment harness; flat Zipf world, `alloc_hamilton`, `cluster_ari`, `ci95` |
| `harness_tfrs.py` | structured TFRS-style chart of accounts: five categories, seven hub accounts, eighteen transaction archetypes |
| `lcfpe_impl.py` | FF1 code layer (encrypt and decrypt, validated against the NIST sample values by `ff1_nist_kat.py`), and the Paillier **comparison baseline** for amounts — not the deployed design |
| `bulletproofs.py` | aggregated Bulletproofs over NIST P-256; also produces Paper B `tab:bp` |
| `hash_to_curve.py` | RFC 9380 `P256_XMD:SHA-256_SSWU_RO_`, used to derive the Bulletproofs generators. Its `self_test` checks the implementation against the RFC's own vectors, intermediate field elements included, and re-derives the curve parameters instead of trusting them |
| `bp_rfc9380_check.py` | validates the P-256 measurement implementation: RFC vectors pass, the soundness tests pass under the derived generators, proof sizes are unchanged, and only setup time moves |
| `paperb_e2e.py` | small in-memory end-to-end fixture: one canonical record list feeds the Merkle root, chain head, signed anchor, beacon sample, interval proof and commitment openings; includes rejection tests |
| `paperb_durable.py` | durable end-to-end research prototype: encrypted SQLite rows, WAL/FULL durability, atomic close, persisted proofs, custodian receipt, future beacon, restart verification and eight fault tests |
| `a_exact_ml.py` | Paper A Proposition `prop:countopt`: closed-form marginal Bayes rule and exact joint maximum-likelihood assignment for one count-only public context cell; explicitly excludes cross-cell co-occurrence |
| `rust/bp_audited/` | exact audited dalek dependency versions, reproducible 2019 Rust toolchain, and the full 20,000-row two-batch verifier; checks both proofs, per-row side links, totals, an anchor digest and negative controls on one record set |
| `bp_width_control.py` | control for `tab:bp`: two equal-$nm$ pairs at $n=32$ and $n=64$ give equal proof sizes, with prover times differing by up to 7.1%; evidence for a cost projection, not identical proof contents |
| `range_configuration.py` | derives the mixed-width full-verifier configuration: $C_i$ at $n_C=64$, $D_i,K_i$ at $n_S=32$, padding, total bit-values and proof sizes |
| `a_freq_bound.py` | scope check for Proposition 1: computes Delta and KL at the operating point and shows that neither subadditivity nor Pinsker carries a per-draw distance to a period of 20,000 rows |
| `a_delta_blind.py` | scope check for Proposition 1: tunes chart size until the two generators agree on Delta, then runs the same clustering attack on both. Same Delta, recovery differing by a factor of 81 |
| `a_impl_audit.py` | correctness audit of the Paper A implementation: LC-FPE round trip, injectivity of the canonical encoding, field-width enforcement, and uniformity of the alias draw |
| `bp_frozen_heart.py` | the Frozen Heart forgery against our own first transcript, which omitted the commitments from the Fiat--Shamir hash. It **must be rejected** by the fixed transcript; it is a negative test, and the flaw it found is reported in Paper B's supplement |
| `f_ablation.py` | provides `build` / `cluster` / `CELLS` to `f_ablation4.py` and `f_ablation5.py`; its own `__main__` runs the first-pass ablation under the sparse pool |

## Canonical — these produce the papers' tables and figures

Run them through `run_all.py`, which records wall-clock and writes output to
`results/`.

| paper | artefact | script |
|---|---|---|
| A | `tab:obs` | `observable_test.py` |
| A | `tab:ari` | `ci_headline.py` |
| A | `tab:ari` (strong column, occupancy 1 and 2) | `ci_decisive.py` |
| A | `tab:ari` (occupancy 103, both configurations) | `ci_curve_ctrl.py` |
| A | `tab:ari` (deployed observable, paired with the alias oracle) | `ci_headline_triple.py` |
| A | `fig:ari` | `ci_curve.py` |
| A | `fig:ari` (occupancy 32–128) | `ci_curve_hi.py` |
| A | `tab:lega` (baseline, rank attack) | `leg_a2.py` |
| A | `tab:comb` (combined adversary, best-of, paired gain) | `a_combined.py` |
| A | `prop:countopt` (25-world scoped optimum) | `a_exact_ml.py` |
| A | `tab:ablation` (clustering column) | `f_ablation5.py` |
| A | `tab:ablation` (recovery column, baseline) | `f_ablation4.py` |
| A | `tab:sens` | `ci_sensitivity.py` |
| A | `tab:drift` (TV, Delta, bound) | `g_drift.py` |
| A | `tab:drift` (ARI column) | `g_drift3.py` |
| A | eq. (psi), repeated-edge predictor | `d_psi.py` |
| A | §Evaluation, ceiling-allocation figures | `fpae_verify4.py` |
| A | `tab:tfrs` | `d_tfrs.py` |
| A | `tab:multiline` | `e_multiline.py` |
| A | `tab:base` | `baseline_bench.py` |
| A | `tab:applic` | `a_applicability.py` |
| A | `tab:chartsize` (occ 1, 2; structure) | `a_chartsize.py` |
| A | `tab:chartsize` (occ 8, 24) | `a_chartsize_hi.py` |
| B | `tab:bp` | `bulletproofs.py` |
| B | `tab:bp` (caption control) | `bp_width_control.py` |
| B | `tab:bp` (cost ladder toward one batch) | `bp_full_partition.py` |
| B | `tab:bp` (mixed-width configuration and proof sizes) | `range_configuration.py` |
| B | `tab:bp` (rows + derived figures) | `gen_tab_bp.py` -> `results/tab_bp_rows.tex`, `results/tab_bp_facts.txt`; post-processor, run after the two measurement scripts, refuses a ladder with no `done` line |
| B | `tab:perf` | `paperb_bench.py` |
| B | `tab:sampling` | `paperb_sampling.py` |
| B | `tab:sampling` (credit side $K_i$, checks (b)/(b′), rejection-sampled units) | `paperb_sampling.py` |
| B | `tab:class` | `paperb_classification.py` |
| B | durable end-to-end fault tests | `paperb_durable.py` |
| B | full target-size verifier on audited dependency | `rust/bp_audited/` -> `results/bp_audited_full.txt` |
| B | full-run totals quoted in `tab:bp` | `bp_audited_summary.py` -> `results/bp_audited_summary.txt` |
| A | correctness | `ff1_nist_kat.py` |

`tab:related` (literature matrix) and `fig:constr` (construction diagram) are
not generated by code.

## Superseded and intermediate

Kept for provenance. Where a script's numbers differ from the paper, the paper
is right and the reason is below.

### Harness controls

| file | status |
|---|---|
| `ci_control.py` | validated the multi-seed harness against the single-seed scripts it replaced |
| `ci_decisive.py` | checked that the occupancy-1 and occupancy-2 figures survived re-measurement |
| `ci_curve_ctrl.py` | showed the ARI $0.88$ figure came from a stronger attack configuration, not from the curve plateauing |
| `ci_curve_hi.py` | extended the curve to occupancy 32–128 |
| `fpae_sens_check.py` | controlled for refinement depth: `refine=3` gives $0.0324\pm0.0088$ against `refine=1`'s $0.0349\pm0.0115$, so depth is not what drives the sweep |

### Task D — why the structured chart is safer

| file | status |
|---|---|
| `d_diag.py` | found the artifact: 128 of 300 accounts were unreachable, which made the structured chart look 100x safer. Fixed by adding eight archetypes |
| `d_mech.py` | measured the co-occurrence graph instead of speculating about it |
| `d_psi.py` | closed-form predictor for repeated alias edges; correlation $+0.9993$ with measurement |
| `d_doseresp.py` | dose-response on the pairing statistic |
| `d_calib.py` | **refuted** the design rule `occupancy <= C/sqrt(Psi)`. The $+0.989$ correlation it was built on came from two clusters of points, not from a relationship. The rule was withdrawn and appears nowhere in the paper |

### Task F — ablation passes

| file | status |
|---|---|
| `f_ablation2.py` | added the chance baseline for the frequency column; showed FPAE's apparent 70% leak is exactly its 71% chance level |
| `f_ablation3.py` | found the cost-centre artifact — a ciphertext-free adversary already scored 87.6% — which invalidated every earlier "adversary knows $(c,d)$" number |

### Claim verification rounds

| file | status |
|---|---|
| `fpae_verify.py`, `fpae_verify2.py`, `fpae_verify3.py` | successive rounds on Claims 2–4; superseded by `fpae_verify4.py`, which fixed a tie-breaking bug in `argsort` |
| `fpae_verify4.py` | final verification of the allocation claims |

### Attack development

| file | status |
|---|---|
| `fpae_attack_ml.py` | scheme-aware maximum-likelihood frequency matcher |
| `fpae_attack_cooc.py`, `fpae_attack_cooc2.py` | co-occurrence AUC; round 2 fixed a rank-handling error for tied counts |
| `fpae_attack_cluster.py`, `fpae_attack_cluster2.py` | first full clustering attack, single-seed; superseded by the harness version in `ci_headline.py` |
| `leg_a_conditional_domain.py` | **superseded by `leg_a2.py`.** It drew each account's cost centres from a pool of 100, so a $(c,d)$ cell held 1.9 accounts and knowing $(c,d)$ almost named the account; it also ranked candidates by the empirical histogram of the rows under attack. Its 100% figure is an artifact and does not appear in the paper |
| `fpae_sensitivity.py` | single-seed predecessor of `ci_sensitivity.py` |
| `ci_headline.py` section 3 | the conditional-domain block inside this script still uses the sparse cost-centre pool and prints the artifact-inflated $100\%$ column. Its sections 1 and 2 produce `tab:ari`; for `tab:lega` read `leg_a2.py` instead |

### Rejected designs and the amount layer

| file | status |
|---|---|
| `fpae_cpa.py`, `fpae_cpa2.py` | Counter-Partitioned Aliasing, a proposed fix for the clustering attack. **Rejected**: it made recovery worse (ARI $0.424 \to 0.844$), because the adversary exploits graph structure rather than profile similarity |
| `fpae_linking.py`, `fpae_linking2.py` | cross-period linking-token trade-off; round 2 fixed a bucketing error in round 1 |
| `pedersen_bench.py` | Pedersen commitments benchmarked against the Paillier baseline |
| `offset_bench.py` | offset encoding $v + 2^{40}$ to keep elliptic-curve scalars short |
| `scalar_test.py` | scalar-field helper check |
| `g_drift2.py` | intermediate drift pass; its frequency instrument had a $\pm38$–$57$ point interval and was replaced by the Spearman statistic in `g_drift3.py` |

---

## A warning about the generator defaults

`harness.py` and `harness_tfrs.py` take `pool_cc` and `pool_dt`, which set how
many distinct cost centres and document types exist in total. **The defaults are
100 and 10, which are the values every pre-task-F experiment used**, and they
are deliberately left alone so those experiments reproduce exactly.

They are not realistic. A real chart has a handful of departments that every
account posts to. `leg_a2.py` and `f_ablation4.py` therefore pass `pool_cc=8,
pool_dt=12` explicitly, and Paper A's `tab:lega` and `tab:ablation` are measured
at those values. Do not "fix" the defaults: experiments that never read the cost
centre at all — which is all of the occupancy and clustering work — would then
draw from a different random stream and stop matching the paper.
