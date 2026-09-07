#!/usr/bin/env python3
"""
Regenerate every measured table and figure in both papers.

Each entry below names the paper artefact it produces, and the script is run in
`code/` with its own working directory so that its imports resolve.  Wall-clock
is recorded per script; the README quotes the times this runner measured, not
estimates.

    python run_all.py              run everything
    python run_all.py tab:drift    run only what produces that table
    python run_all.py --list       show the map and exit

Output of each script is written to results/<script>.txt alongside its timing.
Nothing here reads a real ledger: every number in both papers comes from the
synthetic generators in code/harness.py and code/harness_tfrs.py.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
OUT = os.path.join(HERE, "results")

# (paper, label, script, what it produces)
JOBS = [
    ("A", "tab:obs",       "observable_test.py",
     "effect of the observable on clustering recovery"),
    ("A", "tab:ari",       "ci_headline.py",
     "clustering recovery against occupancy, both adversary configurations"),
    ("A", "tab:ari",       "ci_decisive.py",
     "strong adversary configuration (full k-means, 64 dims, 3 refinements) at occupancy 1 and 2"),
    ("A", "tab:ari",       "ci_curve_ctrl.py",
     "both adversary configurations at occupancy 103, three seeds"),
    ("A", "tab:ari",       "ci_headline_triple.py",
     "headline clustering table under the deployed (alias,c,d) observable, paired with the alias oracle"),
    ("A", "fig:ari",       "ci_curve.py",
     "ARI-vs-occupancy curves, dense and sparse charts"),
    ("A", "fig:ari",       "ci_curve_hi.py",
     "dense curve continued to occupancy 32, 64 and 128"),
    ("A", "tab:lega",      "leg_a2.py",
     "conditional-domain adversary with the ciphertext-free baseline"),
    ("A", "tab:lega",      "a_combined.py",
     "combined Bayesian adversary (context + profile + k(a) + equality pattern); best-of and paired gain"),
    ("A", "prop:countopt", "a_exact_ml.py",
     "closed-form marginal Bayes rule and exact joint ML assignment in one count-only cell"),
    ("A", "tab:ablation",  "f_ablation5.py",
     "ablation, clustering column (realistic cost-centre pool)"),
    ("A", "tab:ablation",  "f_ablation4.py",
     "ablation, row-recovery column and the ciphertext-free baseline"),
    ("A", "tab:sens",      "ci_sensitivity.py",
     "14-configuration one-factor-at-a-time sweep"),
    ("A", "tab:drift",     "g_drift.py",
     "drift: TV(f,fhat), realised Delta, and the bound"),
    ("A", "tab:drift",     "g_drift3.py",
     "drift: ARI column at occupancy 16, seven seeds"),
    ("A", "eq:psi",        "d_psi.py",
     "closed-form repeated-edge predictor against measured counts, and Psi per chart"),
    ("A", "tab:tfrs",      "d_tfrs.py",
     "structured TFRS-style chart of accounts"),
    ("A", "tab:multiline", "e_multiline.py",
     "multi-line vouchers against a realistic line-count mix"),
    ("A", "sec:eval",      "fpae_verify4.py",
     "ceiling versus budget-balanced allocation: scale term and top-10 identification"),
    ("A", "tab:base",      "baseline_bench.py",
     "five schemes on the same 20,000-row ledger"),
    ("A", "tab:applic",    "a_applicability.py",
     "applicability checklist, worked deployments, partition/history tension"),
    ("A", "tab:chartsize", "a_chartsize.py",
     "chart size m=200..1200 at occupancy 1 and 2, floor fraction, Delta"),
    ("A", "tab:chartsize", "a_chartsize_hi.py",
     "chart size at occupancy 8 and 24, where the structured chart has room to move"),
    ("B", "tab:bp",        "bulletproofs.py",
     "aggregated Bulletproofs, measured size and time"),
    ("B", "soundness",     "bp_rfc9380_check.py",
     "RFC 9380 vectors, soundness under the new generators, and the effect on timings"),
    ("B", "tab:bp",        "bp_width_control.py",
     "control: equal n*m at n=32 and n=64 costs the same across the two proof batches"),
    ("B", "tab:bp",        "range_configuration.py",
     "mixed-width verifier configuration, padding and two-proof size"),
    ("B", "tab:bp",        "bp_full_partition.py",
     "n=32 cost ladder toward one partition-level proof batch (HOURS)"),
    ("B", "tab:bp-full-summary", "bp_audited_summary.py",
     "derived totals from the canonical audited-backend full run"),
    ("B", "tab:perf",      "paperb_bench.py",
     "cost of the tamper-evidence layer"),
    ("B", "tab:sampling",  "paperb_sampling.py",
     "public-coin selection, grinding measurement, monetary-unit sampling"),
    ("B", "e2e",           "paperb_e2e.py",
     "role-separated 16-row fixture with canonical records, proofs, anchor, receipt, beacon and openings"),
    ("B", "e2e-durable",   "paperb_durable.py",
     "durable SQLite fixture: public-key verifier, atomic close, restart, receipts, debit/credit beacon paths, full-ledger proofs and 14 tests"),
    ("B", "tab:class",     "paperb_classification.py",
     "account commitments: substitution rejected, omission bounded by MUS"),
    ("A", "scope",         "a_freq_bound.py",
     "can a per-draw distance bound carry to a period of N rows? (it cannot)"),
    ("A", "scope",         "a_delta_blind.py",
     "two ledgers matched on Delta whose recoverability differs sharply"),
    ("A", "correctness",   "a_impl_audit.py",
     "LC-FPE round trip, injectivity, field width, alias uniformity, key separation and RowID encoding"),
    ("B", "soundness",     "bp_frozen_heart.py",
     "weak Fiat-Shamir forgery: must be REJECTED by the fixed transcript"),
    ("A", "correctness",   "ff1_nist_kat.py",
     "FF1 known-answer test against the nine NIST SP 800-38G sample values"),
]


def show():
    print("{:<6} {:<14} {:<22} {}".format("paper", "artefact", "script", "produces"))
    for p, lab, s, d in JOBS:
        print("{:<6} {:<14} {:<22} {}".format(p, lab, s, d))


def main(argv):
    if "--list" in argv:
        show()
        return 0
    sel = [a for a in argv[1:] if not a.startswith("-")]
    jobs = [j for j in JOBS if not sel or j[1] in sel or j[2] in sel]
    if not jobs:
        print("nothing matches {}; try --list".format(sel))
        return 1
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p, lab, s, d in jobs:
        path = os.path.join(CODE, s)
        if not os.path.exists(path):
            print("MISSING {}".format(s))
            rows.append((s, lab, "missing", 0.0))
            continue
        print("[{}/{}] {:<22} {}".format(p, lab, s, d), flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, s], cwd=CODE,
                           capture_output=True, text=True)
        dt = time.time() - t0
        with open(os.path.join(OUT, s.replace(".py", ".txt")), "w",
                  encoding="utf-8") as fh:
            fh.write(r.stdout)
            if r.stderr:
                fh.write("\n--- stderr ---\n" + r.stderr)
        status = "ok" if r.returncode == 0 else "FAILED rc={}".format(r.returncode)
        print("      {:<12} {:.1f}s".format(status, dt), flush=True)
        rows.append((s, lab, status, dt))

    print()
    print("{:<22} {:<14} {:<14} {:>8}".format("script", "artefact", "status", "seconds"))
    for s, lab, st, dt in rows:
        print("{:<22} {:<14} {:<14} {:>8.1f}".format(s, lab, st, dt))
    bad = [r for r in rows if r[2] != "ok"]
    print()
    print("{} of {} scripts completed".format(len(rows) - len(bad), len(rows)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
