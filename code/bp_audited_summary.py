#!/usr/bin/env python3
"""Derive the manuscript totals from the canonical audited-backend full run."""

from pathlib import Path


RESULT = Path(__file__).resolve().parents[1] / "results" / "bp_audited_full.txt"


def main():
    batches = []
    current = None
    checks = {}

    for raw in RESULT.read_text(encoding="utf-8").splitlines():
        if not raw or "=" not in raw:
            continue
        fields = dict(part.split("=", 1) for part in raw.split() if "=" in part)
        if raw.startswith("batch="):
            current = fields
            batches.append(current)
        elif current is not None and raw.startswith(("setup_s=", "prove_s=", "verify_s=", "proof_bytes=")):
            key, value = raw.split("=", 1)
            current[key] = value
        else:
            key, value = raw.split("=", 1)
            checks[key] = value

    if len(batches) != 2:
        raise SystemExit("expected exactly two proof batches")
    if checks.get("exit_code") != "0":
        raise SystemExit("full run did not finish successfully")

    setup = sum(float(batch["setup_s"]) for batch in batches)
    prove = sum(float(batch["prove_s"]) for batch in batches)
    verify = sum(float(batch["verify_s"]) for batch in batches)
    proof_bytes = sum(int(batch["proof_bytes"]) for batch in batches)
    rows = int(checks["ledger_rows"])

    print("source=bp_audited_full.txt")
    print("batches={}".format(len(batches)))
    print("total_setup_s={:.2f}".format(setup))
    print("total_prove_s={:.2f}".format(prove))
    print("total_verify_s={:.2f}".format(verify))
    print("total_proof_bytes={}".format(proof_bytes))
    print("proof_bytes_per_row={:.4f}".format(proof_bytes / rows))
    print("peak_rss_mb={}".format(checks["peak_rss_mb"]))
    print("all_full_run_checks_passed=true")


if __name__ == "__main__":
    main()
