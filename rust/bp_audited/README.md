# Audited Bulletproofs cross-check

This executable pins the exact Bulletproofs and curve library versions named
in Quarkslab report `19-06-594-REP`:

- `bulletproofs` 1.0.2, audited commit
  `6a17ceb3bf3ce9b94cfa16a2a1a7311eef2dc6e7`
- `curve25519-dalek` 1.2.1, audited commit
  `45b316d26b64e8dc729b3463911a919cf31c6f4c`
- Rust nightly `2019-06-11`, the report's toolchain

The audit found a high-impact panic when an untrusted multiparty participant
sent a crafted `ProofShare`. This wrapper accepts no multiparty messages and
checks the serialized range-proof length before deserialization and
verification. The audit is evidence about the pinned dependency, not an audit
of this wrapper or Paper B's surrounding protocol code.

Build and run the small regression:

```text
cargo build --release
target/release/paperb-bp-audited --quick
target/release/paperb-bp-audited --medium
```

Run Paper B's two padded batches sequentially over one deterministic balanced
20,000-row ledger:

```text
target/release/paperb-bp-audited --full
```

After both proofs verify, the full mode checks every public per-row `C/D/K`
side link, the amount balance, equal debit/credit totals, a record digest, and
negative controls for an understated side and an altered digested record.
Padding commitments are proved but excluded from the 20,000-row digest. This
is the target-size range-proof and ledger-consistency component. It does not
implement the canonical record serializer, signatures, custodian receipt,
beacon timing, sampling presentation, SQLite persistence, or the complete
protocol verifier; those are composed in the small Python fixtures.

The two batches use Ristretto, whereas the Python measurement instrument uses
P-256. They instantiate the same aggregated Bulletproof range statement but
are intentionally reported as a cross-implementation check, not as
byte-compatible proofs.

The canonical full run in `../../results/bp_audited_full.txt` measured 79.78 s
of generator setup, 676.91 s proving, 55.84 s verification, 3,264 proof bytes
and 2,523 MiB peak resident memory. Timing excludes the final linear ledger
checks; their pass/fail markers are in the same result file.
