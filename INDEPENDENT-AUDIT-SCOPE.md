# Independent Audit Scope

This document defines an external review; it is not evidence that the review
has occurred. Any public report must identify the audited commit hash and must
not inherit conclusions from the audit of the pinned dalek dependencies.

## In scope

- `code/paperb_protocol.py`: canonical RowID, 228-byte record, 135-byte anchor
  envelope, 199-byte firm anchor, and 271-byte custodian receipt.
- `code/lcfpe_impl.py` and `code/baseline_bench.py`: HKDF key separation,
  rejection-sampled aliases, 512-bit hash-to-scalar, injective nonces, FF1,
  AEAD associated data, and commitment recomputation.
- `code/bulletproofs.py`: transcript construction, statement binding,
  canonical parsing assumptions, and P-256 generator derivation.
- `code/paperb_e2e.py` and `code/paperb_durable.py`: composition of records,
  proofs, signatures, previous-anchor state, receipt timing, authenticated
  beacon timing, account presentations, monetary-unit sampling, persistence,
  crash recovery, and separation of writer/prover from verifier.
- `rust/bp_audited/`: wrapper code and the boundary between the wrapper and the
  independently audited dependency versions.

## Required review questions

1. Can two different field tuples encode to one RowID, record, anchor, or
   receipt, or can a parser accept a non-canonical encoding?
2. Can any key, nonce, PRF input, Fiat-Shamir transcript, or generator be
   reused across domains in a way the papers exclude?
3. Can a writer make `C`, `D`, and `K` pass their range proofs and aggregate
   totals while understating a row's relevant sampling mass?
4. Do all no-wrap guards cover the integer intervals used by balance,
   side-link, and interval-proof arguments?
5. Can an anchor be replayed across entity/period, replace a previous anchor,
   be countersigned after beacon opening, or be sampled before authenticated
   beacon publication?
6. Does either verifier access a data/blinding/signing secret, plaintext rows
   outside the disclosed sample, or unanchored state?
7. Can malformed proof lengths, points, scalars, SQLite values, or interrupted
   transactions cause acceptance, unsafe allocation, or state equivocation?

## Minimum methods

- Independent line-by-line review against `paper/paperA.tex` and
  `paper/paperB.tex` from the parent project.
- Differential tests for every serializer in a second language.
- Property-based and coverage-guided fuzzing of parsers and state transitions.
- Mutation tests for signatures, receipts, proofs, commitments, row ordering,
  totals, sample indices, beacon statements, and previous-anchor state.
- Independent cryptographic test vectors for FF1, RFC 9380 hash-to-curve,
  AES-GCM, Ed25519, HKDF, and both Bulletproof backends.
- Crash and restart tests at every write boundary in period close and receipt
  insertion.

## Deliverables and closure rule

The reviewer should publish a report containing the commit hash, environment,
methods, findings, severity, reproducer, and disposition. This audit is closed
only when every critical/high finding is fixed and retested, medium findings
are fixed or explicitly accepted, and the final report names the retested
commit. Until then the papers must retain the limitation that the wrapper,
SQLite prototype, and protocol composition have not been independently
audited.
