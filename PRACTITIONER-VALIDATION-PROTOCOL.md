# Practitioner Validation Protocol for Paper B

This is a protocol for a future independent practitioner study, not a completed
evaluation.

## Participants and materials

Recruit practising external auditors who have performed general-ledger
sampling and trial-balance work. Record role, years of experience, firm size,
and familiarity with monetary-unit sampling. Use a frozen public artifact
commit, synthetic ledgers only, a short threat-model briefing, and a scripted
auditor interface that exposes no secret key.

## Tasks

1. Verify a period close, custodian receipt, previous-anchor state, range
   proofs, and period balance.
2. Verify one claimed account balance and reject a presentation containing a
   row committed to another account.
3. Reproduce a debit- and credit-side monetary-unit sample from an
   authenticated beacon statement and vouch the disclosed rows.
4. Diagnose one altered row, one late receipt, one forged beacon statement,
   and one omitted material posting.
5. Explain, without prompting, which audit assertions the evidence does and
   does not support.

## Outcomes

Measure task completion, verification errors, time, requests for plaintext or
keys, System Usability Scale, and whether participants correctly distinguish
post-close state integrity from transaction truth and completeness at entry.
Collect qualitative feedback on custody, evidence retention, exception
handling, and fit with ISA 500/530 procedures.

## Acceptance and reporting

Pre-register sample size, hypotheses, exclusion rules, and analysis before
recruitment. Report every task failure and negative result. The paper may claim
practitioner validation only after ethics/consent requirements are met, the
materials and anonymised results are archived, and at least one auditor who was
not an author independently completes all critical verification tasks. Until
then Paper B must continue to state that no practising-auditor evaluation has
been completed.
