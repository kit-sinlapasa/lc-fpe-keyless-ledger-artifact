#!/usr/bin/env python3
"""
Soundness attack on our own aggregated Bulletproof: weak Fiat-Shamir.

The transcript is opened with the proof's own commitments only,

    ts = TS(b"bp/v1").add(A, S)

so the challenges y, z, x are independent of the commitments V being proven.
That is the "Frozen Heart" flaw disclosed against several Bulletproofs
implementations in 2022: the Fiat-Shamir hash must bind every public input of
the statement, and V is a public input.

The verifier's first equation is

    that*G + taux*H  ==  sum_j z^(2+j) V_j  +  delta*G  +  x*T1  +  x^2*T2

and V appears nowhere else -- the inner-product argument never touches it. With
m = 2 the equation constrains only the weighted sum, so for any point D a
prover who knows z can move mass between the two commitments:

    V'_0 = V_0 + D           weight z^2
    V'_1 = V_1 - z^-1 * D    weight z^3     ->  z^2 D - z^3 z^-1 D = 0

The weighted sum is unchanged, the proof still verifies, and V'_0 now commits
to v_0 + d for a d of the attacker's choosing -- including values far outside
the n-bit range the proof claims to certify. Knowing z before choosing V is
exactly what the missing transcript binding permits.

Run this against the fixed implementation and the forgery must fail.
"""
import os
from bulletproofs import BP, TS, Q, mul

n, m = 16, 2
bp = BP(n, m)

gs = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(m)]
vs = [5, 7]
Vs = [bp.commit(vs[i], gs[i]) for i in range(m)]
pf = bp.prove(vs, gs)

print("=" * 76, flush=True)
print("WEAK FIAT-SHAMIR ATTACK on the aggregated range proof", flush=True)
print("=" * 76, flush=True)
print("  range is {} bits, so the honest values 5 and 7 are in range"
      .format(n), flush=True)
ok, _ = bp.verify(Vs, pf)
print("  honest proof verifies                     : {}".format(ok), flush=True)

# Recover the verifier's z.  The attacker can do this because the transcript
# up to z depends only on A and S, both of which the attacker produced.
ts = TS(b"bp/v1").add(pf["A"], pf["S"])
ts.chal(b"y")
z = ts.chal(b"z")
zinv = pow(z, Q - 2, Q)

# Shift an out-of-range amount into the first commitment.
d = 1 << 200
D = mul(bp.G, d)
forged = [Vs[0] + D, Vs[1] + mul(D, (-zinv) % Q)]

ok2, why = bp.verify(forged, pf)
print("  forged commitments, FIXED verifier        : {}  {}"
      .format(ok2, "" if ok2 else "(rejected: " + why + ")"), flush=True)

# The same attack against the implementation as we first shipped it. The
# prover and verifier both used the transcript that omitted (n, m, V), so z
# is recoverable from (A, S) alone and the shifted commitments verify.
pf_w = bp.prove(vs, gs, weak=True)
ts_w = TS(b"bp/v1").add(pf_w["A"], pf_w["S"])
ts_w.chal(b"y")
z_w = ts_w.chal(b"z")
forged_w = [Vs[0] + D, Vs[1] + mul(D, (-pow(z_w, Q - 2, Q)) % Q)]
ok_w_honest, _ = bp.verify(Vs, pf_w, weak=True)
ok_w_forged, _ = bp.verify(forged_w, pf_w, weak=True)
ok_w_fixed, _ = bp.verify(forged_w, pf_w)
print(flush=True)
print("  ORIGINAL (weak) prover + ORIGINAL verifier", flush=True)
print("    honest commitments                      : {}".format(ok_w_honest),
      flush=True)
print("    forged commitments (5 + 2^200)          : {}  <-- accepted: the flaw"
      .format(ok_w_forged), flush=True)
print("  same weak proof against the FIXED verifier: {}  (rejected)"
      .format(ok_w_fixed), flush=True)
assert ok_w_honest and ok_w_forged and not ok_w_fixed and not ok2, \
    "vulnerable/fixed pair must split: weak accepts the forgery, fixed rejects"
print("  first forged commitment now holds 5 + 2^200, far outside {} bits"
      .format(n), flush=True)
print(flush=True)
if ok2:
    print("  RESULT: SOUNDNESS BROKEN -- the proof certifies a range it does "
          "not hold", flush=True)
else:
    print("  RESULT: attack fails, the transcript binds the commitments",
          flush=True)
raise SystemExit(1 if ok2 else 0)
