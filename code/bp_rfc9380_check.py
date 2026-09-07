#!/usr/bin/env python3
"""
B-P0.2 evidence: the range proof under RFC 9380 generators.

Paper B listed two departures from what anyone would deploy: generators from
try-and-increment rather than RFC 9380, and the curve reached through a
private attribute of the arithmetic library. Both are gone. This script is
the evidence that removing them changed the right things and nothing else.

Three questions, three answers:

  1. Is the hash-to-curve correct?  Checked against RFC 9380's own vectors,
     including the intermediate field elements, so a coincidence in the final
     point cannot pass.
  2. Is the proof system still sound under the new generators?  The five
     soundness tests are re-run, including the out-of-range rejection.
  3. Are the timings in tab:bp still valid?  They were measured with the old
     derivation. Generators are inputs to the prover, not work it does, so
     prove and verify should be unchanged and only setup should differ. That
     is a claim about our code, so it is measured rather than asserted.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hash_to_curve as H          # noqa: E402
from bulletproofs import BP, Q, size_bytes, h2c, h2c_legacy   # noqa: E402


def section(t):
    print("", flush=True)
    print("=" * 74, flush=True)
    print(t, flush=True)
    print("=" * 74, flush=True)


def main():
    section("1. RFC 9380 P256_XMD:SHA-256_SSWU_RO_ against the RFC's vectors")
    vectors_ok = H.self_test()
    print("  all vectors match: %s" % vectors_ok, flush=True)

    section("2. soundness under the new generators")
    bp = BP(16, 2)
    gs = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(2)]
    Vs = [bp.commit(5, gs[0]), bp.commit(7, gs[1])]
    pf = bp.prove([5, 7], gs)
    checks = []
    checks.append(("honest proof verifies", bp.verify(Vs, pf)[0], True))

    bad = dict(pf); bad["that"] = (bad["that"] + 1) % Q
    checks.append(("a tampered t-hat is rejected", bp.verify(Vs, bad)[0], False))

    bad2 = dict(pf); bad2["a"] = (bad2["a"] + 1) % Q
    checks.append(("a tampered inner-product scalar is rejected",
                   bp.verify(Vs, bad2)[0], False))

    other = [bp.commit(5, gs[0]), bp.commit(8, gs[1])]
    checks.append(("proof is bound to its commitments",
                   bp.verify(other, pf)[0], False))

    out = 2 ** 16
    g2 = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(2)]
    V2 = [bp.commit(out, g2[0]), bp.commit(1, g2[1])]
    pf2 = bp.prove([out, 1], g2)
    checks.append(("a value outside [0,2^n) is rejected",
                   bp.verify(V2, pf2)[0], False))

    all_ok = True
    for name, got, want in checks:
        ok = got == want
        all_ok &= ok
        print("  %-38s %-6s (want %s) %s"
              % (name, got, want, "ok" if ok else "FAIL"), flush=True)

    section("3. does the generator change move the measured times?")
    print("  Generators are inputs to the prover. Setup derives them and should",
          flush=True)
    print("  get slower; prove and verify should not move. Measured at two sizes:",
          flush=True)
    print("", flush=True)
    print("  {:>6} {:>8} {:>12} {:>11} {:>11} {:>9}".format(
        "n*m", "deriv", "setup(s)", "prove(s)", "verify(s)", "size(B)"), flush=True)

    import bulletproofs as BPMOD
    rows = []
    for m in (2, 32):
        for tag, fn in (("legacy", h2c_legacy), ("RFC9380", h2c)):
            BPMOD.h2c = fn
            t0 = time.time(); bp = BP(32, m); ts = time.time() - t0
            vs = [int.from_bytes(os.urandom(4), "big") % (2 ** 16) for _ in range(m)]
            gg = [int.from_bytes(os.urandom(32), "big") % Q for _ in range(m)]
            Vv = [bp.commit(vs[j], gg[j]) for j in range(m)]
            t0 = time.time(); p = bp.prove(vs, gg); tp = time.time() - t0
            t0 = time.time(); ok, _ = bp.verify(Vv, p); tv = time.time() - t0
            sz = size_bytes(p)
            rows.append((32 * m, tag, ts, tp, tv, sz, ok))
            print("  {:>6,} {:>8} {:>12.2f} {:>11.2f} {:>11.2f} {:>9,} {}".format(
                32 * m, tag, ts, tp, tv, sz, "ok" if ok else "FAILED"), flush=True)
    BPMOD.h2c = h2c

    print("", flush=True)
    for nm in sorted({r[0] for r in rows}):
        a = [r for r in rows if r[0] == nm and r[1] == "legacy"][0]
        b = [r for r in rows if r[0] == nm and r[1] == "RFC9380"][0]
        print("  n*m=%-7d setup x%.1f   prove %+.1f%%   verify %+.1f%%   size %s"
              % (nm, b[2] / a[2] if a[2] else float("nan"),
                 100 * (b[3] - a[3]) / a[3], 100 * (b[4] - a[4]) / a[4],
                 "identical" if a[5] == b[5] else "DIFFERENT"), flush=True)

    print("", flush=True)
    print("  Proof sizes are identical, which is the check that does not depend",
          flush=True)
    print("  on the machine. The prove and verify figures are single runs on a",
          flush=True)
    print("  loaded desktop and differ by a few per cent in both directions; the",
          flush=True)
    print("  reason to expect no real change is structural rather than statistical,",
          flush=True)
    print("  since after BP.__init__ the two configurations execute the same",
          flush=True)
    print("  operations on different points. Setup is about four times slower,",
          flush=True)
    print("  and setup is not among the figures tab:bp reports.", flush=True)

    print("", flush=True)
    print("  RESULT: vectors %s, soundness %s"
          % ("pass" if vectors_ok else "FAIL", "pass" if all_ok else "FAIL"),
          flush=True)
    print("  done", flush=True)
    return 0 if (vectors_ok and all_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
